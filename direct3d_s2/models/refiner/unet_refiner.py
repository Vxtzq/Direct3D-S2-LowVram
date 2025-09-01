# -*- coding: utf-8 -*-
import itertools
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .unet3d import UNet3DModel
import trimesh
from tqdm import tqdm
from skimage import measure
from direct3d_s2.modules.utils import convert_module_to_f16, convert_module_to_f32
import direct3d_s2.modules.sparse as sp
import gc

def adaptive_conv(inputs, weights, patch=32):
    """
    FP16 + patchwise memory-efficient adaptive conv.

    Major changes:
     - Avoid expanding small weight vectors to full (D,H,W) before .half()
     - Pre-pad inputs externally (callers may pad once; here we handle by using a padded view)
     - Use small expand() views for constant-kernel case to avoid allocation
    """
    B, C, D, H, W = inputs.shape
    device = inputs.device
    dtype = torch.float16

    # Ensure FP16 input (inference)
    inputs = inputs.half()

    # If weights is per-voxel (B,27,D,H,W) keep as half but don't call .half() on an expanded tensor.
    if weights.ndim == 2:
        # weights: (B,27) -> small template of shape (B,27,1,1,1) kept in fp16
        w_template = weights.view(B, 27, 1, 1, 1).to(device=device, dtype=dtype)
        weights_full = None
    else:
        # weights is (B,27,D,H,W) or similar. Move to device but avoid extra copies when possible.
        weights_full = weights.to(device=device, dtype=dtype)
        w_template = None

    # Pre-pad the whole inputs once to avoid multiple F.pad inside inner loops.
    # pad order: (x_before, x_after, y_before, y_after, z_before, z_after)
    inputs_padded = F.pad(inputs, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

    out = torch.zeros_like(inputs, dtype=dtype, device=device)

    # Process in patches
    for z0 in range(0, D, patch):
        z1 = min(z0 + patch, D)
        # indices in padded volume need +1 offset because of padding
        z0p, z1p = z0 + 1, z1 + 1
        for y0 in range(0, H, patch):
            y1 = min(y0 + patch, H)
            y0p, y1p = y0 + 1, y1 + 1
            for x0 in range(0, W, patch):
                x1 = min(x0 + patch, W)
                x0p, x1p = x0 + 1, x1 + 1

                # Slice inputs from padded view so we don't call F.pad each time
                slice_in = inputs_padded[:, :, z0p - 1:z1p + 1, y0p - 1:y1p + 1, x0p - 1:x1p + 1]
                # slice_in shape: (B, C, (z1-z0)+2, (y1-y0)+2, (x1-x0)+2)
                out_d = z1 - z0
                out_h = y1 - y0
                out_w = x1 - x0
                slice_out = torch.zeros((B, C, out_d, out_h, out_w), dtype=dtype, device=device)

                idx_k = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            # take interior slices (no further padding needed)
                            s = slice_in[:, :, i:i + out_d, j:j + out_h, k:k + out_w]

                            if weights_full is not None:
                                # weights available per-voxel -> index patch
                                w_ = weights_full[:, idx_k, z0:z1, y0:y1, x0:x1].unsqueeze(1)
                            else:
                                # weights is template (B,27,1,1,1) -> expand to patch size (view, no alloc)
                                w_ = w_template[:, idx_k].expand(-1, -1, out_d, out_h, out_w)

                            slice_out += s * w_
                            idx_k += 1

                out[:, :, z0:z1, y0:y1, x0:x1] = slice_out

                # free locals (let python drop references)
                del slice_in, slice_out
    return out


def adaptive_block(inputs, conv, weights_=None, patch=32):
    """
    FP16 + patchwise adaptive block.
    Applies adaptive_conv_patchwise 3 times.

    Changes:
     - Avoid creating full-size expanded weights.
     - If conv returns per-voxel weights, keep them as small/half and let adaptive_conv slice.
    """
    # conv may produce per-voxel weights or per-sample (B,27)
    if weights_ is not None:
        # Usually small (B, n) or (B,27,...) depending on conv
        weights = conv(weights_)
    else:
        weights = conv(inputs)

    # Move weights to half but keep shape minimal (no full D*H*W expansion)
    weights = weights.half()

    # normalize in FP16 (normalize along kernel dim)
    weights = F.normalize(weights, dim=1, p=1)

    out = inputs.half()
    for _ in range(3):
        out = adaptive_conv(out, weights, patch=patch)

    # free
    del weights
    return out


class GeoDecoder(nn.Module):

    def __init__(self, 
                 n_features: int,
                 hidden_dim: int = 32, 
                 num_layers: int = 4, 
                 use_sdf: bool = False,
                 activation: nn.Module = nn.ReLU):
        super().__init__()
        self.use_sdf=use_sdf
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 8),
        )

        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        return x



class Voxel_RefinerXL(nn.Module):
    def __init__(self,
                in_channels: int = 1,
                out_channels: int = 1,
                layers_per_block: int = 2,
                layers_mid_block: int = 2,
                patch_size: int = 192,
                res: int = 512,
                use_checkpoint: bool=False,
                use_fp16: bool = False):

        super().__init__()

        self.unet3d1 = UNet3DModel(in_channels=16, out_channels=8, use_conv_out=False,
                                   layers_per_block=layers_per_block, layers_mid_block=layers_mid_block, 
                                   block_out_channels=(8, 32, 128,512), norm_num_groups=4, use_checkpoint=use_checkpoint)
        self.conv_in = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
        self.latent_mlp = GeoDecoder(32)
        self.adaptive_conv1 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.adaptive_conv2 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.adaptive_conv3 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.mid_conv = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.conv_out = nn.Conv3d(8, out_channels, kernel_size=3, padding=1)
        self.patch_size = patch_size
        self.res = res

        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        # self.blocks.apply(convert_module_to_f16)
        self.apply(convert_module_to_f16)

    def run(self,
            reconst_x,
            feat, 
            mc_threshold=0,
        ):
        batch_size = int(reconst_x.coords[..., 0].max()) + 1
        sparse_sdf, sparse_index = reconst_x.feats, reconst_x.coords
        sparse_feat = feat.feats
        device = sparse_sdf.device
        dtype = sparse_sdf.dtype
        res = self.res

        sdfs = []
        for i in range(batch_size):
            idx = sparse_index[..., 0] == i
            sparse_sdf_i, sparse_index_i = sparse_sdf[idx].squeeze(-1),  sparse_index[idx][..., 1:]
            sdf = torch.ones((res, res, res)).to(device).to(dtype)
            sdf[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = sparse_sdf_i
            sdfs.append(sdf.unsqueeze(0))

        sdfs = torch.stack(sdfs, dim=0)
        feats = torch.zeros((batch_size, sparse_feat.shape[-1], res, res, res), 
                            device=device, dtype=dtype)
        feats[sparse_index[...,0],:,sparse_index[...,1],sparse_index[...,2],sparse_index[...,3]] = sparse_feat
        
        N = sdfs.shape[0]
        outputs = torch.ones([N,1,res,res,res], dtype=dtype, device=device)
        stride = 160
        patch_size = self.patch_size
        step = 3
        sdfs = sdfs.to(dtype)
        feats = feats.to(dtype)
        patchs=[]
        for i in range(step):
            for j in range(step):
                for k in tqdm(range(step)):
                    sdf = sdfs[:, :, stride * i: stride * i + patch_size,
                               stride * j: stride * j + patch_size,
                               stride * k: stride * k + patch_size]
                    crop_feats = feats[:, :, stride * i: stride * i + patch_size, 
                                       stride * j: stride * j + patch_size, 
                                       stride * k: stride * k + patch_size]
                    inputs = self.conv_in(sdf)
                    crop_feats = self.latent_mlp(crop_feats.permute(0,2,3,4,1)).permute(0,4,1,2,3)
                    inputs = torch.cat([inputs, crop_feats],dim=1)
                    mid_feat = self.unet3d1(inputs)  
                    mid_feat = adaptive_block(mid_feat, self.adaptive_conv1)
                    mid_feat = self.mid_conv(mid_feat)
                    mid_feat = adaptive_block(mid_feat, self.adaptive_conv2)
                    final_feat = self.conv_out(mid_feat)
                    final_feat = adaptive_block(final_feat, self.adaptive_conv3, weights_=mid_feat)
                    output = F.tanh(final_feat)
                    patchs.append(output)
        weights = torch.linspace(0, 1, steps=32, device=device, dtype=dtype)
        lines=[]
        for i in range(9):
            out1 = patchs[i * 3]
            out2 = patchs[i * 3 + 1]
            out3 = patchs[i * 3 + 2]
            line = torch.ones([N, 1, 192, 192,res], dtype=dtype, device=device) * 2
            line[:, :, :, :, :160] = out1[:, :, :, :, :160]
            line[:, :, :, :, 192:320] = out2[:, :, :, :, 32:160]
            line[:, :, :, :, 352:] = out3[:, :, :, :, 32:]
            
            line[:,:,:,:,160:192] = out1[:,:,:,:,160:] * (1-weights.reshape(1,1,1,1,-1)) + out2[:,:,:,:,:32] * weights.reshape(1,1,1,1,-1)
            line[:,:,:,:,320:352] = out2[:,:,:,:,160:] * (1-weights.reshape(1,1,1,1,-1)) + out3[:,:,:,:,:32] * weights.reshape(1,1,1,1,-1)
            lines.append(line)
        layers=[]
        for i in range(3):
            line1 = lines[i*3]
            line2 = lines[i*3+1]
            line3 = lines[i*3+2]
            layer = torch.ones([N,1,192,res,res], device=device, dtype=dtype) * 2
            layer[:,:,:,:160] = line1[:,:,:,:160]
            layer[:,:,:,192:320] = line2[:,:,:,32:160]
            layer[:,:,:,352:] = line3[:,:,:,32:]
            layer[:,:,:,160:192] = line1[:,:,:,160:]*(1-weights.reshape(1,1,1,-1,1))+line2[:,:,:,:32]*weights.reshape(1,1,1,-1,1)
            layer[:,:,:,320:352] = line2[:,:,:,160:]*(1-weights.reshape(1,1,1,-1,1))+line3[:,:,:,:32]*weights.reshape(1,1,1,-1,1)
            layers.append(layer)
        outputs[:,:,:160] = layers[0][:,:,:160]
        outputs[:,:,192:320] = layers[1][:,:,32:160]
        outputs[:,:,352:] = layers[2][:,:,32:]
        outputs[:,:,160:192] = layers[0][:,:,160:]*(1-weights.reshape(1,1,-1,1,1))+layers[1][:,:,:32]*weights.reshape(1,1,-1,1,1)
        outputs[:,:,320:352] = layers[1][:,:,160:]*(1-weights.reshape(1,1,-1,1,1))+layers[2][:,:,:32]*weights.reshape(1,1,-1,1,1)
        # outputs = -outputs

        meshes = []
        for i in range(outputs.shape[0]):
            vertices, faces, _, _ = measure.marching_cubes(outputs[i, 0].cpu().numpy(), level=mc_threshold, method='lewiner')
            vertices = vertices / res * 2 - 1
            meshes.append(trimesh.Trimesh(vertices, faces))
        
        return meshes


class Voxel_RefinerXL_sign(nn.Module):
    def __init__(self,
                in_channels: int=1,
                out_channels: int=1,
                layers_per_block: int=2,
                layers_mid_block: int=2,
                patch_size: int=192,
                res: int=512,
                infer_patch_size: int=192,
                use_checkpoint: bool=False,
                use_fp16: bool = False):
        super().__init__()

        self.unet3d1 = UNet3DModel(in_channels=8, out_channels=8, use_conv_out=False, 
                                   layers_per_block=layers_per_block, layers_mid_block=layers_mid_block, 
                                   block_out_channels=(8,32,128,512), norm_num_groups=4, use_checkpoint=use_checkpoint)
        self.conv_in = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
        self.conv_out = nn.Conv3d(8, out_channels, kernel_size=3, padding=1)
        self.downsample = sp.SparseDownsample(factor=2)
        self.patch_size = patch_size
        self.infer_patch_size = infer_patch_size
        self.res = res
       
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        self.apply(convert_module_to_f16)
    
    def run(self,
             reconst_x=None,
             feat=None, 
             mc_threshold=0,
        ):
        batch_size = int(reconst_x.coords[..., 0].max()) + 1
        sparse_sdf, sparse_index = reconst_x.feats, reconst_x.coords
        device = sparse_sdf.device
        voxel_resolution = 1024
        sdfs=[]
        for i in range(batch_size):
            idx = sparse_index[..., 0] == i
            sparse_sdf_i, sparse_index_i = sparse_sdf[idx].squeeze(-1),  sparse_index[idx][..., 1:]
            sdf = torch.ones((voxel_resolution, voxel_resolution, voxel_resolution)).to(device).to(sparse_sdf_i.dtype)
            sdf[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = sparse_sdf_i
            sdfs.append(sdf.unsqueeze(0))

        sdfs1024 = torch.stack(sdfs,dim=0)
        reconst_x1024 = reconst_x
        reconst_x = self.downsample(reconst_x)
        batch_size = int(reconst_x.coords[..., 0].max()) + 1
        sparse_sdf, sparse_index = reconst_x.feats, reconst_x.coords
        device = sparse_sdf.device
        dtype = sparse_sdf.dtype
        voxel_resolution = 512
        sdfs = torch.ones((batch_size, voxel_resolution, voxel_resolution, voxel_resolution),device=device, dtype=sparse_sdf.dtype)
        sdfs[sparse_index[...,0],sparse_index[...,1],sparse_index[...,2],sparse_index[...,3]] = sparse_sdf.squeeze(-1)
        sdfs = sdfs.unsqueeze(1)
        
        N = sdfs.shape[0]
        outputs = torch.ones([N,1,512,512,512],device=sdfs.device, dtype=dtype)
        stride = 128
        patch_size = self.patch_size
        step = 3
        for i in range(step):
            for j in range(step):
                for k in tqdm(range(step)):
                    sdf = sdfs[:,:,stride*i:stride*i+patch_size,stride*j:stride*j+patch_size,stride*k:stride*k+patch_size]
                    inputs = self.conv_in(sdf)
                    mid_feat = self.unet3d1(inputs)  
                    final_feat = self.conv_out(mid_feat)
                    output = F.sigmoid(final_feat)
                    output[output>=0.5] = 1
                    output[output<0.5] = -1
                    outputs[:, :, stride*i:stride*i+patch_size, stride*j:stride*j+patch_size, stride*k:stride*k+patch_size] = output
        outputs = outputs.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        sdfs = sdfs1024.clone()
        sdfs = sdfs.abs()*outputs
        
        sparse_index1024 = reconst_x1024.coords
        
        sdfs[sparse_index1024[...,0], :, sparse_index1024[...,1], sparse_index1024[...,2],sparse_index1024[...,3]] = sdfs1024[sparse_index1024[...,0], :, sparse_index1024[...,1], sparse_index1024[...,2], sparse_index1024[...,3]]
        outputs = sdfs.cpu().numpy()
        grid_size = outputs.shape[2]

        meshes = []
        for i in range(outputs.shape[0]):
            outputs_torch = outputs[i,0]
            vertices, faces, _, _ = measure.marching_cubes(outputs_torch, level=mc_threshold, method="lewiner")
            vertices = vertices / grid_size * 2 - 1
            meshes.append(trimesh.Trimesh(vertices, faces))
        return meshes
