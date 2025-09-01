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
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        layers_per_block: int = 2,
        layers_mid_block: int = 2,
        patch_size: int =64,
        res: int = 512,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
    ):
        super().__init__()

        self.unet3d1 = UNet3DModel(
            in_channels=16,
            out_channels=8,
            use_conv_out=False,
            layers_per_block=layers_per_block,
            layers_mid_block=layers_mid_block,
            block_out_channels=(8, 32, 128, 512),
            norm_num_groups=4,
            use_checkpoint=use_checkpoint,
        )
        """
        self.unet3d1.set_chunking_params(
            chunk_size=64,        # 4x larger chunks  
            overlap=12
         )
        """
        self.conv_in = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
        self.latent_mlp = GeoDecoder(32)
        self.adaptive_conv1 = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1), nn.ReLU(), nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False)
        )
        self.adaptive_conv2 = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1), nn.ReLU(), nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False)
        )
        self.adaptive_conv3 = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1), nn.ReLU(), nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False)
        )
        self.mid_conv = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.conv_out = nn.Conv3d(8, out_channels, kernel_size=3, padding=1)

        self.patch_size = patch_size
        self.res = res

        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # If requested, convert *entire module* to fp16. Use .half() so *all* submodules/params/buffers
        # are cast consistently. This avoids dtype mismatches.
        if use_fp16:
            self.half()
    def to_cpu(self):
        """
        Move the entire module to CPU and convert all FP16/FP32 parameters and buffers to float32.
        Clears cached GPU memory.
        """
        def recursive_cpu(module):
            # Move parameters and buffers to CPU as float32
            for name, param in module._parameters.items():
                if param is not None:
                    module._parameters[name] = param.detach().to("cpu", dtype=torch.float32)
            for name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[name] = buf.detach().to("cpu", dtype=torch.float32)
            # Recursively handle child modules
            for child in module.children():
                recursive_cpu(child)

        recursive_cpu(self)
        # Ensure module itself is on CPU
        self.cpu()
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    def to_device(self, device: torch.device):
        """Move entire module to device (keeps dtype)."""
        self.to(device)
        return self

    def run(self, reconst_x, feat, mc_threshold=0):
        """
        VRAM-optimized refiner forward: sparse inputs, FP16 on GPU, patchwise streaming to CPU.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(device).eval()

        model_dtype = torch.float16 if self.use_fp16 else torch.float32

        # Move sparse data to device
        sparse_sdf = reconst_x.feats.to(device, dtype=model_dtype)
        sparse_index = reconst_x.coords.to(device)
        sparse_feat = feat.feats.to(device, dtype=model_dtype)
        batch_size = int(sparse_index[..., 0].max()) + 1

        patch_size = self.patch_size
        stride = 160
        step = 3
        feat_dim = sparse_feat.shape[-1]

        # CPU accumulator
        outputs = torch.zeros((batch_size, 1, self.res, self.res, self.res),
                              dtype=torch.float32, device="cpu")

        # Preallocate patch tensors once
        sdf_tensor = torch.empty((1, 1, patch_size, patch_size, patch_size), device=device, dtype=model_dtype)
        feat_tensor = torch.empty((1, feat_dim, patch_size, patch_size, patch_size), device=device, dtype=model_dtype)
        offset_buf = torch.zeros(3, device=device, dtype=torch.long)

        if self.use_fp16:
            self.latent_mlp = self.latent_mlp.half()

        with torch.inference_mode():
            for b in range(batch_size):
                idx = sparse_index[..., 0] == b
                coords_b = sparse_index[idx][:, 1:]
                sdf_b = sparse_sdf[idx].squeeze(-1)
                feat_b = sparse_feat[idx]

                for i in range(step):
                    for j in range(step):
                        for k in range(step):
                            x0, y0, z0 = stride*i, stride*j, stride*k
                            mask = (
                                (coords_b[:, 0] >= x0) & (coords_b[:, 0] < x0 + patch_size) &
                                (coords_b[:, 1] >= y0) & (coords_b[:, 1] < y0 + patch_size) &
                                (coords_b[:, 2] >= z0) & (coords_b[:, 2] < z0 + patch_size)
                            )
                            if not mask.any():
                                continue

                            coords_patch = coords_b[mask].long()
                            offset_buf[:] = torch.tensor([x0, y0, z0], device=device)
                            coords_patch.sub_(offset_buf)

                            sdf_tensor.fill_(1.0)
                            feat_tensor.zero_()
                            sdf_tensor[0, 0, coords_patch[:,0], coords_patch[:,1], coords_patch[:,2]] = sdf_b[mask]
                            feat_tensor[0, :, coords_patch[:,0], coords_patch[:,1], coords_patch[:,2]] = feat_b[mask].T

                            # Forward pass in fp32
                            feat_latent = self.latent_mlp(feat_tensor.permute(0,2,3,4,1)).permute(0,4,1,2,3)
                            x_in = torch.cat([self.conv_in(sdf_tensor), feat_latent], dim=1)

                            mid_feat = self.unet3d1(x_in)
                            mid_feat = adaptive_block(mid_feat, self.adaptive_conv1, patch=patch_size)
                            mid_feat = self.mid_conv(mid_feat)
                            mid_feat = adaptive_block(mid_feat, self.adaptive_conv2, patch=patch_size)
                            final_feat = self.conv_out(mid_feat)
                            final_feat = adaptive_block(final_feat, self.adaptive_conv3, weights_=mid_feat, patch=patch_size)

                            # Stream to CPU as float32
                            out_patch = final_feat[0:1].detach().to("cpu", dtype=torch.float32)
                            _, _, px, py, pz = out_patch.shape
                            outputs[b, :, x0:x0+px, y0:y0+py, z0:z0+pz] = out_patch[:, :1]

                            del feat_latent, x_in, mid_feat, final_feat, out_patch

                del coords_b, sdf_b, feat_b, idx
                torch.cuda.empty_cache()



        # Marching cubes (CPU)
        meshes = []
        for b in range(batch_size):
            out_np = outputs[b,0].numpy()
            verts, faces, _, _ = measure.marching_cubes(out_np, level=mc_threshold, method="lewiner")
            verts = verts / float(self.res) * 2.0 - 1.0
            meshes.append(trimesh.Trimesh(verts, faces))
            del out_np, verts, faces
            gc.collect()

        del sparse_sdf, sparse_feat, sparse_index, outputs, sdf_tensor, feat_tensor
        torch.cuda.empty_cache()
        gc.collect()

        return meshes





class Voxel_RefinerXL_sign(nn.Module):
    def __init__(self,
                in_channels: int=1,
                out_channels: int=1,
                layers_per_block: int=2,
                layers_mid_block: int=2,
                patch_size: int=64,
                res: int=512,
                infer_patch_size: int=64,
                use_checkpoint: bool=False,
                use_fp16: bool=True):
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
    def to_device(self, device: torch.device):
        """Move entire module to device (keeps dtype)."""
        self.to(device)
        return self
    def to_cpu(self):
        """
        Move module parameters and buffers to CPU (float32) in-place, clear grads and cached CUDA tensors.
        """
        for module in self.modules():
            # parameters
            for p_name, p in list(module._parameters.items()):
                if p is None:
                    continue
                # move param tensor data to cpu and replace in-place
                p_data = p.data.detach().to("cpu", dtype=torch.float32)
                # replace Parameter with new CPU tensor wrapped as Parameter
                new_p = torch.nn.Parameter(p_data, requires_grad=p.requires_grad)
                module._parameters[p_name] = new_p
                # clear grad to remove GPU ref
                if p.grad is not None:
                    try:
                        p.grad = None
                    except Exception:
                        pass

            # buffers
            for b_name, buf in list(module._buffers.items()):
                if buf is None:
                    continue
                module._buffers[b_name] = buf.detach().to("cpu", dtype=torch.float32)

        # ensure module itself moved to CPU
        self.cpu()
        # clear cache and gc
        gc.collect()
        torch.cuda.empty_cache()
        return self


    def convert_to_fp16(self):
        """Convert model to FP16, but keep BatchNorm in FP32."""
        def recursive_half(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.float()  # keep BN in FP32
            else:
                module.half()
            for child in module.children():
                recursive_half(child)
        recursive_half(self)

    def run(self, reconst_x=None, feat=None, mc_threshold=0):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dtype = self.dtype
        cpu_dtype = torch.float16 if self.use_fp16 else torch.float32

        sparse_sdf, sparse_index = reconst_x.feats.to(device, dtype=dtype), reconst_x.coords.to(device)
        batch_size = int(sparse_index[..., 0].max()) + 1
        voxel_resolution = 1024
        patch_size = self.patch_size
        stride = 128
        step = (voxel_resolution + stride - 1) // stride

        # CPU accumulator
        sdfs_cpu = torch.ones((batch_size, 1, voxel_resolution, voxel_resolution, voxel_resolution),
                              dtype=cpu_dtype, device="cpu")

        # Preallocate GPU patch tensor once
        patch_tensor = torch.ones((1, 1, patch_size, patch_size, patch_size), device=device, dtype=dtype)
        offset_buf = torch.zeros(3, device=device, dtype=torch.long)

        self.to(device).eval()

        with torch.inference_mode():
            for b in range(batch_size):
                idx = sparse_index[..., 0] == b
                coords_b = sparse_index[idx][:, 1:]
                sdf_b = sparse_sdf[idx].squeeze(-1)

                sdfs_cpu[b, 0, coords_b[:,0].cpu(), coords_b[:,1].cpu(), coords_b[:,2].cpu()] = sdf_b.cpu()

                for i in range(step):
                    for j in range(step):
                        for k in range(step):
                            start = torch.tensor([stride*i, stride*j, stride*k], device=device)
                            end = torch.min(start + patch_size, torch.tensor([voxel_resolution]*3, device=device))

                            mask = (
                                (coords_b[:,0] >= start[0]) & (coords_b[:,0] < end[0]) &
                                (coords_b[:,1] >= start[1]) & (coords_b[:,1] < end[1]) &
                                (coords_b[:,2] >= start[2]) & (coords_b[:,2] < end[2])
                            )
                            if not mask.any():
                                continue

                            coords_patch = coords_b[mask].long()
                            offset_buf[:] = start
                            local_coords = coords_patch - offset_buf

                            patch_tensor.fill_(1.0)
                            patch_tensor[0,0,local_coords[:,0],local_coords[:,1],local_coords[:,2]] = sdf_b[mask]

                            # Forward
                            x_in = self.conv_in(patch_tensor)
                            mid_feat = self.unet3d1(x_in)
                            final_feat = self.conv_out(mid_feat)
                            output = torch.sign(final_feat)

                            # Stream to CPU
                            out_slice = output[0,0,:end[0]-start[0],:end[1]-start[1],:end[2]-start[2]].cpu()
                            sdfs_cpu[b,0,start[0]:end[0],start[1]:end[1],start[2]:end[2]] = out_slice

                            del x_in, mid_feat, final_feat, output, local_coords, out_slice

                del coords_b, sdf_b, idx
                torch.cuda.empty_cache()

        # Marching cubes (CPU)
        meshes = []
        for b in range(batch_size):
            out_np = sdfs_cpu[b,0].float().numpy()
            verts, faces, _, _ = measure.marching_cubes(out_np, level=mc_threshold, method="lewiner")
            verts = verts / voxel_resolution * 2 - 1
            meshes.append(trimesh.Trimesh(verts, faces))

        del sdfs_cpu, sparse_sdf, sparse_index
        torch.cuda.empty_cache()
        gc.collect()

        return meshes



