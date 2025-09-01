import torch
import numpy as np  
import udf_ext


def compute_valid_udf(vertices, faces, dim=512, threshold=8.0):
    if not faces.is_cuda or not vertices.is_cuda:
        raise ValueError("Both maze and visited tensors must be CUDA tensors")
    
    # Allocate as fp16
    udf = torch.zeros(dim**3, device=vertices.device, dtype=torch.half) + 10000000.0
    n_faces = faces.shape[0]

    # udf_ext likely expects int32, so compute in int32 and cast later
    udf_int = udf.to(torch.int32)
    udf_ext.compute_valid_udf(vertices, faces, udf_int, n_faces, dim, threshold)

    # Return fp16
    return udf_int.to(torch.half) / 10000000.0

def normalize_mesh(mesh, scale=0.95):
    vertices = mesh.vertices
    min_coords, max_coords = vertices.min(axis=0), vertices.max(axis=0)
    dxyz = max_coords - min_coords
    dist = max(dxyz)
    mesh_scale = 2.0 * scale / dist
    mesh_offset = -(min_coords + max_coords) / 2
    vertices = (vertices + mesh_offset) * mesh_scale
    mesh.vertices = vertices
    return mesh

def mesh2index(mesh, size=1024, factor=8):
    vertices = torch.Tensor(mesh.vertices).float().cuda() * 0.5
    faces = torch.Tensor(mesh.faces).int().cuda()
    sdf = compute_valid_udf(vertices, faces, dim=size, threshold=4.0)
    sdf = sdf.reshape(size, size, size).unsqueeze(0)

    sparse_index = (sdf < 4/size).nonzero()
    sparse_index[..., 1:] = sparse_index[..., 1:] // factor
    latent_index = torch.unique(sparse_index, dim=0)
    return latent_index
    

import torch
import numpy as np  
import udf_ext

"""
def compute_valid_udf_blocked(vertices, faces, dim=512, threshold=8.0, block_size=128):
    
    # Allocate on CPU
    udf = torch.full((dim**3,), 10000000, dtype=torch.int32)
    n_faces = faces.shape[0]

    # Move vertices/faces to CPU if needed
    if vertices.is_cuda:
        vertices = vertices.cpu()
    if faces.is_cuda:
        faces = faces.cpu()

    # Process in blocks
    for start in range(0, dim**3, block_size**3):
        end = min(start + block_size**3, dim**3)
        udf_block = udf[start:end]
        # Call the original udf_ext function for this block
        udf_ext.compute_valid_udf(vertices, faces, udf_block, n_faces, dim, threshold)
        udf[start:end] = udf_block

    return udf.float() / 1e7

def normalize_mesh(mesh, scale=0.95):
    vertices = mesh.vertices
    min_coords, max_coords = vertices.min(axis=0), vertices.max(axis=0)
    dxyz = max_coords - min_coords
    dist = max(dxyz)
    mesh_scale = 2.0 * scale / dist
    mesh_offset = -(min_coords + max_coords) / 2
    vertices = (vertices + mesh_offset) * mesh_scale
    mesh.vertices = vertices
    return mesh

def mesh2index(mesh, size=1024, factor=8):
    vertices = torch.Tensor(mesh.vertices).float() * 0.5
    faces = torch.Tensor(mesh.faces).int()
    sdf = compute_valid_udf_blocked(vertices, faces, dim=size, threshold=4.0)
    sdf = sdf.reshape(size, size, size).unsqueeze(0)

    sparse_index = (sdf < 4/size).nonzero()
    sparse_index[..., 1:] = sparse_index[..., 1:] // factor
    latent_index = torch.unique(sparse_index, dim=0)
    return latent_index
"""
