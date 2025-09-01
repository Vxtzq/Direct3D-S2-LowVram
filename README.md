# Direct3D‑S2-LowVram: Gigascale 3D Generation Made Easy with Spatial Sparse Attention on Lower GPU

<div align="center">
  <a href="https://www.neural4d.com/research/direct3d-s2" target="_blank"><img src="https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome" height="22px"></a>
  <a href="https://huggingface.co/spaces/wushuang98/Direct3D-S2-v1.0-demo" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg" height="22px"></a>
  <a href="https://huggingface.co/spaces/wushuang98/Direct3D-S2-v1.0-demo" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg" height="22px"></a>
  <a href="https://arxiv.org/pdf/2505.17412" target="_blank"><img src="https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv" height="22px"></a>
</div>

<div align="center" style="background: #fff; box-shadow: 0 4px 12px rgba(0,0,0,.15); display: inline-block; padding: 8px;">
  <img id="teaser" src="assets/teaserv6.png" alt="Teaser image of Direct3D-S2"/>
</div>

---
# Easy install 
### Requirements : 

- *torch>2.0.0*
- *cuda 11.8* or above

### Install dependencies:

  ```bash
  git clone https://github.com/DreamTechAI/Direct3D-S2.git
  
  cd Direct3D-S2
  
  pip install -r requirements.txt
  
  pip install -e .
  
  ```
### Install torchsparse
  ```bash

  export MAX_JOBS=4
  pip install git+https://github.com/Deathdadev/torchsparse.git
  ```

### Install submodules 
  ```bash
  cd third_party
  cd voxelize
  python setup.py install
  ```
# Easy usage
  ```
  python inference.py
  ```

# Changes

This is an **optimized version** of the original repo [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2) from DreamTechAI, designed to **work with ~3x less VRAM** at the cost of some inference speed.

**Key modifications:**
- Use of **fp16** almost everywhere in the pipeline  
- **Chunked processing** for some modules to reduce the need for huge 3D arrays  

---

## Original repo VRAM requirements

| Resolution | VRAM Requirements | Inference Time |
|------------|-----------------|----------------|
| 512        | 10 GB           | unspecified    |
| 1024       | 24 GB           | unspecified    |

---

## This repo VRAM requirements

**Without refiner:**

| Resolution | VRAM Requirements | Inference Time |
|------------|-----------------|----------------|
| 512        | 4 GB            |
| 1024       | 8 GB            |

**With refiner:**

| Resolution | VRAM Requirements | Inference Time |
|------------|-----------------|----------------|
| 1080p      | 6 GB            |
| 4K         | 8 GB            |

> ⚠️ Note: Using the refiner decreases inference speed due to patched processing of voxels.

---

## ✨ Credits
- Original Direct3D-S2 Implementation: [DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)  
- Thanks to the **Direct3D-S2 authors** for their work and inspiration

---

# Original repository description

## 📝 Abstract

Generating high-resolution 3D shapes using volumetric representations such as Signed Distance Functions (SDFs) presents substantial computational and memory challenges. We introduce <strong class="has-text-weight-bold">Direct3D‑S2</strong>, a scalable 3D generation framework based on sparse volumes that achieves superior output quality with dramatically reduced training costs. Our key innovation is the <strong class="has-text-weight-bold">Spatial Sparse Attention (SSA)</strong> mechanism, which greatly enhances the efficiency of Diffusion Transformer (DiT) computations on sparse volumetric data. SSA allows the model to effectively process large token sets within sparse volumes, substantially reducing computational overhead and achieving a <em>3.9&times;</em> speedup in the forward pass and a <em>9.6&times;</em> speedup in the backward pass. Our framework also includes a variational autoencoder (VAE) that maintains a consistent sparse volumetric format across input, latent, and output stages. Compared to previous methods with heterogeneous representations in 3D VAE, this unified design significantly improves training efficiency and stability. Our model is trained on public available datasets, and experiments demonstrate that <strong class="has-text-weight-bold">Direct3D‑S2</strong> not only surpasses state-of-the-art methods in generation quality and efficiency, but also enables <strong class="has-text-weight-bold">training at 1024<sup>3</sup>  resolution with just 8 GPUs</strong>, a task typically requiring at least 32 GPUs for volumetric representations at 256<sup>3</sup> resolution, thus making gigascale 3D generation both practical and accessible.

## 🌟 Highlight

- **Gigascale 3D Generation**: Direct3D-S2 enables training at 1024<sup>3</sup> resolution with only 8 GPUs.
- **Spatial Sparse Attention (SSA)**: A novel attention mechanism designed for sparse volumetric data, enabling efficient processing of large token sets.
- **Unified Sparse VAE**: A variational autoencoder that maintains a consistent sparse volumetric format across input, latent, and output stages, improving training efficiency and stability.

## 🚀 Getting Started

### Installation

### ✅ Tested Environment

> 💡 *If you're setting up on Windows, check out [issue #11](https://github.com/DreamTechAI/Direct3D-S2/issues/11) and [issue #12](https://github.com/DreamTechAI/Direct3D-S2/issues/12). Big thanks to the contributors who helped get Direct3D-S2 working on Windows!*

- **System**: Ubuntu 22.04  
- **CUDA Toolkit**: [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)  
- **PyTorch**: Install `torch` and `torchvision` first.  
  Make sure the PyTorch CUDA version matches your installed CUDA Toolkit.

  ```bash
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  ```
- **Torchsparse**:https://github.com/DreamTechAI/Direct3D-S2
  follow the [offical guide](https://github.com/mit-han-lab/torchsparse) or:
  
  ```bash
  git clone https://github.com/mit-han-lab/torchsparse
  cd torchsparse && python -m pip install .
  ```

- Install dependencies:

  ```bash
  git clone https://github.com/DreamTechAI/Direct3D-S2.git
  
  cd Direct3D-S2
  
  pip install -r requirements.txt
  
  pip install -e .
  
  ```

### Usage

```python
from direct3d_s2.pipeline import Direct3DS2Pipeline
pipeline = Direct3DS2Pipeline.from_pretrained(
  'wushuang98/Direct3D-S2', 
  subfolder="direct3d-s2-v-1-1"
)
pipeline.to("cuda:0")

mesh = pipeline(
  'assets/test/13.png', 
  sdf_resolution=1024, # 512 or 1024
  remove_interior=True,
  remesh=False, # Switch to True if you need to reduce the number of triangles.
)["mesh"]

mesh.export('output.obj')
```

### Web Demo

We provide a Gradio web demo for Direct3D-S2, which allows you to generate 3D meshes from images interactively.

```bash
python app.py
```

## 🤗 Acknowledgements

Thanks to the following repos for their great work, which helps us a lot in the development of Direct3D-S2:

- [Trellis](https://github.com/microsoft/TRELLIS)
- [SparseFlex](https://github.com/VAST-AI-Research/TripoSF)
- [native-sparse-attention-triton](https://github.com/XunhaoLai/native-sparse-attention-triton)
- [diffusers](https://github.com/huggingface/diffusers)

## 📄 License

Direct3D-S2 is released under the MIT License. See [LICENSE](LICENSE) for details.

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{wu2025direct3ds2gigascale3dgeneration,
  title={Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention}, 
  author={Shuang Wu and Youtian Lin and Feihu Zhang and Yifei Zeng and Yikang Yang and Yajie Bao and Jiachen Qian and Siyu Zhu and Philip Torr and Xun Cao and Yao Yao},
  journal={arXiv preprint arXiv:2505.17412},
  year={2025}
}
```
