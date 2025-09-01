import os
import torch
import numpy as np
from typing import Any, Union, List, Optional, Sequence
from contextlib import nullcontext
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import gc
from direct3d_s2.modules import sparse as sp
from direct3d_s2.utils import (
    instantiate_from_config,
    preprocess_image,
    sort_block,
    extract_tokens_and_coords,
    normalize_mesh,
    mesh2index,
)


class Direct3DS2Pipeline(object):

    def __init__(self,
                 dense_vae,
                 dense_dit,
                 sparse_vae_512,
                 sparse_dit_512,
                 sparse_vae_1024,
                 sparse_dit_1024,
                 refiner,
                 refiner_1024,
                 dense_image_encoder,
                 sparse_image_encoder,
                 dense_scheduler,
                 sparse_scheduler_512,
                 sparse_scheduler_1024,
                 dtype=torch.float16,  # default fp16 on CUDA
                 ):
        self.dense_vae = dense_vae
        self.dense_dit = dense_dit
        self.sparse_vae_512 = sparse_vae_512
        self.sparse_dit_512 = sparse_dit_512
        self.sparse_vae_1024 = sparse_vae_1024
        self.sparse_dit_1024 = sparse_dit_1024
        self.refiner = refiner
        self.refiner_1024 = refiner_1024
        self.dense_image_encoder = dense_image_encoder
        self.sparse_image_encoder = sparse_image_encoder
        self.dense_scheduler = dense_scheduler
        self.sparse_scheduler_512 = sparse_scheduler_512
        self.sparse_scheduler_1024 = sparse_scheduler_1024
        self.dtype = dtype
        self.device = torch.device("cpu")  # default until .to() is called
        self._active_models = set()
    # ----------------------
    # Device / dtype helpers
    # ----------------------
    def _autocast_ctx(self):
        # Use autocast only on CUDA; on CPU it's a nullcontext
        return torch.autocast(device_type="cuda", dtype=self.dtype) if self.device.type == "cuda" else nullcontext()

    def _set_model_device(self, model, device: torch.device, prefer_half: bool = True):
        """Enhanced model device management with proper cleanup"""
        if model is None:
            return
            
        try:
            # Clear any existing GPU memory first
            if hasattr(model, 'cpu'):
                model.cpu()
            torch.cuda.empty_cache()
            
            model.to(device)
            
            if device.type == "cuda" and prefer_half:
                try:
                    model.half()
                    self._active_models.add(id(model))
                except Exception as e:
                    print(f"Warning: Could not convert model to half precision: {e}")
            else:
                try:
                    model.float()
                    if id(model) in self._active_models:
                        self._active_models.remove(id(model))
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Could not move model to {device}: {e}")

    def _offload_models(self, models: Sequence[Any]):
        """Enhanced model offloading with proper cleanup"""
        for m in models:
            if m is None:
                continue
            try:
                # Special handling for refiners with custom methods
                if hasattr(m, 'to_cpu'):
                    m.to_cpu()
                else:
                    m.cpu()
                    
                # Always convert to float32 on CPU for stability
                if hasattr(m, 'float'):
                    m.float()
                    
                # Remove from active tracking
                if id(m) in self._active_models:
                    self._active_models.remove(id(m))
                    
            except Exception as e:
                print(f"Warning: Could not offload model: {e}")
        
        # Aggressive cleanup
        torch.cuda.empty_cache()
        gc.collect()
    def _free_gpu_aggressive(self, local_vars=None, delete_vars=None):
        """More aggressive GPU memory cleanup"""
        # Delete specified variables
        if local_vars and delete_vars:
            for var_name in delete_vars:
                if var_name in local_vars:
                    try:
                        del local_vars[var_name]
                    except:
                        pass
        
        # Clear autocast cache
        if torch.cuda.is_available():
            try:
                torch.clear_autocast_cache()
            except:
                pass
                
        # Multiple cleanup passes
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
    def _force_offload_all_models(self):
        """Nuclear option: force all models to CPU and clear all GPU memory"""
        all_models = [
            self.dense_vae, self.dense_dit, self.dense_image_encoder, self.dense_scheduler,
            self.sparse_vae_512, self.sparse_dit_512, self.sparse_image_encoder, self.sparse_scheduler_512,
            self.sparse_vae_1024, self.sparse_dit_1024, self.sparse_scheduler_1024,
            self.refiner, self.refiner_1024
        ]
        
        for model in all_models:
            if model is None:
                continue
            try:
                # Force CPU migration
                if hasattr(model, 'to_cpu'):
                    model.to_cpu()
                elif hasattr(model, 'cpu'):
                    model.cpu()
                
                # Force float32 to free any half-precision GPU tensors
                if hasattr(model, 'float'):
                    model.float()
                    
            except Exception as e:
                print(f"Warning: Could not offload model: {e}")
        
        # Aggressive cleanup
        for _ in range(5):  # Multiple passes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                try:
                    torch.clear_autocast_cache()
                except:
                    pass

    def _debug_gpu_memory(self, label=""):
        """Debug GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            free = torch.cuda.get_device_properties(0).total_memory / (1024**3) - reserved
            print(f"{label}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free≈{free:.2f}GB")
        def _bring_models(self, models: Sequence[Any]):
            """Move a list of models to self.device and cast to half if CUDA."""
            for m in models:
                if m is None:
                    continue
                try:
                    self._set_model_device(m, self.device, prefer_half=(self.device.type == "cuda"))
                except Exception:
                    pass

    def _free_gpu(self, delete_vars: Optional[List[str]] = None, local_vars: Optional[dict] = None):
        """
        Free GPU memory: delete named variables from local_vars (if provided),
        then call empty_cache and synchronize. Best-effort.
        """
        if local_vars is not None and delete_vars is not None:
            for name in delete_vars:
                try:
                    if name in local_vars:
                        del local_vars[name]
                except Exception:
                    pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    # ----------------------
    # load / to
    # ----------------------
    def to(self, device):
        """Bring the pipeline to device. Models are moved and cast to fp16 on CUDA."""
        self.device = torch.device(device)
        # We won't bring every model to the GPU at once by default: keep them ready but not necessarily on GPU.
        # Still set their device so .to() sets internal buffers; then we'll move active ones later.
        for model in [
            self.dense_vae, self.dense_dit,
            self.sparse_vae_512, self.sparse_dit_512,
            self.sparse_vae_1024, self.sparse_dit_1024,
            self.refiner, self.refiner_1024,
            self.dense_image_encoder, self.sparse_image_encoder,
            self.dense_scheduler, self.sparse_scheduler_512, self.sparse_scheduler_1024
        ]:
            if model is None:
                continue
            try:
                # move to chosen device but keep on CPU if GPU memory is scarce;
                # we'll explicitly bring only what's needed during inference.
                model.to(self.device)
                if self.device.type == "cuda":
                    try:
                        model.half()
                    except Exception:
                        pass
            except Exception:
                pass

    # ----------------------
    # from_pretrained (unchanged semantics)
    # ----------------------
    @classmethod
    def from_pretrained(cls, pipeline_path, subfolder="direct3d-s2-v-1-1"):
        # Resolve files (local dir or HF Hub)
        if os.path.isdir(pipeline_path):
            config_path = os.path.join(pipeline_path, 'config.yaml')
            model_dense_path = os.path.join(pipeline_path, 'model_dense.ckpt')
            model_sparse_512_path = os.path.join(pipeline_path, 'model_sparse_512.ckpt')
            model_sparse_1024_path = os.path.join(pipeline_path, 'model_sparse_1024.ckpt')
            model_refiner_path = os.path.join(pipeline_path, 'model_refiner.ckpt')
            model_refiner_1024_path = os.path.join(pipeline_path, 'model_refiner_1024.ckpt')
        else:
            config_path = hf_hub_download(
                repo_id=pipeline_path,
                subfolder=subfolder,
                filename="config.yaml",
                repo_type="model"
            )
            model_dense_path = hf_hub_download(
                repo_id=pipeline_path,
                subfolder=subfolder,
                filename="model_dense.ckpt",
                repo_type="model"
            )
            model_sparse_512_path = hf_hub_download(
                repo_id=pipeline_path,
                subfolder=subfolder,
                filename="model_sparse_512.ckpt",
                repo_type="model"
            )
            model_sparse_1024_path = hf_hub_download(
                repo_id=pipeline_path,
                subfolder=subfolder,
                filename="model_sparse_1024.ckpt",
                repo_type="model"
            )
            model_refiner_path = hf_hub_download(
                repo_id=pipeline_path,
                subfolder=subfolder,
                filename="model_refiner.ckpt",
                repo_type="model"
            )
            model_refiner_1024_path = hf_hub_download(
                repo_id=pipeline_path,
                subfolder=subfolder,
                filename="model_refiner_1024.ckpt",
                repo_type="model"
            )

        cfg = OmegaConf.load(config_path)

        # Dense
        state_dict_dense = torch.load(model_dense_path, map_location='cpu', weights_only=True)
        dense_vae = instantiate_from_config(cfg.dense_vae)
        dense_vae.load_state_dict(state_dict_dense["vae"], strict=True)
        dense_vae.eval()
        dense_dit = instantiate_from_config(cfg.dense_dit)
        dense_dit.load_state_dict(state_dict_dense["dit"], strict=True)
        dense_dit.eval()

        # Sparse 512
        state_dict_sparse_512 = torch.load(model_sparse_512_path, map_location='cpu', weights_only=True)
        sparse_vae_512 = instantiate_from_config(cfg.sparse_vae_512)
        sparse_vae_512.load_state_dict(state_dict_sparse_512["vae"], strict=True)
        sparse_vae_512.eval()
        sparse_dit_512 = instantiate_from_config(cfg.sparse_dit_512)
        sparse_dit_512.load_state_dict(state_dict_sparse_512["dit"], strict=True)
        sparse_dit_512.eval()

        # Sparse 1024
        state_dict_sparse_1024 = torch.load(model_sparse_1024_path, map_location='cpu', weights_only=True)
        sparse_vae_1024 = instantiate_from_config(cfg.sparse_vae_1024)
        sparse_vae_1024.load_state_dict(state_dict_sparse_1024["vae"], strict=True)
        sparse_vae_1024.eval()
        sparse_dit_1024 = instantiate_from_config(cfg.sparse_dit_1024)
        sparse_dit_1024.load_state_dict(state_dict_sparse_1024["dit"], strict=True)
        sparse_dit_1024.eval()

        # Refiners
        state_dict_refiner = torch.load(model_refiner_path, map_location='cpu', weights_only=True)
        refiner = instantiate_from_config(cfg.refiner)
        refiner.load_state_dict(state_dict_refiner["refiner"], strict=True)
        refiner.eval()

        state_dict_refiner_1024 = torch.load(model_refiner_1024_path, map_location='cpu', weights_only=True)
        refiner_1024 = instantiate_from_config(cfg.refiner_1024)
        refiner_1024.load_state_dict(state_dict_refiner_1024["refiner"], strict=True)
        refiner_1024.eval()
        refiner = refiner.to('cpu')
        refiner_1024 = refiner_1024.to('cpu')
        refiner = refiner.half()
        refiner_1024 = refiner_1024.half()
        # Encoders & schedulers
        dense_image_encoder = instantiate_from_config(cfg.dense_image_encoder)
        dense_image_encoder.eval()
        sparse_image_encoder = instantiate_from_config(cfg.sparse_image_encoder)
        sparse_image_encoder.eval()

        dense_scheduler = instantiate_from_config(cfg.dense_scheduler)
        sparse_scheduler_512 = instantiate_from_config(cfg.sparse_scheduler_512)
        sparse_scheduler_1024 = instantiate_from_config(cfg.sparse_scheduler_1024)

        return cls(
            dense_vae=dense_vae,
            dense_dit=dense_dit,
            sparse_vae_512=sparse_vae_512,
            sparse_dit_512=sparse_dit_512,
            sparse_vae_1024=sparse_vae_1024,
            sparse_dit_1024=sparse_dit_1024,
            dense_image_encoder=dense_image_encoder,
            sparse_image_encoder=sparse_image_encoder,
            dense_scheduler=dense_scheduler,
            sparse_scheduler_512=sparse_scheduler_512,
            sparse_scheduler_1024=sparse_scheduler_1024,
            refiner=refiner,
            refiner_1024=refiner_1024,
        )

    # ----------------------
    # preprocessing helpers
    # ----------------------
    def preprocess(self, image):
        # BiRefNet on GPU => half; otherwise keep fp32
        if image.mode == 'RGBA':
            image = np.array(image)
        else:
            if getattr(self, 'birefnet_model', None) is None:
                from direct3d_s2.utils import BiRefNet
                self.birefnet_model = BiRefNet(self.device)
                if self.device.type == "cuda":
                    try:
                        self.birefnet_model.half()
                    except Exception:
                        pass
            image = self.birefnet_model.run(image)
        image = preprocess_image(image)  # likely returns float32 tensor
        return image

    def prepare_image(self, image: Union[str, List[str], Image.Image, List[Image.Image]]):
        if not isinstance(image, list):
            image = [image]
        if isinstance(image[0], str):
            image = [Image.open(img) for img in image]
        image = [self.preprocess(img) for img in image]
        # Move and cast input to target dtype/device
        image = torch.stack([img for img in image]).to(self.device)
        if self.device.type == "cuda":
            try:
                image = image.to(self.dtype)
            except Exception:
                pass
        return image

    def encode_image(self, image: torch.Tensor, conditioner: Any, 
                 do_classifier_free_guidance: bool = True, use_mask: bool = False):
        # Ensure conditioner is on correct device & dtype
        conditioner = conditioner.to(self.device)
        if self.device.type == "cuda":
            conditioner = conditioner.half()  # match input dtype

        # Run conditioner under autocast when on CUDA
        with self._autocast_ctx():
            if use_mask:
                cond = conditioner(image[:, :3], image[:, 3:])
            else:
                cond = conditioner(image[:, :3])

        if isinstance(cond, tuple):
            cond, cond_mask = cond
            cond, cond_coords = extract_tokens_and_coords(cond, cond_mask)
        else:
            cond_mask, cond_coords = None, None

        if do_classifier_free_guidance:
            uncond = torch.zeros_like(cond)
        else:
            uncond = None
    
        if cond_coords is not None:
            cond = sp.SparseTensor(cond, cond_coords.int())
            if uncond is not None:
                uncond = sp.SparseTensor(uncond, cond_coords.int())

        return cond, uncond


    # ----------------------
    # inference (memory-optimized)
    # ----------------------
    def inference(
        self,
        image,
        vae,
        dit,
        conditioner,
        scheduler,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        generator: Optional[torch.Generator] = None,
        latent_index: torch.Tensor = None,
        mode: str = 'dense',  # 'dense', 'sparse512', 'sparse1024'
        remove_interior: bool = False,
        mc_threshold: float = 0.02,
    ):
        import gc

        # model groups
        dense_models = [self.dense_vae, self.dense_dit, self.dense_image_encoder, self.dense_scheduler]
        sparse512_models = [self.sparse_vae_512, self.sparse_dit_512, self.sparse_image_encoder, self.sparse_scheduler_512]
        sparse1024_models = [self.sparse_vae_1024, self.sparse_dit_1024, self.sparse_image_encoder, self.sparse_scheduler_1024]
        refiner_models = [self.refiner, self.refiner_1024]

        # helper movers: best-effort; no hard errors if model can't move
        def _to_device(models, device, dtype=None):
            if not isinstance(models, (list, tuple)):
                models = [models]
            for m in models:
                if m is None:
                    continue
                try:
                    m.to(device)
                    if device.type == "cuda" and dtype == torch.float16:
                        try:
                            m.half()
                        except Exception:
                            pass
                    else:
                        try:
                            m.float()
                        except Exception:
                            pass
                except Exception:
                    # ignore functions or unmovable objects
                    pass

        def _to_cpu(models):
            # move to cpu and convert to float for numerical stability (minimizes dtype mismatch later)
            if not isinstance(models, (list, tuple)):
                models = [models]
            for m in models:
                if m is None:
                    continue
                try:
                    m.to(torch.device("cpu"))
                    try:
                        m.float()
                    except Exception:
                        pass
                except Exception:
                    pass

        # prefer device configured on self
        device = getattr(self, "device", torch.device("cpu"))
        use_cuda = device.type == "cuda"

        # Selectively bring required models to GPU and offload others immediately
        if use_cuda:
            if mode == 'dense':
                _to_device(dense_models, device, dtype=self.dtype)
                # offload everything else
                _to_cpu(sparse512_models + sparse1024_models + refiner_models)
            elif mode == 'sparse512':
                _to_device(sparse512_models, device, dtype=self.dtype)
                _to_cpu(dense_models + sparse1024_models + refiner_models)
            elif mode == 'sparse1024':
                _to_device(sparse1024_models, device, dtype=self.dtype)
                _to_cpu(dense_models + sparse512_models + refiner_models)
        else:
            # CPU-only: ensure everything is CPU/float
            _to_cpu(dense_models + sparse512_models + sparse1024_models + refiner_models)

        # ============ encode image ============
        do_classifier_free_guidance = guidance_scale > 0
        sparse_conditions = dit.sparse_conditions if mode != 'dense' else False

        if use_cuda:
            image = image.to(device, dtype=self.dtype, non_blocking=True)

        cond, uncond = self.encode_image(image, conditioner, do_classifier_free_guidance, sparse_conditions)
        batch_size = cond.shape[0]

        # ============ prepare latents ============
        latent_shape = (batch_size, *dit.latent_shape) if mode == 'dense' else (len(latent_index), dit.out_channels)
        latents_dtype = self.dtype if use_cuda else torch.float32
        latents = torch.randn(latent_shape, dtype=latents_dtype, device=device, generator=generator)

        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        extra_step_kwargs = {"generator": generator}

        # ============ sampling loop (keep necessary models on GPU) ============
        for i, t in enumerate(tqdm(timesteps, desc=f"{mode} Sampling:")):
            timestep_tensor = torch.tensor([t], dtype=torch.float32, device=device)

            if mode == 'dense':
                x_input = latents
            else:
                # pack sparse latent (this keeps data on GPU device)
                x_input = sp.SparseTensor(latents, latent_index.int())

            diffusion_inputs = {"x": x_input, "t": timestep_tensor, "cond": cond}

            # forward (autocast context if present)
            with self._autocast_ctx():
                noise_pred_cond = dit(**diffusion_inputs)
                if mode != 'dense':
                    noise_pred_cond = noise_pred_cond.feats

                if do_classifier_free_guidance:
                    diffusion_inputs["cond"] = uncond
                    noise_pred_uncond = dit(**diffusion_inputs)
                    if mode != 'dense':
                        noise_pred_uncond = noise_pred_uncond.feats
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

            # scheduler step
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # aggressively clear per-step temp tensors and CPU/GPU caches
            try:
                del noise_pred_cond, noise_pred_uncond, noise_pred
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()

        # ============ post-process latents ============
        latents = 1. / vae.latents_scale * latents + vae.latents_shift
        if mode != 'dense':
            latents = sp.SparseTensor(latents, latent_index.int())

        # Prepare decode args
        decoder_inputs = {"latents": latents, "mc_threshold": mc_threshold}
        if mode == 'dense':
            decoder_inputs['return_index'] = True
        elif remove_interior:
            decoder_inputs['return_feat'] = True
        if mode == 'sparse1024':
            decoder_inputs['voxel_resolution'] = 1024

        # Drop things we no longer need and free VRAM
        try:
            del cond, uncond, x_input
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        # Ensure VAE is on the device we want and offload unrelated heavy models
        if use_cuda:
            # Move VAE to GPU (and to fp16 if set). Offload everything else (including samplers & refiners).
            _to_device([vae], device, dtype=self.dtype)
            _to_cpu([m for m in (dense_models + sparse512_models + sparse1024_models + refiner_models) if m is not vae])
            torch.cuda.empty_cache()
            gc.collect()
        else:
            _to_cpu([vae])

        # ============ decode (can be large) ============
        with self._autocast_ctx():
            outputs = vae.decode_mesh(**decoder_inputs)

        # free latents & decoder temporaries quickly
        try:
            del latents
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        self._free_gpu(delete_vars=None, local_vars=None)
        # ======== prepare for refinement (OFFLOAD sampling models if not already) ========
        if remove_interior:
            print("Starting refinement - clearing all GPU memory...")
            self._debug_gpu_memory("Before refinement cleanup")

            # NUCLEAR CLEANUP: Force all models off GPU
            self._force_offload_all_models()
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self._debug_gpu_memory("After nuclear cleanup")

            # Helper to move outputs to CPU only if needed
            def force_cpu_float32(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().float() if x.is_cuda else x.float()
                return x

            if isinstance(outputs, tuple):
                outputs = tuple(force_cpu_float32(o) for o in outputs)
            else:
                outputs = force_cpu_float32(outputs)

            # Select appropriate refiner
            refiner_model = None
            if mode == 'sparse512':
                refiner_model = self.refiner
                mc_scale = mc_threshold * 2.0
            elif mode == 'sparse1024':
                refiner_model = self.refiner_1024
                mc_scale = mc_threshold

            if refiner_model is not None:
                try:
                    print(f"Loading {mode} refiner to GPU in FP16...")
                    # Directly move to GPU and convert to FP16
                    refiner_model = refiner_model.half().to("cuda")
                    self._debug_gpu_memory(f"{mode} refiner loaded")

                    # Run refinement with no_grad for minimal VRAM
                    with torch.no_grad():
                        outputs = refiner_model.run(*outputs, mc_threshold=mc_scale)

                except torch.cuda.OutOfMemoryError as e:
                    print(f"OOM during {mode} refiner execution: {e}")
                    print("Falling back to CPU refinement...")
                    refiner_model = refiner_model.cpu().float()
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        outputs = refiner_model.run(*outputs, mc_threshold=mc_scale)

                finally:
                    # Always clean up refiner from GPU
                    refiner_model.cpu().float()
                    self._force_offload_all_models()
                    torch.cuda.empty_cache()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            self._debug_gpu_memory("After refinement")

        # Final cleanup
        self._force_offload_all_models()
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return outputs

    def _set_model_device(self, model, device: torch.device, prefer_half: bool = True):
        """Move model to device with proper dtype handling"""
        if model is None:
            return
            
        try:
            # Clear any existing GPU memory first
            if hasattr(model, 'cpu'):
                model.cpu()
            torch.cuda.empty_cache()
            
            model.to(device)
            
            if device.type == "cuda" and prefer_half:
                try:
                    model.half()
                    self._active_models.add(id(model))
                except Exception as e:
                    print(f"Warning: Could not convert model to half precision: {e}")
            else:
                try:
                    model.float()
                    if id(model) in self._active_models:
                        self._active_models.remove(id(model))
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Could not move model to {device}: {e}")

    def _bring_models(self, models: Sequence[Any]):
        """Move a list of models to self.device and cast appropriately"""
        for m in models:
            if m is None:
                continue
            try:
                self._set_model_device(m, self.device, prefer_half=(self.device.type == "cuda"))
            except Exception as e:
                print(f"Warning: Could not bring model to device: {e}")

    def _offload_models(self, models: Sequence[Any]):
        """Move models to CPU with proper cleanup"""
        for m in models:
            if m is None:
                continue
            try:
                # Special handling for refiners with custom methods
                if hasattr(m, 'to_cpu'):
                    m.to_cpu()
                else:
                    m.cpu()
                    
                # Always convert to float32 on CPU for stability
                if hasattr(m, 'float'):
                    m.float()
                    
                # Remove from active tracking
                if id(m) in self._active_models:
                    self._active_models.remove(id(m))
                    
            except Exception as e:
                print(f"Warning: Could not offload model: {e}")
        
        # Cleanup after offloading
        torch.cuda.empty_cache()
        gc.collect()

    def _force_offload_all_models(self):
        """Nuclear option: force all models to CPU and clear all GPU memory"""
        all_models = [
            self.dense_vae, self.dense_dit, self.dense_image_encoder, self.dense_scheduler,
            self.sparse_vae_512, self.sparse_dit_512, self.sparse_image_encoder, self.sparse_scheduler_512,
            self.sparse_vae_1024, self.sparse_dit_1024, self.sparse_scheduler_1024,
            self.refiner, self.refiner_1024
        ]
        
        for model in all_models:
            if model is None:
                continue
            try:
                # Force CPU migration
                if hasattr(model, 'to_cpu'):
                    model.to_cpu()
                elif hasattr(model, 'cpu'):
                    model.cpu()
                
                # Force float32 to free any half-precision GPU tensors
                if hasattr(model, 'float'):
                    model.float()
                    
            except Exception as e:
                print(f"Warning: Could not offload model: {e}")
        
        # Clear tracking
        self._active_models.clear()
        
        # Aggressive cleanup
        for _ in range(5):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                try:
                    torch.clear_autocast_cache()
                except:
                    pass

    def _debug_gpu_memory(self, label=""):
        """Debug GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - reserved
            print(f"{label}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free≈{free:.2f}GB")

    def _free_gpu(self, delete_vars: Optional[List[str]] = None, local_vars: Optional[dict] = None):
        """Free GPU memory with variable deletion"""
        if local_vars is not None and delete_vars is not None:
            for name in delete_vars:
                try:
                    if name in local_vars:
                        del local_vars[name]
                except Exception:
                    pass
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.clear_autocast_cache()
            except Exception:
                pass
        gc.collect()

    # ----------------------
    # Device management
    # ----------------------
    def to(self, device):
        """Move pipeline to device"""
        self.device = torch.device(device)
        # Don't move all models immediately - we'll do it selectively during inference
        return self
    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, List[Image.Image]] = None,
        sdf_resolution: int = 1024,
        dense_sampler_params: dict = {'num_inference_steps': 50, 'guidance_scale': 7.0},
        sparse_512_sampler_params: dict = {'num_inference_steps': 30, 'guidance_scale': 7.0},
        sparse_1024_sampler_params: dict = {'num_inference_steps': 15, 'guidance_scale': 7.0},
        generator: Optional[torch.Generator] = None,
        remesh: bool = False,
        simplify_ratio: float = 0.90,
        mc_threshold: float = 0.2,
        remove_interior: bool = True):

        # ensure device configured
        if getattr(self, "device", None) is None:
            self.device = torch.device("cpu")

        image = self.prepare_image(image)
        if self.device.type == "cuda":
            image = image.to(self.device, dtype=torch.float16, non_blocking=True)

        # Bring dense models and offload others prior to dense inference
        if self.device.type == "cuda":
            # explicitly move only dense models to GPU
            self._bring_models([self.dense_vae, self.dense_dit, self.dense_image_encoder, self.dense_scheduler])
            # ensure everything else is on CPU to minimize fragmentation
            self._offload_models([self.sparse_vae_512, self.sparse_dit_512, self.sparse_vae_1024,
                              self.sparse_dit_1024, self.refiner, self.refiner_1024,
                              self.sparse_image_encoder, self.sparse_scheduler_512, self.sparse_scheduler_1024])
        else:
            # CPU-only: keep models CPU
            self._offload_models([self.dense_vae, self.dense_dit, self.dense_image_encoder, self.dense_scheduler,
                              self.sparse_vae_512, self.sparse_dit_512, self.sparse_vae_1024,
                              self.sparse_dit_1024, self.refiner, self.refiner_1024,
                              self.sparse_image_encoder, self.sparse_scheduler_512, self.sparse_scheduler_1024])

        # Dense inference -> returns latent index
        latent_index = self.inference(
            image, self.dense_vae, self.dense_dit, self.dense_image_encoder,
            self.dense_scheduler, generator=generator, mode='dense',
            mc_threshold=0.1, **dense_sampler_params
        )[0]

        # Immediately offload dense models and free VRAM
        self._free_gpu(delete_vars=None, local_vars=None)
        if self.device.type == "cuda":
            self._offload_models([self.dense_vae, self.dense_dit, self.dense_image_encoder, self.dense_scheduler])
            # bring sparse512 models for the next stage
            self._bring_models([self.sparse_vae_512, self.sparse_dit_512, self.sparse_image_encoder, self.sparse_scheduler_512])

        # sort latent index, then free any temporary memory
        latent_index = sort_block(latent_index, self.sparse_dit_512.selection_block_size)
        self._free_gpu(delete_vars=None, local_vars=None)

        # Sparse512 stage (heavy): run with sparse models; refiners remain on CPU to minimize VRAM
        mesh = self.inference(
            image, self.sparse_vae_512, self.sparse_dit_512,
            self.sparse_image_encoder, self.sparse_scheduler_512,
            generator=generator, mode='sparse512',
            mc_threshold=mc_threshold, latent_index=latent_index,
            remove_interior=remove_interior, **sparse_512_sampler_params
        )[0]

        # If user requested final higher-resolution pass
        if sdf_resolution == 1024:
            # free latent_index and cached GPU memory before building 1024 index
            try:
                del latent_index
            except Exception:
                pass
            self._free_gpu(delete_vars=None, local_vars=None)

            mesh = normalize_mesh(mesh)
    
            # mesh2index is large — ensure inputs live on CPU to avoid GPU OOM
            # move heavy model/state off GPU and run mesh2index on CPU
            if self.device.type == "cuda":
                self._offload_models([self.sparse_vae_512, self.sparse_dit_512, self.sparse_image_encoder, self.sparse_scheduler_512])
                self._offload_models([self.refiner, self.refiner_1024])
                self._offload_models([self.dense_vae, self.dense_dit, self.dense_image_encoder, self.dense_scheduler])
            torch.cuda.empty_cache()
            gc.collect()

            latent_index = mesh2index(mesh, size=1024, factor=8)
            latent_index = sort_block(latent_index, self.sparse_dit_1024.selection_block_size)
            print(f"number of latent tokens: {len(latent_index)}")

            # bring sparse1024 models to GPU if device available; keep refiners on CPU
            if self.device.type == "cuda":
                self._bring_models([self.sparse_vae_1024, self.sparse_dit_1024, self.sparse_image_encoder, self.sparse_scheduler_1024])
                
            del mesh  # optional if you will reload/reuse it
            torch.cuda.empty_cache()
            gc.collect()

            mesh = self.inference(
                image, self.sparse_vae_1024, self.sparse_dit_1024,
                self.sparse_image_encoder, self.sparse_scheduler_1024,
                generator=generator, mode='sparse1024',
                mc_threshold=mc_threshold, latent_index=latent_index,
                remove_interior=remove_interior, **sparse_1024_sampler_params
            )[0]

        if remesh:
            import trimesh
            from direct3d_s2.utils import postprocess_mesh
            filled_mesh = postprocess_mesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                simplify=True,
                simplify_ratio=simplify_ratio,
                verbose=True,
            )
            mesh = trimesh.Trimesh(filled_mesh[0], filled_mesh[1])
        else:
            filled_mesh = postprocess_mesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                simplify=False,
                verbose=True,
            )
            mesh = trimesh.Trimesh(filled_mesh[0], filled_mesh[1])
        # final cleanup: ensure everything that is not needed is on CPU
        try:
            self._offload_models([self.dense_vae, self.dense_dit, self.sparse_vae_512, self.sparse_dit_512,
                              self.sparse_vae_1024, self.sparse_dit_1024, self.refiner, self.refiner_1024,
                              self.dense_image_encoder, self.sparse_image_encoder,
                              self.dense_scheduler, self.sparse_scheduler_512, self.sparse_scheduler_1024])
        except Exception:
            pass
        self._free_gpu(delete_vars=None, local_vars=None)

        return {"mesh": mesh}
