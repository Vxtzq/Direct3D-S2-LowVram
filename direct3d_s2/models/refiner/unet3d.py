from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}

def get_activation(act_fn: str) -> nn.Module:

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")

def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dropout: float = 0.0,
) -> Union[
    "DownBlock3D",
]:
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            dropout=dropout,
        )

    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: Optional[int] = None,
    dropout: float = 0.0,
) -> Union[
    "UpBlock3D",
]:
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            dropout=dropout,
        )
    raise ValueError(f"{up_block_type} does not exist.")

class Downsample3D(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        kernel_size=2,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2

        self.conv = nn.Conv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias
            )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states

class Upsample3D(nn.Module):

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = True,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        bias=True,
        interpolate=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose3d(
                channels, self.out_channels, kernel_size=2, stride=2, padding=0, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv3d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor:

        assert hidden_states.shape[1] == self.channels


        if self.use_conv_transpose:
            return self.conv(hidden_states)
    
        if hidden_states.shape[0] >= 64 or hidden_states.shape[-1] >= 64:
            hidden_states = hidden_states.contiguous()

        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states
    
class ResnetBlock3D(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv3d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample3D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = Downsample3D(in_channels)

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:

        hidden_states = input_tensor
        dtype = hidden_states.dtype
        hidden_states = self.norm1(hidden_states.float()).to(dtype)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states.float()).to(dtype)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        out_channels=out_channels
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states
    
class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        upsample_size: Optional[int] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]


            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

    
class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        use_linear_projection: bool = True,
    ):
        super().__init__()

        self.has_cross_attention = True
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
            )
        ]

        for _ in range(num_layers):
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states
    
def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward
                
class UNet3DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        use_conv_out: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock3D",
            "DownBlock3D", 
            "DownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock3D",
            "UpBlock3D",
            "UpBlock3D", 
            "UpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (4, 16, 64, 256),
        layers_per_block: int = 4,
        layers_mid_block: int = 4,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 4,
        norm_eps: float = 1e-5,
        use_checkpoint: bool = True,
        # New chunking parameters
        chunk_size: int = 16,  # Increased minimum chunk size
        overlap: int = 4,      # Increased overlap
        enable_chunking: bool = True,
        min_chunk_size: int = 8,  # Minimum chunk size to avoid conv errors
    ):
        super().__init__()
        
        # Store chunking parameters
        self.chunk_size = max(chunk_size, min_chunk_size)
        self.overlap = overlap
        self.enable_chunking = enable_chunking
        self.min_chunk_size = min_chunk_size
        
        # Calculate minimum required temporal size based on downsampling
        self.temporal_downsample_factor = 2 ** len([bt for bt in down_block_types if "Down" in bt])
        self.min_temporal_size = max(min_chunk_size, self.temporal_downsample_factor * 2)
        
        # Original validation logic
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv3d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            num_layers=layers_mid_block,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_groups=norm_num_groups,
        )

        self.num_upsamplers = 0

        reversed_block_out_channels = list(reversed(block_out_channels))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
            self.conv_act = get_activation("silu")
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        if use_conv_out:
            self.conv_out = nn.Conv3d(
                block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
            )
        else:
            self.conv_out = None

        self.use_checkpoint = use_checkpoint

    def _get_chunks(self, total_frames: int):
        """Calculate chunk indices with overlap, ensuring minimum chunk size"""
        # If chunking disabled or input is small enough, process all at once
        if not self.enable_chunking or total_frames <= self.min_temporal_size:
            return [(0, total_frames)]
        
        # Ensure chunk size is large enough for downsampling operations
        effective_chunk_size = max(self.chunk_size, self.min_temporal_size)
        
        # If total frames is smaller than effective chunk size, process all
        if total_frames <= effective_chunk_size:
            return [(0, total_frames)]
        
        chunks = []
        start = 0
        while start < total_frames:
            end = min(start + effective_chunk_size, total_frames)
            
            # Ensure chunk is large enough
            if end - start < self.min_temporal_size and start > 0:
                # Extend the previous chunk instead of creating a too-small chunk
                chunks[-1] = (chunks[-1][0], end)
                break
            
            chunks.append((start, end))
            if end == total_frames:
                break
            
            # Calculate next start with overlap, but ensure we don't create too small chunks
            next_start = end - self.overlap
            remaining_frames = total_frames - next_start
            
            # If remaining frames would create a chunk that's too small, 
            # adjust the overlap to create a properly sized final chunk
            if remaining_frames < self.min_temporal_size and remaining_frames > 0:
                next_start = total_frames - self.min_temporal_size
                next_start = max(next_start, end - effective_chunk_size + 1)
            
            start = next_start
            
        return chunks

    def _blend_chunks(self, chunks_results, chunks_indices, total_frames, device, dtype):
        """Blend overlapping chunks together"""
        if len(chunks_results) == 1:
            return chunks_results[0]
        
        # Get output shape from first chunk
        first_chunk = chunks_results[0]
        B, C, _, H, W = first_chunk.shape
        
        # Initialize output tensor
        output = torch.zeros(B, C, total_frames, H, W, device=device, dtype=dtype)
        weight_sum = torch.zeros(total_frames, device=device, dtype=dtype)
        
        for chunk_result, (start, end) in zip(chunks_results, chunks_indices):
            chunk_length = end - start
            actual_chunk_length = chunk_result.shape[2]
            
            # Handle case where chunk result might be different size due to processing
            if actual_chunk_length != chunk_length:
                chunk_length = actual_chunk_length
                end = start + chunk_length
            
            # Create blending weights (higher in center, lower at edges)
            weights = torch.ones(chunk_length, device=device, dtype=dtype)
            
            # Apply fade in/out for overlapping regions
            if len(chunks_results) > 1:  # Only blend if multiple chunks
                overlap_frames = min(self.overlap, chunk_length // 3)  # Don't use more than 1/3 for blending
                
                if start > 0:  # Not first chunk, fade in
                    weights[:overlap_frames] = torch.linspace(0.1, 1.0, overlap_frames, device=device, dtype=dtype)
                if end < total_frames:  # Not last chunk, fade out
                    weights[-overlap_frames:] = torch.linspace(1.0, 0.1, overlap_frames, device=device, dtype=dtype)
            
            # Apply weighted blending
            for i in range(min(chunk_length, total_frames - start)):
                frame_idx = start + i
                if frame_idx < total_frames:
                    weight = weights[i]
                    output[:, :, frame_idx] += chunk_result[:, :, i] * weight
                    weight_sum[frame_idx] += weight
        
        # Normalize by weight sum
        weight_sum = weight_sum.clamp(min=1e-8)
        output = output / weight_sum[None, None, :, None, None]
        
        return output

    def _process_chunk(self, chunk: torch.Tensor):
        """Process a single chunk through the UNet"""
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in chunk.shape[-2:]):
            forward_upsample_size = True

        sample = self.conv_in(chunk)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if self.use_checkpoint:
                sample, res_samples = torch.utils.checkpoint.checkpoint(
                    downsample_block, sample, use_reentrant=False
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample)

            down_block_res_samples += res_samples

        if self.mid_block is not None:
            if self.use_checkpoint:
                sample = torch.utils.checkpoint.checkpoint(
                    self.mid_block, sample, use_reentrant=False
                )
            else:
                sample = self.mid_block(sample)

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
                
            if self.use_checkpoint:
                sample = torch.utils.checkpoint.checkpoint(
                    upsample_block, (sample, res_samples, upsample_size), use_reentrant=False
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if self.conv_norm_out:
            dtype = sample.dtype
            sample = self.conv_norm_out(sample.float()).to(dtype)
            sample = self.conv_act(sample)
            
        if self.conv_out is not None:
            sample = self.conv_out(sample)
            return F.tanh(sample) * 2
        else:
            return sample

    def forward(self, sample: torch.Tensor):
        B, C, T, H, W = sample.shape
        device = sample.device
        dtype = sample.dtype
        
        # Check if we should use chunking
        chunks_indices = self._get_chunks(T)
        
        if len(chunks_indices) == 1:
            # Process without chunking
            return self._process_chunk(sample)
        
        # Process each chunk
        chunks_results = []
        for start, end in chunks_indices:
            chunk = sample[:, :, start:end]
            
            # Verify chunk size is valid
            chunk_temporal_size = chunk.shape[2]
            if chunk_temporal_size < self.min_temporal_size:
                print(f"Warning: Chunk size {chunk_temporal_size} is smaller than minimum {self.min_temporal_size}, processing without chunking")
                return self._process_chunk(sample)
            
            # Clear cache before processing each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            chunk_result = self._process_chunk(chunk)
            chunks_results.append(chunk_result.cpu())  # Move to CPU to save VRAM
        
        # Move results back to GPU and blend
        chunks_results = [chunk.to(device) for chunk in chunks_results]
        result = self._blend_chunks(chunks_results, chunks_indices, T, device, dtype)
        
        return result

    def set_chunking_params(self, chunk_size: int = None, overlap: int = None, enable_chunking: bool = None):
        """Dynamically adjust chunking parameters"""
        if chunk_size is not None:
            self.chunk_size = max(chunk_size, self.min_temporal_size)
        if overlap is not None:
            self.overlap = overlap
        if enable_chunking is not None:
            self.enable_chunking = enable_chunking
            
        print(f"Chunking params: chunk_size={self.chunk_size}, overlap={self.overlap}, "
              f"enabled={self.enable_chunking}, min_size={self.min_temporal_size}")
