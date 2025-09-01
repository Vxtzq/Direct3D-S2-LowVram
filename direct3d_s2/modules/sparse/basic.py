from typing import *
import torch
import torch.nn as nn
from . import BACKEND, DEBUG
SparseTensorData = None # Lazy import

# default feature dtype to prefer
_DEFAULT_FP_DTYPE = torch.float16

__all__ = [
    'SparseTensor',
    'sparse_batch_broadcast',
    'sparse_batch_op',
    'sparse_cat',
    'sparse_unbind',
]


def _maybe_half(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert floating tensors to half to save VRAM. Leave integer tensors unchanged."""
    if t is None:
        return None
    return t.half() if t.is_floating_point() else t


class SparseTensor:
    """
    Sparse tensor with support for both torchsparse and spconv backends.

    Parameters:
    - feats (torch.Tensor): Features of the sparse tensor.
    - coords (torch.Tensor): Coordinates of the sparse tensor.
    - shape (torch.Size): Shape of the sparse tensor.
    - layout (List[slice]): Layout of the sparse tensor for each batch
    - data (SparseTensorData): Sparse tensor data used for convolusion

    NOTE:
    - Data corresponding to a same batch should be contiguous.
    - Coords should be in [0, 1023]
    """
    @overload
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor, shape: Optional[torch.Size] = None, layout: Optional[List[slice]] = None, **kwargs): ...

    @overload
    def __init__(self, data, shape: Optional[torch.Size] = None, layout: Optional[List[slice]] = None, **kwargs): ...

    def __init__(self, *args, **kwargs):
        # Lazy import of sparse tensor backend
        global SparseTensorData
        if SparseTensorData is None:
            import importlib
            if BACKEND == 'torchsparse':
                SparseTensorData = importlib.import_module('torchsparse').SparseTensor
            elif BACKEND == 'spconv':
                SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor

        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if 'data' in kwargs else 0

        if method_id == 0:
            feats, coords, shape, layout = args + (None,) * (4 - len(args))
            if 'feats' in kwargs:
                feats = kwargs['feats']
                del kwargs['feats']
            if 'coords' in kwargs:
                coords = kwargs['coords']
                del kwargs['coords']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']
            if 'layout' in kwargs:
                layout = kwargs['layout']
                del kwargs['layout']

            if shape is None:
                shape = self.__cal_shape(feats, coords)
            if layout is None:
                layout = self.__cal_layout(coords, shape[0])

            # Convert floating features to fp16 by default to reduce VRAM footprint.
            # Keep coords as integer tensors.
            if feats is not None and isinstance(feats, torch.Tensor) and feats.is_floating_point():
                feats = feats.half()

            if BACKEND == 'torchsparse':
                # torchsparse expects (feats, coords)
                self.data = SparseTensorData(feats, coords, **kwargs)
            elif BACKEND == 'spconv':
                spatial_shape = list(coords.max(0)[0] + 1)[1:]
                # spconv expects features flattened; preserve dtype (already half if floating)
                self.data = SparseTensorData(feats.reshape(feats.shape[0], -1), coords, spatial_shape, shape[0], **kwargs)
                # keep a direct pointer to features (avoid unnecessary copies)
                self.data._features = feats

        elif method_id == 1:
            data, shape, layout = args + (None,) * (3 - len(args))
            if 'data' in kwargs:
                data = kwargs['data']
                del kwargs['data']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']
            if 'layout' in kwargs:
                layout = kwargs['layout']
                del kwargs['layout']

            self.data = data
            if shape is None:
                shape = self.__cal_shape(self.feats, self.coords)
            if layout is None:
                layout = self.__cal_layout(self.coords, shape[0])

        self._shape = shape
        self._layout = layout
        self._scale = kwargs.get('scale', (1, 1, 1))
        self._spatial_cache = kwargs.get('spatial_cache', {})

        if DEBUG:
            try:
                assert self.feats.shape[0] == self.coords.shape[0], f"Invalid feats shape: {self.feats.shape}, coords shape: {self.coords.shape}"
                assert self.shape == self.__cal_shape(self.feats, self.coords), f"Invalid shape: {self.shape}"
                assert self.layout == self.__cal_layout(self.coords, self.shape[0]), f"Invalid layout: {self.layout}"
                for i in range(self.shape[0]):
                    assert torch.all(self.coords[self.layout[i], 0] == i), f"The data of batch {i} is not contiguous"
            except Exception as e:
                print('Debugging information:')
                print(f"- Shape: {self.shape}")
                print(f"- Layout: {self.layout}")
                print(f"- Scale: {self._scale}")
                print(f"- Coords: {self.coords}")
                raise e

    def __cal_shape(self, feats, coords):
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)

    def __cal_layout(self, coords, batch_size):
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0)
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]
        return layout

    @property
    def shape(self) -> torch.Size:
        return self._shape

    def dim(self) -> int:
        return len(self.shape)

    @property
    def layout(self) -> List[slice]:
        return self._layout

    @property
    def feats(self) -> torch.Tensor:
        if BACKEND == 'torchsparse':
            return self.data.F
        elif BACKEND == 'spconv':
            # spconv stores a couple of representations; prefer the direct _features if present
            if hasattr(self.data, '_features') and self.data._features is not None:
                return self.data._features
            return self.data.features

    @feats.setter
    def feats(self, value: torch.Tensor):
        # ensure value floats are half to reduce VRAM
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            value = value.half()
        if BACKEND == 'torchsparse':
            self.data.F = value
        elif BACKEND == 'spconv':
            # set direct _features if available to avoid copies
            try:
                self.data._features = value
            except Exception:
                self.data.features = value

    @property
    def coords(self) -> torch.Tensor:
        if BACKEND == 'torchsparse':
            return self.data.C
        elif BACKEND == 'spconv':
            return self.data.indices

    @coords.setter
    def coords(self, value: torch.Tensor):
        if BACKEND == 'torchsparse':
            self.data.C = value
        elif BACKEND == 'spconv':
            self.data.indices = value

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    @overload
    def to(self, dtype: torch.dtype) -> 'SparseTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> 'SparseTensor': ...

    def to(self, *args, **kwargs) -> 'SparseTensor':
        """
        Move feats and coords. If moving to CUDA and dtype is unspecified, convert floating feats to half.
        """
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']

        new_feats = self.feats
        new_coords = self.coords

        # apply device move
        if device is not None:
            new_coords = new_coords.to(device=device)
            # keep dtype behavior: if dtype given, use it; else if moving to cuda convert floats to half
            if dtype is not None:
                new_feats = new_feats.to(device=device, dtype=dtype)
            else:
                if new_feats.is_floating_point():
                    # prefer fp16 on GPU for memory savings
                    if isinstance(device, torch.dtype):
                        # weird call: treat as dtype only
                        new_feats = new_feats.to(dtype=device)
                    else:
                        dev = torch.device(device) if not isinstance(device, torch.device) else device
                        if dev.type == 'cuda':
                            new_feats = new_feats.to(device=dev).half()
                        else:
                            new_feats = new_feats.to(device=dev)
                else:
                    new_feats = new_feats.to(device=device)
        else:
            # only dtype change requested
            if dtype is not None:
                new_feats = new_feats.to(dtype=dtype)

        return self.replace(new_feats, new_coords)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'SparseTensor':
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        # keep features half on CPU if they were half on GPU (avoid converting back unless requested)
        return self.replace(new_feats, new_coords)

    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        # ensure floats are half on GPU
        if new_feats.is_floating_point():
            new_feats = new_feats.half()
        return self.replace(new_feats, new_coords)

    def half(self) -> 'SparseTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)

    def float(self) -> 'SparseTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)

    def detach(self) -> 'SparseTensor':
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        return self.replace(new_feats, new_coords)

    def dense(self) -> torch.Tensor:
        # delegate to backend dense; note this may explode VRAM if full dense
        if BACKEND == 'torchsparse':
            return self.data.dense()
        elif BACKEND == 'spconv':
            return self.data.dense()

    def reshape(self, *shape) -> 'SparseTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)

    def unbind(self, dim: int) -> List['SparseTensor']:
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> 'SparseTensor':
        """
        Create a new SparseTensor object that reuses the backend data structures where possible.
        We convert floating `feats` to half to save memory.
        """
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])

        # convert floating features to half eagerly
        if isinstance(feats, torch.Tensor) and feats.is_floating_point():
            feats = feats.half()

        if BACKEND == 'torchsparse':
            new_data = SparseTensorData(
                feats=feats,
                coords=self.data.coords if coords is None else coords,
                stride=self.data.stride,
                spatial_range=self.data.spatial_range,
            )
            # try to preserve internal caches if available
            try:
                new_data._caches = self.data._caches
            except Exception:
                pass
        elif BACKEND == 'spconv':
            # Try to avoid creating large intermediate arrays where possible.
            # Create a new sparseconv object referencing the same metadata where possible,
            # but set its feature pointer to our (possibly half) feats tensor.
            try:
                new_data = self.data  # reuse underlying data structure (shallow)
                new_data._features = feats
                if coords is not None:
                    new_data.indices = coords
            except Exception:
                # fallback to constructing a fresh SparseConvTensor if reuse fails
                new_data = SparseTensorData(
                    feats.reshape(feats.shape[0], -1),
                    self.data.indices,
                    self.data.spatial_shape,
                    self.data.batch_size,
                    self.data.grid,
                    self.data.voxel_num,
                    self.data.indice_dict
                )
                new_data._features = feats

            # copy other metadata pointers
            try:
                new_data.benchmark = self.data.benchmark
                new_data.benchmark_record = self.data.benchmark_record
                new_data.thrust_allocator = self.data.thrust_allocator
                new_data._timer = self.data._timer
                new_data.force_algo = self.data.force_algo
                new_data.int8_scale = self.data.int8_scale
            except Exception:
                pass

        new_tensor = SparseTensor(new_data, shape=torch.Size(new_shape), layout=self.layout, scale=self._scale, spatial_cache=self._spatial_cache)
        return new_tensor

    @staticmethod
    def full(aabb, dim, value, dtype=_DEFAULT_FP_DTYPE, device=None) -> 'SparseTensor':
        """
        Build a dense block of coords and constant feats. Default feats dtype is FP16 to save memory.
        Coords are built on CPU and only moved to `device` if requested.
        """
        N, C = dim
        # create coords on CPU to avoid big GPU allocation
        x = torch.arange(aabb[0], aabb[3] + 1, device='cpu')
        y = torch.arange(aabb[1], aabb[4] + 1, device='cpu')
        z = torch.arange(aabb[2], aabb[5] + 1, device='cpu')
        coords = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        coords = torch.cat([
            torch.arange(N, device='cpu').view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
            coords.repeat(N, 1),
        ], dim=1).to(dtype=torch.int32)

        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device='cpu')
        if device is not None:
            coords = coords.to(device=device)
            feats = feats.to(device=device)
        return SparseTensor(feats=feats, coords=coords)

    def __merge_sparse_cache(self, other: 'SparseTensor') -> dict:
        new_cache = {}
        for k in set(list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k]
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k]
                else:
                    # update shallowly (avoid copying large tensors)
                    try:
                        new_cache[k].update(other._spatial_cache[k])
                    except Exception:
                        new_cache[k] = other._spatial_cache[k]
        return new_cache

    def __neg__(self) -> 'SparseTensor':
        return self.replace(-self.feats)

    def __elemwise__(self, other: Union[torch.Tensor, 'SparseTensor'], op: callable) -> 'SparseTensor':
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = sparse_batch_broadcast(self, other)
            except Exception:
                pass
        if isinstance(other, SparseTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor

    def __add__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.add)

    def __sub__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.sub)

    def __rsub__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")

        coords = []
        feats = []
        for new_idx, old_idx in enumerate(idx):
            c = self.coords[self.layout[old_idx]]
            # clone coords to avoid aliasing; coords are int32, keep as-is
            coords.append(c.clone())
            coords[-1][:, 0] = new_idx
            feats.append(self.feats[self.layout[old_idx]])
        coords = torch.cat(coords, dim=0).contiguous()
        feats = torch.cat(feats, dim=0).contiguous()
        return SparseTensor(feats=feats, coords=coords)

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        We avoid storing GPU tensors in cache to prevent accidental retention.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        # store CPU copies for cache by default to avoid GPU retention
        try:
            if isinstance(value, torch.Tensor):
                self._spatial_cache[scale_key][key] = value.detach().cpu()
            else:
                self._spatial_cache[scale_key][key] = value
        except Exception:
            self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache. Returns CPU tensors (if present) to avoid GPU retention.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.
    Uses the dtype of `feats` to create the broadcast buffer (so it's FP16 if feats are).
    """
    coords, feats = input.coords, input.feats
    broadcasted = torch.zeros_like(feats)
    # make sure other is same dtype as feats to avoid implicit upcasts/copies
    if other.dtype != feats.dtype:
        try:
            other = other.to(dtype=feats.dtype)
        except Exception:
            pass
    for k in range(input.shape[0]):
        broadcasted[input.layout[k]] = other[k]
    return broadcasted


def sparse_batch_op(input: SparseTensor, other: torch.Tensor, op: callable = torch.add) -> SparseTensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.
    """
    return input.replace(op(input.feats, sparse_batch_broadcast(input, other)))


def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    """
    Concatenate a list of sparse tensors.
    """
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            c = input.coords.clone()
            c[:, 0] += start
            coords.append(c)
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)

    return output


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    """
    Unbind a sparse tensor along a dimension.
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]

