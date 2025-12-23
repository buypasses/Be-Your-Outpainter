# MPS (Apple Silicon) Compatibility

This fork has been modified to support Apple Silicon (M1/M2/M3/M4) using MPS (Metal Performance Shaders) as the GPU backend.

## Environment

- **Hardware**: Apple Silicon (tested on M4 Pro, 64GB RAM)
- **Python**: 3.12
- **PyTorch**: 2.5+ with MPS support

## Changes Made

### 1. `src/data/naive_dataset.py`
- **Issue**: `decord` is not available for Python 3.12 on macOS
- **Fix**: Replaced `decord` with a custom `CV2VideoReader` class using OpenCV
- **Details**: Implements `__len__` and `get_batch(indices)` to match decord's API

### 2. `src/pipelines/pipelineoutpaint.py`
- **Issue**: `diffusers.pipeline_utils` import path changed in newer versions
- **Fix**: Changed `from diffusers.pipeline_utils import DiffusionPipeline` to `from diffusers import DiffusionPipeline`
- **Issue**: `enable_sequential_cpu_offload` hardcoded CUDA device
- **Fix**: Added device detection logic to prefer CUDA, fallback to MPS, then CPU

### 3. `src/schedulers/scheduling_ddim.py`
- **Issue**: `randn_tensor` import path changed in newer diffusers versions
- **Fix**: Changed import to `from diffusers.utils.torch_utils import randn_tensor`

### 4. `scripts/train_outpaint.py`
- **Issue**: `torch.autocast("cuda")` not supported on MPS
- **Fix**: Added `get_autocast_context(device)` helper function that returns appropriate context:
  - CUDA: `torch.autocast("cuda")`
  - MPS: `torch.autocast("mps")` with fallback to `nullcontext()`
  - CPU: `nullcontext()`
- **Issue**: `torch.cuda.max_memory_allocated()` not available on MPS
- **Fix**: Added conditional memory tracking that prints info message on MPS

### 5. `src/models/pretrain/attention.py`
- **Issue**: `xformers` requires CUDA and is not available on MPS
- **Fix**: Modified `set_use_memory_efficient_attention_xformers()` to return early on non-CUDA devices with info message
- **Issue**: `_slice_size` attribute not initialized
- **Fix**: Added `self._slice_size = None` in `CrossAttention.__init__`

### 6. `src/models/vanilla/attention.py`
- **Issue**: Same xformers issue as pretrain/attention.py
- **Fix**: Same fix - return early on non-CUDA devices instead of raising errors

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (MPS-compatible versions)
pip install -r requirements-mps.txt

# Download models from HuggingFace
# (see README.md for model download instructions)
```

## Usage Notes

- xformers acceleration is not available on MPS; standard attention will be used instead
- Memory tracking during training shows "MPS backend used - memory tracking not available"
- autocast may fall back to no mixed precision if MPS autocast is not supported
- Performance may vary compared to CUDA; adjust batch sizes as needed

## Known Limitations

1. **No xformers**: Memory-efficient attention via xformers is CUDA-only
2. **Limited autocast**: MPS autocast support varies by PyTorch version
3. **Memory tracking**: `torch.cuda.max_memory_allocated()` not available on MPS
4. **Some operations**: Certain CUDA-specific operations may not be available

## Contributing

When adding new CUDA-specific code, please ensure MPS compatibility by:
1. Using `torch.cuda.is_available()` checks before CUDA-only operations
2. Providing MPS fallbacks using `torch.backends.mps.is_available()`
3. Using device-agnostic tensor creation with explicit `device=` parameter
