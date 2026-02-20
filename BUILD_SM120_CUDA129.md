# Build vLLM for SM120 with CUDA 12.9

This guide is for building vLLM from source on Blackwell-class GPUs (`sm_120`) in a Conda environment.

## Why this is needed

- `sm_120` requires a recent CUDA toolchain.
- Building with `TORCH_CUDA_ARCH_LIST="8.9;9.0+PTX"` can build, but may fail at runtime with:
  - `CUDA error: no kernel image is available for execution on the device`
- For native `sm_120` support, build with CUDA 12.9 and arch `12.0`.

## Prerequisites

- Conda env (example: `vllm`)
- `torch` installed with CUDA 12.9 (`+cu129`)
- `nvcc` 12.9 available in the env

Check:

```bash
which nvcc
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_capability())"
```

Expected:
- `nvcc` shows `release 12.9`
- `torch.version.cuda` is `12.9`
- capability is `(12, 0)` on SM120 GPUs

## Install matching PyTorch (if needed)

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

## Conda CUDA include/lib layout fix

In Conda, CUDA headers are under:

`$CONDA_PREFIX/targets/x86_64-linux/include`

If CMake fails with `cuda_runtime.h: No such file or directory`, use the following environment setup before building:

```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CUDAToolkit_ROOT=$CONDA_PREFIX/targets/x86_64-linux
export NVCC_PREPEND_FLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include"
export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc -DCUDAToolkit_ROOT=$CONDA_PREFIX/targets/x86_64-linux -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux"
```

Optional sanity check:

```bash
ls $CONDA_PREFIX/targets/x86_64-linux/include/cuda_runtime.h
```

## Build vLLM from source

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
rm -rf build/ .deps/ ~/.cache/torch_extensions ~/.cache/vllm
pip install -e . --no-build-isolation
```

## Verify build

```bash
python -c "import vllm, torch; print(vllm.__version__); print(torch.cuda.get_device_capability())"
```

## Run API server (example)

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.25 \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8030 \
  --reasoning-parser qwen3
```

## Common failure modes

- `no kernel image is available for execution on the device`
  - Cause: built for old archs only (`8.9/9.0`) on an `sm_120` GPU.
  - Fix: rebuild with `TORCH_CUDA_ARCH_LIST="12.0"` and CUDA 12.9.

- `cuda_runtime.h: No such file or directory`
  - Cause: Conda CUDA include path not picked by CMake.
  - Fix: set `CUDACXX`, `CUDAToolkit_ROOT`, and `NVCC_PREPEND_FLAGS` as shown above.
