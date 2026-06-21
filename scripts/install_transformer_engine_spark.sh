#!/usr/bin/env bash
# Build Transformer Engine on DGX Spark (GB10, aarch64, CUDA 13).
# Fixes: cudnn.h / nccl.h not found under /usr/local/cuda (headers live in pip nvidia-* wheels).
#
# NOTE: TE is NOT required for Gyanateet_tracking Stage 2 SSL — use meta.dtype: bfloat16.
# After install, verify: python -c "import transformer_engine.pytorch as te"
# If you see libcublasLt undefined symbols, your host cuBLAS is older than TE 2.15 expects;
# use bf16 training or an NGC container (nvcr.io/nvidia/vllm:26.03-py3) for TE workloads.

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate conda env first, e.g.: conda activate surgi_track"
  exit 1
fi

PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NCCL_HOME="${CONDA_PREFIX}/lib/python${PYVER}/site-packages/nvidia/nccl"
CUDNN_PATH="${CONDA_PREFIX}/lib/python${PYVER}/site-packages/nvidia/cudnn"

export CUDA_HOME
export NCCL_HOME
export CUDNN_PATH
export NVTE_CUDA_INCLUDE_PATH="${CUDA_HOME}/include:${CUDNN_PATH}/include:${NCCL_HOME}/include"
export CPATH="${NVTE_CUDA_INCLUDE_PATH}:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CPATH}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${CUDNN_PATH}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# GB10 (sm_121): compile PTX for compute_120, JIT at runtime
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export MAX_JOBS="${MAX_JOBS:-4}"

pip install -U pip setuptools wheel ninja
pip install --no-build-isolation 'transformer_engine[pytorch]==2.15.0'

echo ""
echo "Build finished. Testing import..."
python -c "import transformer_engine.pytorch as te; print('Transformer Engine OK:', te.__name__)"
