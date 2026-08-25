#!/usr/bin/env bash
# Load secrets for Hugging Face (create .env on the VM with HF_TOKEN=hf_...)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
fi
