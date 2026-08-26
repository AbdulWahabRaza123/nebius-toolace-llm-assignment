#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/home/nebius-assignment/hf-cache
export BFCL_PROJECT_ROOT=/home/nebius-assignment/ml-ops/results/bfcl
export LOCAL_SERVER_ENDPOINT=127.0.0.1
export LOCAL_SERVER_PORT=8000
mkdir -p "$BFCL_PROJECT_ROOT"
cd /home/nebius-assignment/ml-ops
source .venv/bin/activate
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct-FC}"
CATEGORY="${2:-python}"
git -C /home/nebius-assignment/gorilla rev-parse HEAD | tee "$BFCL_PROJECT_ROOT/gorilla_commit.txt"
echo "BFCL generate model=$MODEL category=$CATEGORY"
bfcl generate \
  --model "$MODEL" \
  --test-category "$CATEGORY" \
  --backend vllm \
  --skip-server-setup \
  --num-gpus 1 \
  --allow-overwrite
echo "BFCL evaluate"
bfcl evaluate --model "$MODEL" --test-category "$CATEGORY"
echo DONE
