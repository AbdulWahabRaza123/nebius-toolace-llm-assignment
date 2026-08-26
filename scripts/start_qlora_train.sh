#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/home/nebius-assignment/hf-cache
cd /home/nebius-assignment/ml-ops
# shellcheck disable=SC1091
source scripts/load_env.sh || true
source .venv/bin/activate

# Free GPU from vLLM / prior jobs
pkill -f 'vllm serve' || true
pkill -f 'VLLM::EngineCore' || true
pkill -f 'bfcl generate' || true
pkill -f 'src/train/sft.py' || true
sleep 8

mkdir -p logs checkpoints/qlora_bfcl
: > logs/train_qlora_bfcl.log
nohup python src/train/sft.py --config configs/train_qlora.yaml > logs/train_qlora_bfcl.log 2>&1 &
echo "TRAIN_PID=$!"
sleep 60
tail -50 logs/train_qlora_bfcl.log
