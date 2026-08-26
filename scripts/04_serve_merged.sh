#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/home/nebius-assignment/hf-cache
cd /home/nebius-assignment/ml-ops
source .venv/bin/activate

# Stop previous server (patterns must be separate for pkill)
pkill -f 'vllm serve' || true
pkill -f 'VLLM::EngineCore' || true
pkill -f 'bfcl generate' || true
sleep 5
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8000/tcp 2>/dev/null || true
  sleep 2
fi

mkdir -p logs
: > logs/vllm.log
nohup vllm serve checkpoints/merged-lora-bfcl \
  --served-model-name llama-3.1-8b-toolace meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  > logs/vllm.log 2>&1 &

for i in $(seq 1 90); do
  if grep -Fq 'Application startup complete' logs/vllm.log; then
    echo READY
    grep -E 'max_model_len|Application startup' logs/vllm.log | tail -8
    exit 0
  fi
  if grep -qE 'CUDA out of memory|Address already in use|ValueError:.*max_model_len' logs/vllm.log; then
    echo FAIL
    tail -80 logs/vllm.log
    exit 1
  fi
  sleep 5
done
echo TIMEOUT
tail -40 logs/vllm.log
exit 1
