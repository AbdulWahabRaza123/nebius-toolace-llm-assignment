#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/home/nebius-assignment/hf-cache
cd /home/nebius-assignment/ml-ops
source .venv/bin/activate

# Stop previous jobs
pkill -f 'bfcl generate' || true
pkill -f '05_run_bfcl' || true
pkill -f 'vllm.entrypoints' || true
sleep 4

mkdir -p logs
nohup vllm serve checkpoints/merged-lora \
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
    exit 0
  fi
  sleep 5
done
echo TIMEOUT
tail -40 logs/vllm.log
exit 1
