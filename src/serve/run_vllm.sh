#!/usr/bin/env bash
# Production-style vLLM OpenAI server (Token Factory Dedicated Endpoint analogue).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

MODEL_PATH="${1:-${ROOT}/checkpoints/merged-lora}"
SERVED_NAME="${2:-llama-3.1-8b-toolace}"
PORT="${PORT:-8000}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  echo "Merge LoRA first: python src/train/merge_lora.py --adapter checkpoints/lora/final --output checkpoints/merged-lora" >&2
  exit 1
fi

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
