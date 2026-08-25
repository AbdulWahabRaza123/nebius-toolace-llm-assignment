#!/usr/bin/env bash
# Download official Hugging Face dataset + base model onto the VM (not the laptop).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
mkdir -p "$ROOT/models/llama-3.1-8b-instruct"

echo "==> Dataset Team-ACE/ToolACE"
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("Team-ACE/ToolACE", split="train")
print("ToolACE rows:", len(ds), "columns:", ds.column_names)
print("example system chars:", len(ds[0]["system"]))
print("example turns:", len(ds[0]["conversations"]))
PY

echo "==> Base model meta-llama/Meta-Llama-3.1-8B-Instruct"
if command -v hf >/dev/null 2>&1; then
  hf download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir "$ROOT/models/llama-3.1-8b-instruct"
else
  huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir "$ROOT/models/llama-3.1-8b-instruct"
fi

echo "Downloads finished."
