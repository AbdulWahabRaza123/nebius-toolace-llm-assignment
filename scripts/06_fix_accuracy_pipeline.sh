#!/usr/bin/env bash
# End-to-end accuracy fix: re-prepare BFCL-aligned data, LoRA train, merge, serve, BFCL.
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/home/nebius-assignment/hf-cache
cd /home/nebius-assignment/ml-ops
source .venv/bin/activate
source scripts/load_env.sh || true

pkill -f 'python src/train/sft.py' || true
pkill -f 'vllm serve' || true
sleep 5

mkdir -p logs

echo "=== 1) Patch ToolACE-8B chat template from Meta Llama 3.1 Instruct ==="
python - <<'PY'
import json
from pathlib import Path
meta = Path.home() / "hf-cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
snaps = list(meta.glob("*/tokenizer_config.json"))
if not snaps:
    raise SystemExit("Meta Llama tokenizer_config not found in HF cache")
meta_cfg = json.loads(snaps[0].read_text())
toolace = Path("models/toolace-8b/tokenizer_config.json")
t = json.loads(toolace.read_text())
old_len = len(t.get("chat_template") or "")
t["chat_template"] = meta_cfg["chat_template"]
toolace.write_text(json.dumps(t, indent=2) + "\n")
print(f"chat_template {old_len} -> {len(t['chat_template'])}")
PY

echo "=== 2) Re-prepare ToolACE in BFCL FC JSON format ==="
python src/data/prepare_toolace.py --config configs/data.yaml | tee logs/prepare_bfcl.log
python - <<'PY'
import json
from pathlib import Path
from datasets import load_from_disk
from src.train.sft import render_bfcl_llama31
stats = json.loads(Path("data/processed/stats.json").read_text())
print("STATS", stats)
ds = load_from_disk("data/processed/toolace_llama31")
json_hits = 0
for i in range(min(80, len(ds["train"]))):
    text = render_bfcl_llama31(ds["train"][i])
    if '"name"' in text and '"parameters"' in text:
        json_hits += 1
        if json_hits == 1:
            idx = text.rfind("<|start_header_id|>assistant<|end_header_id|>")
            print("ASSISTANT_TAIL", repr(text[idx:idx+500]))
            print("tools_in_row", len(json.loads(ds["train"][i].get("tools_json") or "[]")))
assert json_hits >= 10, json_hits
print("json_hits_in_80", json_hits)
PY

echo "=== 3) LoRA train (BFCL-aligned) ==="
: > logs/train_lora_bfcl.log
python src/train/sft.py --config configs/train_lora.yaml 2>&1 | tee logs/train_lora_bfcl.log

echo "=== 4) Merge LoRA ==="
python src/train/merge_lora.py \
  --base-model models/toolace-8b \
  --adapter checkpoints/lora_bfcl/final \
  --output checkpoints/merged-lora-bfcl

echo "=== 5) Serve merged ==="
# point serve script at new merge
pkill -f 'vllm serve' || true
sleep 3
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
    echo VLLM_READY
    break
  fi
  if grep -qE 'CUDA out of memory|Address already in use' logs/vllm.log; then
    echo VLLM_FAIL
    tail -40 logs/vllm.log
    exit 1
  fi
  sleep 5
done

echo "=== 6) BFCL Python generate + evaluate ==="
bash scripts/05_run_bfcl.sh 2>&1 | tee logs/bfcl_after_fix.log
echo PIPELINE_DONE
