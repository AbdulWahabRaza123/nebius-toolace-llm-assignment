#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/load_env.sh"
source "$ROOT/.venv/bin/activate"

METHOD="${1:-lora}"
case "$METHOD" in
  lora) CONFIG="$ROOT/configs/train_lora.yaml" ;;
  qlora) CONFIG="$ROOT/configs/train_qlora.yaml" ;;
  full) CONFIG="$ROOT/configs/train_full.yaml" ;;
  *) echo "Usage: $0 {lora|qlora|full}" >&2; exit 1 ;;
esac

LOG="$ROOT/logs/train_${METHOD}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/logs"
echo "Training $METHOD -> $LOG"
python "$ROOT/src/train/sft.py" --config "$CONFIG" 2>&1 | tee "$LOG"
