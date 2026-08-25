#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

python src/data/prepare_toolace.py --config configs/data.yaml
echo "Prepared data. Train with:"
echo "  python src/train/sft.py --config configs/train_lora.yaml"
echo "  python src/train/sft.py --config configs/train_qlora.yaml"
echo "  python src/train/sft.py --config configs/train_full.yaml"
