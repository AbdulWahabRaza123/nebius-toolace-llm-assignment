#!/usr/bin/env bash
# One-time setup on the Nebius 1xH100 VM. Run from the cloned repo root:
#   bash scripts/00_setup_vm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> GPU"
nvidia-smi

echo "==> Directories"
mkdir -p "$ROOT/models" "$ROOT/checkpoints" "$ROOT/data/processed" "$ROOT/results/bfcl" "$ROOT/logs" "$HOME/hf-cache"
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Llama 3.1 download will fail until you export it." >&2
fi

echo "==> Python venv"
python3 -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "==> vLLM + PyTorch CUDA wheel (pulls a compatible torch)"
pip install --upgrade vllm

echo "==> Assignment Python stack"
pip install -r "$ROOT/requirements.txt"

echo "==> Hugging Face CLI login (uses HF_TOKEN if present)"
if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential || true
fi

echo "==> Clone official BFCL from GitHub"
if [[ ! -d "$HOME/gorilla/.git" ]]; then
  git clone https://github.com/ShishirPatil/gorilla.git "$HOME/gorilla"
else
  git -C "$HOME/gorilla" pull --ff-only || true
fi
pip install -e "$HOME/gorilla/berkeley-function-call-leaderboard"
git -C "$HOME/gorilla" rev-parse HEAD | tee "$ROOT/results/bfcl/gorilla_commit.txt"

echo "==> Smoke imports"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import transformers, datasets, peft, trl, vllm
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("peft", peft.__version__)
print("trl", trl.__version__)
print("vllm", vllm.__version__)
PY

echo "Setup complete. Next:"
echo "  source $ROOT/.venv/bin/activate"
echo "  bash scripts/01_download_assets.sh"
echo "  python src/data/prepare_toolace.py --config configs/data.yaml"
