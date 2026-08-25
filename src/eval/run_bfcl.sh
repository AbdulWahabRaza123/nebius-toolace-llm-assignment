#!/usr/bin/env bash
# Clone official BFCL (GitHub) if needed and evaluate the Python subset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

GORILLA_DIR="${GORILLA_DIR:-$HOME/gorilla}"
MODEL_NAME="${1:-llama-3.1-8b-toolace-FC}"
CATEGORY="${2:-python}"
ENDPOINT="${LOCAL_SERVER_ENDPOINT:-127.0.0.1}"
PORT="${LOCAL_SERVER_PORT:-8000}"

if [[ ! -d "$GORILLA_DIR/.git" ]]; then
  git clone https://github.com/ShishirPatil/gorilla.git "$GORILLA_DIR"
fi

pushd "$GORILLA_DIR/berkeley-function-call-leaderboard" >/dev/null
pip install -e . >/dev/null
COMMIT="$(git rev-parse HEAD)"
echo "BFCL gorilla commit=$COMMIT"
popd >/dev/null

export BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-$ROOT/results/bfcl}"
mkdir -p "$BFCL_PROJECT_ROOT"
export LOCAL_SERVER_ENDPOINT="$ENDPOINT"
export LOCAL_SERVER_PORT="$PORT"

echo "Generating BFCL responses for $MODEL_NAME category=$CATEGORY"
bfcl generate \
  --model "$MODEL_NAME" \
  --test-category "$CATEGORY" \
  --backend vllm \
  --skip-server-setup \
  --num-gpus 1

echo "Scoring..."
bfcl evaluate --model "$MODEL_NAME" --test-category "$CATEGORY"

echo "Results under $BFCL_PROJECT_ROOT"
echo "$COMMIT" > "$BFCL_PROJECT_ROOT/gorilla_commit.txt"
