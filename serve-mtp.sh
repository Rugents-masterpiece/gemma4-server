#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
: "${LLAMA_API_KEY:?Set LLAMA_API_KEY in .env or environment}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/rugent/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
DRAFT="${DRAFT:-/home/rugent/models/gemma-4-26B-A4B-QAT/mtp-gemma-4-26B-A4B-it.gguf}"
PORT="${LLAMA_MTP_PORT:-8003}"
[[ -x "$LLAMA_SERVER" && -f "$MODEL" && -f "$DRAFT" ]] || { echo "ERROR: missing llama-server, target, or draft" >&2; exit 1; }
if ss -ltn | grep -q ":${PORT} "; then echo "ERROR: port ${PORT} is already listening" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-349f1acc-02f8-51c4-f834-b47c94e2983b}"
API_KEY_FILE="$(mktemp /tmp/gemma4-mtp-api-key.XXXXXX)"; chmod 600 "$API_KEY_FILE"; printf '%s\n' "$LLAMA_API_KEY" > "$API_KEY_FILE"
cleanup() { rm -f -- "$API_KEY_FILE"; }
trap cleanup EXIT
"$LLAMA_SERVER" -m "$MODEL" --model-draft "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 4 --host 0.0.0.0 --port "$PORT" --alias gemma-4-mtp --api-key-file "$API_KEY_FILE" -ngl all --n-cpu-moe 28 -c 32768 -np 1 -fa on -ctk q4_1 -ctv q4_1 -b 128 -ub 64
