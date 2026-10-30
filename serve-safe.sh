#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
: "${LLAMA_API_KEY:?Set LLAMA_API_KEY in .env or environment}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/rugent/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
PORT="${LLAMA_SAFE_PORT:-8001}"; CTX="${LLAMA_SAFE_CTX:-131072}"
POOL=$((CTX * 2))
[[ -x "$LLAMA_SERVER" ]] || { echo "ERROR: llama-server not found: $LLAMA_SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ "$LLAMA_API_KEY" != CHANGE_ME ]] || { echo "ERROR: replace CHANGE_ME API key" >&2; exit 1; }
if ss -ltn | grep -q ":${PORT} "; then echo "ERROR: port ${PORT} is already listening" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-349f1acc-02f8-51c4-f834-b47c94e2983b}"
mkdir -p "$SCRIPT_DIR/logs"
printf '%s\n' "Gemma 4 safe server" "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "Model: $MODEL" "Context per slot: $CTX" "Context pool: $POOL" "Parallel slots: 2" "All MoE experts: CPU" "KV: K=q8_0 V=q8_0" "Batch: 512 ubatch=128" "Flash Attention: on" "Port: $PORT"
API_KEY_FILE="$(mktemp /tmp/gemma4-safe-api-key.XXXXXX)"; chmod 600 "$API_KEY_FILE"; printf '%s\n' "$LLAMA_API_KEY" > "$API_KEY_FILE"
cleanup() { rm -f -- "$API_KEY_FILE"; }
trap cleanup EXIT
"$LLAMA_SERVER" -m "$MODEL" --host 0.0.0.0 --port "$PORT" --alias gemma-4 --api-key-file "$API_KEY_FILE" -ngl all --cpu-moe -c "$CTX" -np 2 --kv-unified-per-slot "$CTX" -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 128 --cont-batching --kv-unified
