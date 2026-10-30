#!/usr/bin/env bash
# Stability-first Gemma 4 profile: vision projector on CPU, Q8 KV, reduced context.
# Use when the full 2x196608 profile hits CUDA OOM under load.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
: "${LLAMA_API_KEY:?Set LLAMA_API_KEY in .env or environment}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/rugent/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
MMPROJ="${LLAMA_MMPROJ:-/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf}"
PORT="${LLAMA_PORT:-8001}"
CTX="${LLAMA_CTX:-131072}"
PARALLEL="${LLAMA_PARALLEL:-2}"
NCPU="${LLAMA_N_CPU_MOE:-5}"
CTK="${LLAMA_CTK:-q8_0}"
CTV="${LLAMA_CTV:-q8_0}"
BATCH="${LLAMA_BATCH:-128}"
UBATCH="${LLAMA_UBATCH:-1024}"
[[ -f "$MMPROJ" ]] || { echo "ERROR: mmproj not found: $MMPROJ" >&2; exit 1; }
if ss -ltn | grep -q ":${PORT} "; then echo "ERROR: port ${PORT} already listening" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-349f1acc-02f8-51c4-f834-b47c94e2983b}"
API_KEY_FILE="$(mktemp /tmp/gemma4-api-key.XXXXXX)"
chmod 600 "$API_KEY_FILE"
printf '%s\n' "$LLAMA_API_KEY" > "$API_KEY_FILE"
trap 'rm -f "$API_KEY_FILE"' EXIT
printf '%s\n' "Gemma 4 SAFE profile" "Context per slot: $CTX" "Unified pool: $((PARALLEL * CTX))" "Parallel: $PARALLEL" "CPU MoE: $NCPU" "KV: K=$CTK V=$CTV" "Vision projector: CPU" "Batch $BATCH / ubatch $UBATCH"
exec "$LLAMA_SERVER" -m "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --alias gemma-4 --api-key-file "$API_KEY_FILE" \
  -ngl all --n-cpu-moe "$NCPU" \
  -np "$PARALLEL" --kv-unified --kv-unified-per-slot "$CTX" \
  -fa on -ctk "$CTK" -ctv "$CTV" \
  -b "$BATCH" -ub "$UBATCH" --cont-batching --load-mode auto \
  --mmproj "$MMPROJ" --no-mmproj-offload
