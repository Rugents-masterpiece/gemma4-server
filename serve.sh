#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
: "${LLAMA_API_KEY:?Set LLAMA_API_KEY in .env or environment}"
LLAMA_SERVER="${LLAMA_SERVER:-/home/rugent/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
MMPROJ="${LLAMA_MMPROJ:-/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf}"
MMPROJ_OFFLOAD="${LLAMA_MMPROJ_OFFLOAD:-0}"
PORT="${LLAMA_PORT:-8001}"
CTX="${LLAMA_CTX:-131072}"
PARALLEL="${LLAMA_PARALLEL:-2}"
NCPU="${LLAMA_N_CPU_MOE:-6}"
MOE_ARGS=(--n-cpu-moe "$NCPU")
if [[ "${LLAMA_ALL_CPU_MOE:-0}" == 1 ]]; then MOE_ARGS=(--cpu-moe); fi
CTK="${LLAMA_CTK:-q8_0}"; CTV="${LLAMA_CTV:-q5_1}"
LOAD_MODE="${LLAMA_LOAD_MODE:-auto}"
FIT="${LLAMA_FIT:-off}"
BATCH="${LLAMA_BATCH:-128}"; UBATCH="${LLAMA_UBATCH:-1024}"
KV_ARGS=(-np "$PARALLEL" --kv-unified --kv-unified-per-slot "$CTX")
if [[ "${LLAMA_USE_C:-0}" == 1 ]]; then
  KV_ARGS=(-np "$PARALLEL" -c "$CTX" --kv-unified --kv-unified-per-slot "$CTX")
fi
MMPROJ_ARGS=(--mmproj "$MMPROJ")
if [[ "$MMPROJ_OFFLOAD" != "1" ]]; then MMPROJ_ARGS+=(--no-mmproj-offload); fi
[[ -x "$LLAMA_SERVER" ]] || { echo "ERROR: llama-server not found: $LLAMA_SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ "$LLAMA_API_KEY" != CHANGE_ME ]] || { echo "ERROR: replace CHANGE_ME API key" >&2; exit 1; }
if ss -ltn | grep -q ":${PORT} "; then echo "ERROR: port ${PORT} is already listening" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-349f1acc-02f8-51c4-f834-b47c94e2983b}"
mkdir -p "$SCRIPT_DIR/logs"
printf '%s\n' "Gemma 4 server" "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "Model: $MODEL" "Vision: mmproj=$MMPROJ offload=$([[ $MMPROJ_OFFLOAD == 1 ]] && echo GPU || echo CPU)" "Context per slot: $CTX" "Unified pool: $((PARALLEL * CTX)) (auto-sized, no -c)" "Parallel slots: $PARALLEL" "CPU MoE layers: $NCPU" "KV: K=$CTK V=$CTV" "Batch: $BATCH ubatch=$UBATCH" "Load mode: $LOAD_MODE" "Flash Attention: on" "Port: $PORT"
API_KEY_FILE="${LLAMA_API_KEY_FILE:-}"
GENERATED_API_KEY_FILE=0
if [[ -z "$API_KEY_FILE" ]]; then
  API_KEY_FILE="$(mktemp /tmp/gemma4-api-key.XXXXXX)"
  chmod 600 "$API_KEY_FILE"
  printf '%s\n' "$LLAMA_API_KEY" > "$API_KEY_FILE"
  GENERATED_API_KEY_FILE=1
else
  [[ -f "$API_KEY_FILE" ]] || { echo "ERROR: API key file not found: $API_KEY_FILE" >&2; exit 1; }
fi
cleanup_api_key_file() {
  if [[ "$GENERATED_API_KEY_FILE" == 1 ]]; then rm -f -- "$API_KEY_FILE"; fi
}
trap cleanup_api_key_file EXIT
"$LLAMA_SERVER" -m "$MODEL" --host 0.0.0.0 --port "$PORT" --alias gemma-4 --api-key-file "$API_KEY_FILE" -ngl all "${MOE_ARGS[@]}" "${KV_ARGS[@]}" -fa on -ctk "$CTK" -ctv "$CTV" -b "$BATCH" -ub "$UBATCH" --cont-batching --load-mode "$LOAD_MODE" --fit "$FIT" "${MMPROJ_ARGS[@]}"
