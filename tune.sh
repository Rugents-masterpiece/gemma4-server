#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${LLAMA_SERVER:-/home/rugent/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
NAME="${1:?usage: $0 NAME NCPU CTK CTV BATCH UBATCH [CTX] [LOAD_MODE]}"
NCPU="${2:?missing NCPU}"; CTK="${3:?missing CTK}"; CTV="${4:?missing CTV}"
BATCH="${5:?missing BATCH}"; UBATCH="${6:?missing UBATCH}"
CTX="${7:-262144}"; LOAD_MODE="${8:-auto}"
PORT="${TUNE_PORT:-8001}"
if [[ -f "$SCRIPT_DIR/.env" ]]; then set -a; source "$SCRIPT_DIR/.env"; set +a; fi
LOG="$SCRIPT_DIR/logs/tune-${NAME}.log"; JSON="$SCRIPT_DIR/logs/tune-${NAME}.json"
[[ -x "$SERVER" ]] || { echo "ERROR: llama-server not found: $SERVER" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
if ss -ltn | grep -q ":${PORT} "; then echo "ERROR: port ${PORT} is occupied" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-349f1acc-02f8-51c4-f834-b47c94e2983b}"
export LLAMA_N_CPU_MOE="$NCPU" LLAMA_CTK="$CTK" LLAMA_CTV="$CTV" LLAMA_BATCH="$BATCH" LLAMA_UBATCH="$UBATCH" LLAMA_CTX="$CTX" LLAMA_PORT="$PORT" LLAMA_LOAD_MODE="$LOAD_MODE"
ARGS=( -m "$MODEL" --host 0.0.0.0 --port "$PORT" --api-key "$LLAMA_API_KEY" -ngl all --n-cpu-moe "$NCPU" -c "$CTX" -np 2 --kv-unified-per-slot "$CTX" -fa on -ctk "$CTK" -ctv "$CTV" -b "$BATCH" -ub "$UBATCH" --cont-batching --kv-unified --load-mode "$LOAD_MODE" )
mkdir -p "$SCRIPT_DIR/logs"
{ printf 'PROFILE name=%s ncpu=%s ctk=%s ctv=%s batch=%s ubatch=%s ctx=%s load_mode=%s\n' "$NAME" "$NCPU" "$CTK" "$CTV" "$BATCH" "$UBATCH" "$CTX" "$LOAD_MODE"; exec "$SERVER" "${ARGS[@]}"; } >"$LOG" 2>&1 &
PID=$!
peak_vram=0; peak_ram=0; peak_swap=0; ready=0; ready_at=0; start=$(date +%s); end=$((start+240))
while kill -0 "$PID" 2>/dev/null && (( $(date +%s) < end )); do
  line=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | grep 'RTX 4060 Ti' || true)
  used=$(printf '%s\n' "$line" | awk -F',' '{gsub(/ MiB/,"",$2); print $2+0}')
  (( used > peak_vram )) && peak_vram=$used || true
  read -r ram swap < <(free -m | awk '/^Mem:/{ram=$3} /^Swap:/{swap=$3} END{print ram+0,swap+0}')
  (( ram > peak_ram )) && peak_ram=$ram || true
  (( swap > peak_swap )) && peak_swap=$swap || true
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${PORT}$"; then ready=1; (( ready_at == 0 )) && ready_at=$(date +%s); fi
  (( ready == 1 && $(date +%s) - ready_at >= 10 )) && break
  sleep 1
done
status=running
if ! kill -0 "$PID" 2>/dev/null; then status=exited; fi
if kill -0 "$PID" 2>/dev/null; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
printf '{"name":"%s","ready":%s,"status":"%s","peak_vram_mib":%s,"peak_ram_mib":%s,"peak_swap_mib":%s,"log":"%s"}\n' "$NAME" "$([[ $ready == 1 ]] && echo true || echo false)" "$status" "$peak_vram" "$peak_ram" "$peak_swap" "$LOG" | tee "$JSON"
