#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-349f1acc-02f8-51c4-f834-b47c94e2983b}"
PORT="${LLAMA_PORT:-8001}"

# Find the running gemma server process command line
PIDS=$(pgrep -f "llama-server.*--port ${PORT}( |$)" || true)
CMD=""
for p in $PIDS; do CMD="$CMD $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)"; done

echo '=== GPU ==='; nvidia-smi
echo '=== RAM ==='; free -h
printf 'RAM used/available: '; free -h | awk '/^Mem:/ {print $3 "/" $7}'
printf 'Swap used/free: '; free -h | awk '/^Swap:/ {print $3 "/" $4}'
vmstat 1 2 | tail -1 | awk '{printf "vmstat si/so: %s/%s KiB/s\n", $7, $8}'
echo
echo '=== PROFILE ==='
getarg() { # name pattern
  echo "$CMD" | grep -oE "\-$1 [a-z0-9_.-]+" | awk "{print \$2}" | head -1
}
CTX="$(echo "$CMD" | grep -oE '\-\-kv-unified-per-slot [0-9]+' | awk '{print $2}' | head -1)"
[[ -z "$CTX" ]] && CTX="$(echo "$CMD" | grep -oE '\-c [0-9]+' | awk '{print $2}' | head -1)"
echo "Model: $(echo "$CMD" | grep -oE '\-m [^ ]+' | awk '{print $2}' | head -1)"
echo "Service: gemma4.service"
echo "PID: ${PIDS:-unknown}"
echo "Context per slot: ${CTX:-unknown}"
PARALLEL_LIVE="$(echo "$CMD" | grep -oE '\-np [0-9]+' | awk '{print $2}' | head -1)"
echo "Parallel: ${PARALLEL_LIVE:-unknown}"
if [[ -n "$CTX" && -n "$PARALLEL_LIVE" ]]; then echo "True KV pool: $((CTX * PARALLEL_LIVE))"; else echo "True KV pool: unknown"; fi
echo "Unified KV: $(echo "$CMD" | grep -qE '\-\-kv-unified($| )' && echo yes || echo no)"
echo "KV K: $(echo "$CMD" | grep -oE '\-ctk [a-z0-9_]+' | awk '{print $2}' | head -1)"
echo "KV V: $(echo "$CMD" | grep -oE '\-ctv [a-z0-9_]+' | awk '{print $2}' | head -1)"
echo "Flash Attention: $(echo "$CMD" | grep -qE '\-fa on' && echo on || echo off/auto)"
echo "Fit: $(echo "$CMD" | grep -oE '\-\-fit (on|off)' | awk '{print $2}' | head -1)"
CPU_MOE_LIVE="$(echo "$CMD" | grep -oE '\-\-n-cpu-moe [0-9]+' | awk '{print $2}' | head -1)"
echo "CPU MoE layers: ${CPU_MOE_LIVE:-unknown}"
if [[ "$CPU_MOE_LIVE" == 6 ]]; then echo "GPU MoE: 24 / 30 (selected baseline)"; else echo "GPU MoE: derived/unknown"; fi
echo "Batch: $(echo "$CMD" | grep -oE '\-b [0-9]+' | awk '{print $2}' | head -1)"
echo "Ubatch: $(echo "$CMD" | grep -oE '\-ub [0-9]+' | awk '{print $2}' | head -1)"
MMP=$(echo "$CMD" | grep -oE '\-\-mmproj [^ ]+' | awk '{print $2}' | head -1)
if [[ -n "$MMP" ]]; then
  echo "Vision: enabled"
  echo "mmproj: $MMP"
  echo "mmproj offload: $(echo "$CMD" | grep -qE '\-\-no-mmproj-offload' && echo CPU || echo GPU)"
else
  echo "Vision: disabled"
fi
if [[ -n "${LLAMA_API_KEY:-}" && "$LLAMA_API_KEY" != CHANGE_ME ]]; then
  echo '=== API /v1/models ==='
  curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/v1/models" -H "Authorization: Bearer $LLAMA_API_KEY" || true
  echo
fi
# Vision smoke test (optional, uncomment to run):
# bash "$SCRIPT_DIR/vision-smoke.sh" || echo "VISION SMOKE FAIL"
echo "=== SERVICE ==="
systemctl is-active gemma4.service 2>/dev/null || echo "gemma4.service inactive"
