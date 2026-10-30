#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-/home/rugent/gemma4-server/logs/memory-monitor}"
INTERVAL="${INTERVAL:-1}"
DURATION="${DURATION:-0}"
mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUTDIR/memory-$STAMP.log"

exec > >(tee -a "$LOG") 2>&1
printf 'timestamp\tram_used_mib\tram_available_mib\tswap_used_mib\tswap_free_mib\tsi_kib_s\tso_kib_s\tgpu_used_mib\tgpu_free_mib\tgpu_util_pct\n'

start=$(date +%s)
while :; do
  now=$(date --iso-8601=seconds)
  read -r _ total used free shared buff_cache available < <(free -m | awk '/^Mem:/ {print $1,$2,$3,$4,$5,$6,$7}')
  read -r _ stotal sused sfree _ < <(free -m | awk '/^Swap:/ {print $1,$2,$3,$4,$5}')
  [[ -n "${stotal:-}" ]] || { stotal=0; sused=0; sfree=0; }
  vm=$(vmstat 1 2 | tail -1)
  si=$(awk '{print $7}' <<< "$vm")
  so=$(awk '{print $8}' <<< "$vm")
  gpu=$(nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1 ~ /4060 Ti/ {print $2,$3,$4; exit}')
  read -r gu gf gi <<< "${gpu:-0 0 0}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "${used:-0}" "${available:-0}" "${sused:-0}" "${sfree:-0}" "${si:-0}" "${so:-0}" "${gu:-0}" "${gf:-0}" "${gi:-0}"
  if (( DURATION > 0 && $(date +%s) - start >= DURATION )); then break; fi
  sleep "$INTERVAL"
done
printf 'log=%s\n' "$LOG"
