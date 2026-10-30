#!/usr/bin/env bash
# Background VRAM/RAM sampler for profile logs. Usage: vram-sampler.sh <label>
# Writes CSV to logs/vram-<label>.csv, samples both GPUs + free every 1s.
set -u
LABEL="${1:-run}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$DIR"
OUT="$DIR/vram-$LABEL.csv"
echo "ts,gpu_idx,uuid,mem_used_mib,mem_free_mib,util_pct,power_w,ram_used_mib,swap_used_mib" > "$OUT"
while true; do
  RAM=$(free -m | awk '/^Mem/{print $3}'); SW=$(free -m | awk '/^Swap/{print $3}')
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits | while IFS=, read -r idx uuid mu mf util pw; do
    echo "$(date '+%F %T'),${idx},$(echo "$uuid"|tr -d ' '),${mu},${mf},${util},${pw},${RAM},${SW}" >> "$OUT"
  done
  sleep 1
done