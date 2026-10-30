#!/usr/bin/env bash
# Q8 true-dual ladder + retrieval, exact tokens. Usage: test-exact.sh ... (ladder180 mode)
set -uo pipefail
cd /home/rugent/gemma4-server
CTX=180224; K=q8_0; V=q8_0; NCPU=9; MODE=ladder180
MODEL="/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
MMPROJ="/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf"
LS=/home/rugent/llama.cpp/build/bin/llama-server
P="exact-ladder-${CTX}-${K}-${V}-m${NCPU}"
SLOG="logs/server-$P.log"; RLOG="logs/results-$P.tsv"
KEYFILE=/tmp/gemma4-test-key
[[ -f "$KEYFILE" && $(wc -c < "$KEYFILE") -ge 16 ]] || grep '^LLAMA_API_KEY' .env | cut -d= -f2 > "$KEYFILE"
stop_server() {
  pkill -f "port 18001" 2>/dev/null || true
  sleep 5
}
restore_prod() {
  if [[ ${RESTORED:-0} == 1 ]]; then return 0; fi
  RESTORED=1
  stop_server
  systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service 2>/dev/null || true
  sudo -n systemctl restart gemma4.service 2>/dev/null || systemctl restart gemma4.service 2>/dev/null || true
  sleep 5
  for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && { echo "PROD_OK $(date +%T)"; return 0; }; sleep 5; done
  echo "PROD_RESTORE_FAILED"
}
trap 'if [[ ${DONE:-0} != 1 ]]; then kill $SAMPLER 2>/dev/null || true; restore_prod; fi' EXIT
systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service 2>/dev/null || true
sleep 4; stop_server
bash vram-sampler.sh "v-$P" > /dev/null 2>&1 &
SAMPLER=$!
setsid "$LS" -m "$MODEL" --host 127.0.0.1 --port 18001 --alias gemma-4 --api-key-file "$KEYFILE" \
  -ngl all --n-cpu-moe "$NCPU" -np 2 --kv-unified --kv-unified-per-slot "$CTX" \
  -fa on -ctk "$K" -ctv "$V" -b 128 -ub 1024 --cont-batching --load-mode auto \
  --mmproj "$MMPROJ" --no-mmproj-offload > "$SLOG" 2>&1 &
SRV=$!
BOOT=UNKNOWN
for i in $(seq 1 150); do
  kill -0 $SRV 2>/dev/null || { BOOT=CRASH; break; }
  curl -s -m 2 http://127.0.0.1:18001/health 2>/dev/null | grep -q ok && { BOOT=PASS; break; }
  sleep 2
done
grep -qiE "out of memory|cudaMalloc failed" "$SLOG" && BOOT=OOM
printf "BOOT\t%s\t\n" "$BOOT" | tee -a "$RLOG"
[[ "$BOOT" != PASS ]] && { DONE=1; restore_prod; exit 2; }
grep -a "sizing KV pool" "$SLOG" | tee -a "$RLOG"
for t in 170000 175000 178000 179000; do
  python3 exact.py dual "$t" >> "$RLOG" 2>&1 || break
done
python3 exact.py retrieval 175000 3 >> "$RLOG" 2>&1
python3 exact.py bench >> "$RLOG" 2>&1
kill $SAMPLER 2>/dev/null
DONE=1
restore_prod
grep -aE "DUAL|RET|LONG" "$RLOG"
echo LADDER_DONE
