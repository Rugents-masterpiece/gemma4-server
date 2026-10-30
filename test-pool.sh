#!/usr/bin/env bash
# True-pool profile test: stop prod, start test server (port 18001) with given params,
# run boot/smoke/vision/long/dual/retrieval, monitor VRAM, always restore production.
# Usage: test-pool.sh CTX K V NCPU [B] [UB] [MODE]
#   MODE: full=tests+dual+retrieval (default), boot=boot+smoke+vision only
set -uo pipefail
cd "$(dirname "$BASH_SOURCE[0]}")"
CTX="$1"; K="$2"; V="$3"; NCPU="$4"; B="${5:-128}"; UB="${6:-1024}"; MODE="${7:-full}"
MODEL="/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
MMPROJ="/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf"
LS=/home/rugent/llama.cpp/build/bin/llama-server
P="pool-${CTX}-${K}-${V}-m${NCPU}-b${B}ub${UB}"
SLOG="logs/server-$P.log"; RLOG="logs/results-$P.tsv"
KEYFILE=/tmp/gemma4-test-key
: > "$RLOG"
[[ -f "$KEYFILE" && $(wc -c < "$KEYFILE") -ge 16 ]] || grep '^LLAMA_API_KEY' .env | cut -d= -f2 > "$KEYFILE"
chmod 600 "$KEYFILE"

stop_server() {
  local pids; pids=$(pgrep -f "port 18001" || true)
  [[ -n "$pids" ]] && kill $pids 2>/dev/null || true
  sleep 5
  for i in $(seq 1 60); do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    [[ "${f:-0}" -lt 100 ]] && break; sleep 2
  done
  echo "gpu free after stop: ${f:-?} MiB"
}
restore_prod() {
  if [[ ${RESTORED:-0} == 1 ]]; then return 0; fi
  RESTORED=1
  echo "RESTORING (exit $? or signal)..."
  stop_server
  systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service 2>/dev/null || true
  sudo -n systemctl restart gemma4.service 2>/dev/null || systemctl restart gemma4.service 2>/dev/null || true
  sleep 3
  for i in $(seq 1 240); do
    curl -s -m 3 "http://127.0.0.1:8001/health" >/dev/null 2>&1 && { echo "PROD_OK $(date +%T)"; return 0; }
    sleep 5
  done
  echo "PROD_RESTORE_FAILED"
}
trap 'if [[ ${DONE:-0} != 1 ]]; then kill $SAMPLER 2>/dev/null || true; stop_server; restore_prod; fi' EXIT

echo "stopping production..."
if systemctl is-active --quiet gemma4.service; then
  systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service; sleep 4
fi
stop_server
bash vram-sampler.sh "v-$P" > /dev/null 2>&1 &
SAMPLER=$!
setsid "$LS" -m "$MODEL" \
  --host 127.0.0.1 --port 18001 --alias gemma-4 --api-key-file "$KEYFILE" \
  -ngl all --n-cpu-moe "$NCPU" \
  -np 2 --kv-unified --kv-unified-per-slot "$CTX" \
  -fa on -ctk "$K" -ctv "$V" \
  -b "$B" -ub "$UB" --cont-batching --load-mode auto \
  --mmproj "$MMPROJ" --no-mmproj-offload \
  > "$SLOG" 2>&1 &
SRV=$!
sleep 15
BOOT=UNKNOWN
for i in $(seq 1 150); do
  if ! kill -0 "$SRV" 2>/dev/null; then BOOT=CRASH; break; fi
  curl -s -m 2 http://127.0.0.1:18001/health 2>/dev/null | grep -q "ok" && { BOOT=PASS; break; }
  sleep 2
done
grep -qiE "out of memory|cudaMalloc failed|failed to allocate" "$SLOG" && BOOT=OOM
printf "BOOT\t%s\t\n" "$BOOT" >> "$RLOG"
echo "BOOT=$BOOT"
if [[ "$BOOT" != PASS ]]; then
  grep -aoE "allocating [0-9.]+ MiB[^:]*: cudaMalloc failed: out of memory" "$SLOG" | tail -1 >> "$RLOG"
  kill $SAMPLER 2>/dev/null || true
  DONE=1
  restore_prod
  exit 2
fi
# verify true pool line
grep -a "sizing KV pool" "$SLOG" | head -1 | tee -a "$RLOG"
# FA runtime check
grep -aiE "flash" "$SLOG" | grep -aivE "off|auto" | head -2 | tee -a "$RLOG" || true
python3 test-client.py smoke >> "$RLOG" 2>&1
python3 test-client.py vision /tmp/test_image.png "Co je na obrazku? Kratce." >> "$RLOG" 2>&1
python3 test-client.py bench 300 >> "$RLOG" 2>&1
if [[ "$MODE" == full ]]; then
  python3 test-client.py long "$CTX" 64 >> "$RLOG" 2>&1
  for t in 131072 163840 184320 190464 194560; do python3 test-client.py dual "$t" 32 >> "$RLOG" 2>&1; done
  python3 retrieval_test.py >> "$RLOG" 2>&1
fi
kill $SAMPLER 2>/dev/null || true
stop_server
kill "$SRV" 2>/dev/null || true
DONE=1
echo "done tests"
restore_prod
echo "PROFILE_DONE $P"
grep -aE "FAIL|OOM" "$RLOG" | head -6
grep -aiE "out of memory|failed to find free space" "$SLOG" | tail -2
