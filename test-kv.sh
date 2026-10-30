#!/usr/bin/env bash
# One-shot asymmetric-KV profile test (port 18001). No -c anywhere;
# --kv-unified + --kv-unified-per-slot sizes the shared pool to n_parallel*N.
# Usage: test-kv.sh <ctv> [ncpu]  e.g. ./test-kv.sh q5_1 3
set -uo pipefail
cd "$(dirname "$BASH_SOURCE[0]}")"
MODEL="/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
MMPROJ="/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf"
LS=/home/rugent/llama.cpp/build/bin/llama-server
CTV="${1:-q5_1}"
NCPU="${2:-3}"
PER_SLOT="${PER_SLOT:-196608}"
P="kv-q8-${CTV}-${NCPU}-${PER_SLOT}"
SLOG="logs/server-$P.log"
RLOG="logs/results-$P.tsv"
KEYFILE=/tmp/gemma4-test-key
: > "$RLOG"

[[ -f "$KEYFILE" && $(wc -c < "$KEYFILE") -ge 16 ]] || grep '^LLAMA_API_KEY' .env | cut -d= -f2 > "$KEYFILE"
chmod 600 "$KEYFILE"

stop_server() {
  local pids
  pids=$(pgrep -f "port 18001" || true)
  [[ -n "$pids" ]] && kill $pids 2>/dev/null || true
  sleep 5
  for i in $(seq 1 60); do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    [[ "$f" -lt 100 ]] && break; sleep 2
  done
  echo "gpu free after stop: ${f:-?} MiB"
}

restore_prod() {
  echo "RESTORING (exit $?)..."
  stop_server
  systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service 2>/dev/null || true
  sudo -n systemctl restart gemma4.service 2>/dev/null || systemctl restart gemma4.service 2>/dev/null || true
  sleep 2
  for i in $(seq 1 240); do
    if curl -s -m 3 "http://127.0.0.1:8001/health" >/dev/null 2>&1; then
      echo "PROD_OK $(date +%T)"; return 0
    fi
    sleep 5
  done
  echo "PROD_RESTORE_FAILED"
}

echo "POOL: n_parallel*per_slot=2*${PER_SLOT}=$((2*PER_SLOT))"
echo "stopping production server..."
if systemctl is-active --quiet gemma4.service; then
  systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service
  sleep 4
fi
stop_server

bash vram-sampler.sh "v-$P" &
VS=$!
setsid "$LS" -m "$MODEL" \
  --host 127.0.0.1 --port 18001 --alias gemma-4 --api-key-file "$KEYFILE" \
  -ngl all --n-cpu-moe "$NCPU" \
  -np 2 --kv-unified --kv-unified-per-slot "$PER_SLOT" \
  -fa on -ctk q8_0 -ctv "$CTV" \
  -b 128 -ub 1024 --cont-batching --load-mode auto \
  --mmproj "$MMPROJ" --no-mmproj-offload \
  > "$SLOG" 2>&1 &
SRV=$!
sleep 15
BOOT=UNKNOWN
for i in $(seq 1 150); do
  if ! kill -0 "$SRV" 2>/dev/null; then BOOT=CRASH; break; fi
  curl -s -m 3 http://127.0.0.1:18001/health | grep -q "ok" && { BOOT=PASS; break; }
  sleep 2
done
grep -qi "out of memory\|cudaMalloc\|failed to allocate" "$SLOG" && BOOT=OOM
echo "gpu free after boot: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1) MiB"
echo "BOOT=$BOOT"
if [[ "$BOOT" != PASS ]]; then kill $VS 2>/dev/null || true; restore_prod; exit 2; fi
printf "BOOT\tPASS\t\n" >> "$RLOG"

# KV pool / flash attention verification from log
echo "---- KV/FA startup lines ----"
grep -aiE "kv|cache|unified|flash|attn|slot|n_ctx|cells|cuda" "$SLOG" | grep -aivE "slot (get|print|launch|release)|task" | head -15

# smoke + vision
python3 test-client.py smoke >> "$RLOG" 2>&1
python3 test-client.py vision /tmp/test_image.png "Co je na obrazku? Kratce." >> "$RLOG" 2>&1

# single-slot long context
for t in 131072 163840 180224 190464 194560; do
  python3 test-client.py long "$t" 32 >> "$RLOG" 2>&1
done

# dual-slot tests
for t in 131072 153600 163840 174080 184320 194560; do
  python3 test-client.py dual "$t" 32 >> "$RLOG" 2>&1
done

kill $VS 2>/dev/null || true
stop_server
kill "$SRV" 2>/dev/null || true
kill $VS 2>/dev/null || true
echo "done tests"
restore_prod
echo "PROFILE_DONE $P"
grep -a "FAIL\|OOM" "$RLOG" | head -8
grep -aiE "out of memory|failed to find free space|exceeds|assert" "$SLOG" | tail -4
