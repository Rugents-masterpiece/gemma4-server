#!/usr/bin/env bash
# exact-token true-dual profile runner — uses exact.py for real token counts.
# Usage: test-exact.sh CTX K V NCPU MODE
#   MODE: boot | full | boots (5x restart stability)
set -uo pipefail
cd "$(dirname "$BASH_SOURCE[0]}")"
CTX="${1:-196608}"; K="${2:-q8_0}"; V="${3:-q8_0}"; NCPU="${4:-3}"; MODE="${5:-full}"
FIT="${6:-on}"
MODEL="/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
MMPROJ="/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf"
LS=/home/rugent/llama.cpp/build/bin/llama-server
P="exact-${CTX}-${K}-${V}-m${NCPU}-fit${FIT}-${MODE}"
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
    [[ "${f:-0}" -gt 15000 ]] && break; sleep 2
  done
}
start_server() {
  setsid "$LS" -m "$MODEL" --host 127.0.0.1 --port 18001 --alias gemma-4 --api-key-file "$KEYFILE" \
    -ngl all --n-cpu-moe "$NCPU" -np 2 --kv-unified --kv-unified-per-slot "$CTX" \
    -fa on -ctk "$K" -ctv "$V" -b 128 -ub 1024 --cont-batching --load-mode auto --fit "$FIT" \
    --mmproj "$MMPROJ" --no-mmproj-offload > "$SLOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do
    if ! kill -0 "$SRV" 2>/dev/null; then echo CRASH; return 1; fi
    curl -s -m 2 http://127.0.0.1:18001/health 2>/dev/null | grep -q "ok" && { echo PASS; return 0; }
    sleep 2
  done
  echo TIMEOUT
}
restore_prod() {
  if [[ ${RESTORED:-0} == 1 ]]; then return 0; fi
  RESTORED=1
  stop_server
  systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service 2>/dev/null || true
  sudo -n systemctl restart gemma4.service 2>/dev/null || systemctl restart gemma4.service 2>/dev/null || true
  sleep 5
  for i in $(seq 1 240); do
    curl -s -m 3 "http://127.0.0.1:8001/health" >/dev/null 2>&1 && { echo "PROD_OK $(date +%T)"; return 0; }
    sleep 5
  done
  echo "PROD_RESTORE_FAILED"
}
trap 'if [[ ${DONE:-0} != 1 ]]; then kill $SAMPLER 2>/dev/null || true; stop_server; restore_prod; fi' EXIT

echo "== profile $P fit=$FIT"
if systemctl is-active --quiet gemma4.service; then
  systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service 2>/dev/null || true
  sleep 4
fi
stop_server
if [[ "$MODE" == "boots" ]]; then
  pass=0; fail=0
  for n in 1 2 3 4 5; do
    st=$(start_server)
    if [[ "$st" == PASS ]]; then
      if python3 exact.py bench >> "$RLOG" 2>&1; then pass=$((pass+1)); echo "boot $n: PASS (bench ok)";
      else fail=$((fail+1)); echo "boot $n: PASS boot / FAIL bench"; fi
    else fail=$((fail+1)); echo "boot $n: $st"; fi
    stop_server
  done
  printf "BOOTS\t%s/5\t\n" "$pass" >> "$RLOG"
  DONE=1; restore_prod
  echo "BOOTS_RESULT $pass/5"
  exit 0
fi

bash vram-sampler.sh "v-$P" > /dev/null 2>&1 &
SAMPLER=$!
st=$(start_server)
printf "BOOT\t%s\t\n" "$st" >> "$RLOG"
echo "BOOT=$st"
if [[ "$st" != PASS ]]; then
  grep -aoE "allocating [0-9.]+ MiB[^:]*: out of memory" "$SLOG" | tail -1 >> "$RLOG"
  grep -aE "out of memory" "$SLOG" | tail -1 >> "$RLOG"
  DONE=1; restore_prod; exit 2
fi
grep -a "sizing KV pool" "$SLOG" | head -1 | tee -a "$RLOG"
python3 exact.py bench >> "$RLOG" 2>&1
python3 exact.py vision >> "$RLOG" 2>&1

if [[ "$MODE" == "full" ]]; then
  python3 exact.py long 194000 >> "$RLOG" 2>&1
  python3 exact.py dual 190000 >> "$RLOG" 2>&1
  python3 exact.py dual 194000 >> "$RLOG" 2>&1
  python3 exact.py retrieval 194000 3 >> "$RLOG" 2>&1
fi
kill $SAMPLER 2>/dev/null || true
DONE=1
echo "done tests"
restore_prod
echo "PROFILE_DONE $P"
grep -aE "LONG|DUAL|RET|VISION|BENCH" "$RLOG" | tail -14
grep -aiE "out of memory|failed to find free space" "$SLOG" | tail -2