#!/usr/bin/env bash
# One-shot profile test: stop prod, start test server, full verification, restore prod.
# Usage: test-one.sh <NCPU> [CTX] [CTK] [CTV]  (env: TEST_ONLY=1 stops after boot+smoke)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"
NCPU="$1"; CTX="${2:-196608}"; CTK="${3:-q8_0}"; CTV="${4:-q8_0}"
LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"
PROFILE="p-${CTX}-${CTK}-${NCPU}"
SLOG="$LOGDIR/server-$PROFILE.log"
RESULTS="$LOGDIR/results-$PROFILE.tsv"
: > "$RESULTS"
KEYFILE="$(mktemp /tmp/gemma4-key.XXXXXX)"; chmod 600 "$KEYFILE"
printf '%s\n' "$LLAMA_API_KEY" > "$KEYFILE"
restore() {
  if [[ -f "$LOGDIR/test.pid" ]]; then kill "$(cat "$LOGDIR/test.pid")" 2>/dev/null || true; rm -f "$LOGDIR/test.pid"; sleep 4; fi
  rm -f "$KEYFILE"
  sudo -n systemctl restart gemma4.service 2>/dev/null || true
  sleep 2
  for i in $(seq 1 240); do
    if curl -s -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1; then echo "PROD_OK $(date +%T)"; break; fi
    sleep 5
  done
}
trap 'status=$?; echo "RESTORING (exit $status)"; restore; echo "PROFILE_DONE $PROFILE status=$status"' EXIT
sudo -n systemctl stop gemma4.service
sleep 6
for i in $(seq 1 30); do USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1); (( USED < 800 )) && break; sleep 3; done
echo "gpu free after stop: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)  (profile: $PROFILE)"
cd "$SCRIPT_DIR"
setsid /home/rugent/llama.cpp/build/bin/llama-server \
  -m /home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 --port 18001 --alias gemma-4 \
  --api-key-file "$KEYFILE" \
  -ngl all --n-cpu-moe "$NCPU" -c "$CTX" -np 2 --kv-unified-per-slot "$CTX" \
  -fa on -ctk "$CTK" -ctv "$CTV" -b 128 -ub 1024 --cont-batching --kv-unified --load-mode auto \
  --mmproj /home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf --no-mmproj-offload \
  > "$SLOG" 2>&1 &
echo $! > "$LOGDIR/test.pid"
nohup "$SCRIPT_DIR/vram-sampler.sh" "v-$PROFILE" > /dev/null 2>&1 &
SAMPLER=$!
BOOT=TIMEOUT
for i in $(seq 1 72); do
  if curl -s -m 2 http://127.0.0.1:18001/health >/dev/null 2>&1; then BOOT=PASS; break; fi
  if grep -q "cudaMalloc failed: out of memory\|failed to allocate CUDA\|exiting due to model loading error" "$SLOG" 2>/dev/null; then
    grep -q "out of memory" "$SLOG" && BOOT=OOM || BOOT=CRASH; break
  fi
  sleep 5
done
echo "BOOT=$BOOT" | tee -a "$RESULTS"
if [[ "$BOOT" != "PASS" ]]; then
  grep -m1 "allocating.*MiB on device\|cudaMalloc failed" "$SLOG" >> "$RESULTS" || true
  kill $SAMPLER 2>/dev/null || true
  exit 3
fi
export LLAMA_TEST_KEY="$LLAMA_API_KEY"
run_test() {
  local name="$1"; shift
  if timeout 1800 python3 "$SCRIPT_DIR/test-client.py" "$@" > "$LOGDIR/t-$PROFILE-$name.log" 2>&1; then
    echo "$name PASS $(tr '\n' ' ' < "$LOGDIR/t-$PROFILE-$name.log" | tail -c 250)" >> "$RESULTS"
  else
    echo "$name FAIL $(tr '\n' ' ' < "$LOGDIR/t-$PROFILE-$name.log" | tail -c 250)" >> "$RESULTS"
  fi
}
run_test smoke smoke
sleep 3
run_test vision vision
sleep 2
if [[ "${TEST_ONLY:-0}" == "1" ]]; then kill $SAMPLER 2>/dev/null || true; echo TEST_ONLY_DONE; exit 0; fi
run_test long64k long 65536
sleep 1
run_test long128k long 131072
sleep 1
run_test long160k long 163840
sleep 1
run_test long180k long 180224
sleep 1
run_test dual128k dual 131072
sleep 1
run_test dual160k dual 163840
sleep 1
run_test dual180k dual 180224
kill $SAMPLER 2>/dev/null || true
wait $SAMPLER 2>/dev/null || true
kill "$(cat "$LOGDIR/test.pid")" 2>/dev/null || true
echo "done tests"