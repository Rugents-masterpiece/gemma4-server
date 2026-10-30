#!/usr/bin/env bash
# test-profile.sh <n_cpu_moe>
# Starts test server on port 18001, runs boot+smoke+vision, logs, stops.
# Exits 0 = stable (boot+smoke+vision PASS), 3 = boot OOM, 1=smoke FAIL, 2=vision FAIL
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCPU="$1"
LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"
PROFILE="moe${NCPU}"
# Cleanup any previous test server
PIDFILE="$LOGDIR/test-$PROFILE.pid"
[ -f "$PIDFILE" ] && { kill "$(cat "$PIDFILE")" 2>/dev/null || true; sleep 3; }
# Cleanup any test server on port 18001
ss -ltn 2>/dev/null | grep -q ':18001 ' && {
  PID18=$(ss -ltnp 2>/dev/null | awk '/:18001 /{print $NF}' | sed 's/.*pid=\([0-9]*\).*/\1/' | head -1)
  [ -n "$PID18" ] && { kill "$PID18" 2>/dev/null || true; sleep 3; }
}
# Start
cd "$SCRIPT_DIR"
TEST_NCPU="$NCPU" TEST_CTX="${TEST_CTX:-196608}" TEST_CTK="${TEST_CTK:-q8_0}" TEST_CTV="${TEST_CTV:-q8_0}"   TEST_MMPROJ="${TEST_MMPROJ:-/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf}"   TEST_MMPROJ_OFFLOAD="${TEST_MMPROJ_OFFLOAD:-0}"   ./test-serve.sh > "$LOGDIR/test-$PROFILE.out" 2>&1 &
sleep 5
PID=$(cat "$LOGDIR/test-$PROFILE.pid")
# Boot wait with VRAM sampler
LABEL="boot-$PROFILE-$(date +%Y%m%d-%H%M%S)"
nohup "$SCRIPT_DIR/vram-sampler.sh" "$LABEL" > /dev/null 2>&1 &
SAMPLER=$!
# Poll /health up to 90s
for i in $(seq 1 90); do
  if curl -s -m 3 http://127.0.0.1:18001/health > /dev/null 2>&1; then
    BOOT=PASS
    break
  fi
  sleep 1
done
kill "$SAMPLER" 2>/dev/null || true
# Wait for sampler to flush
wait "$SAMPLER" 2>/dev/null || true
if [ "$BOOT" != "PASS" ]; then
  # Check for OOM in log
  if grep -q "out of memory\|CUDA OOM\|failed to allocate" "$LOGDIR/test-$PROFILE.log" 2>/dev/null; then
    echo "PROFILE moe$NCPU boot=OOM"; exit 3
  fi
  echo "PROFILE moe$NCPU boot=TIMEOUT (90s)"; exit 4
fi
# Smoke (text)
if python3 "$SCRIPT_DIR/test-client.py" smoke > "$LOGDIR/test-$PROFILE.smoke.log" 2>&1; then
  SMOKE=PASS
else
  SMOKE=FAIL
fi
# Vision
if python3 "$SCRIPT_DIR/test-client.py" vision > "$LOGDIR/test-$PROFILE.vision.log" 2>&1; then
  VISION=PASS
else
  VISION=FAIL
fi
# Final kill
kill "$PID" 2>/dev/null || true
sleep 2
# Cleanup pidfile
rm -f "$PIDFILE"
echo "PROFILE moe$NCPU boot=$BOOT smoke=$SMOKE vision=$VISION"
# Exit codes: 0=PASS, 1=smoke FAIL, 2=vision FAIL, 3=OOM, 4=timeout
if [ "$SMOKE" == "PASS" ] && [ "$VISION" == "PASS" ]; then exit 0; fi
[ "$SMOKE" == "FAIL" ] && { echo "SMOKE failed"; exit 1; }
echo "VISION failed"; exit 2
