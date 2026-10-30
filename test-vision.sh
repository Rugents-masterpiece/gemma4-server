#!/usr/bin/env bash
# One-shot vision repro: stop prod, start test server, ONE vision request, capture, restore.
# Usage: test-vision.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$SCRIPT_DIR/logs"
PROFILE="vision-repro"
SLOG="$LOGDIR/server-$PROFILE.log"
restore() {
  PIDFILE="$LOGDIR/test.pid"
  if [[ -f "$PIDFILE" ]]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; sleep 4; fi
  sudo -n systemctl restart gemma4.service 2>/dev/null || true
  sleep 2
  for i in $(seq 1 240); do
    if curl -s -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1; then echo "PROD_OK $(date +%T)"; break; fi
    sleep 5
  done
}
trap 'restore' EXIT
sudo -n systemctl stop gemma4.service
sleep 6
for i in $(seq 1 30); do USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1); (( USED < 800 )) && break; sleep 3; done
cd "$SCRIPT_DIR"
setsid /home/rugent/llama.cpp/build/bin/llama-server \
  -m /home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 --port 18001 --alias gemma-4 \
  --api-key-file /tmp/gemma4-test-key \
  -ngl all --n-cpu-moe 3 -c 196608 -np 2 --kv-unified-per-slot 196608 \
  -fa on -ctk q8_0 -ctv q8_0 -b 128 -ub 1024 --cont-batching --kv-unified --load-mode auto \
  --mmproj /home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf --no-mmproj-offload \
  > "$SLOG" 2>&1 &
echo $! > "$LOGDIR/test.pid"
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:18001/health >/dev/null 2>&1 && break; sleep 5; done
python3 "$SCRIPT_DIR/test-client.py" vision > "$LOGDIR/vision-repro-client.log" 2>&1 && echo "VISION PASS" || echo "VISION FAIL"
tail -3 "$LOGDIR/vision-repro-client.log"