#!/usr/bin/env bash
# debug runner: start server, keep running for interactive debugging, manual stop later
set -uo pipefail
cd "$(dirname "$BASH_SOURCE[0]}")"
CTX=196608; K=q5_1; V=q5_1; NCPU=7
MODEL="/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
MMPROJ="/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf"
LS=/home/rugent/llama.cpp/build/bin/llama-server
KEYFILE=/tmp/gemma4-test-key
[[ -f "$KEYFILE" && $(wc -c < "$KEYFILE") -ge 16 ]] || grep '^LLAMA_API_KEY' .env | cut -d= -f2 > "$KEYFILE"
chmod 600 "$KEYFILE"
systemctl stop gemma4.service 2>/dev/null || sudo -n systemctl stop gemma4.service 2>/dev/null || true
pkill -f "port 18001" 2>/dev/null || true
sleep 3
setsid "$LS" -m "$MODEL" --host 127.0.0.1 --port 18001 --alias gemma-4 --api-key-file "$KEYFILE" \
  -ngl all --n-cpu-moe "$NCPU" -np 2 --kv-unified --kv-unified-per-slot "$CTX" \
  -fa on -ctk "$K" -ctv "$V" -b 128 -ub 1024 --cont-batching --load-mode auto \
  --mmproj "$MMPROJ" --no-mmproj-offload > logs/debug-server.log 2>&1 &
SRV=$!
echo "debug server pid $SRV"
for i in $(seq 1 120); do
  curl -s -m 2 http://127.0.0.1:18001/health 2>/dev/null | grep -q "ok" && { echo READY; exit 0; }
  sleep 2
done
echo NOTREADY