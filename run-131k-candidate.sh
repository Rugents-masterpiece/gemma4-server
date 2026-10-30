#!/usr/bin/env bash
set -u
set -o pipefail
cd /home/rugent/gemma4-server
CTX=${1:?ctx}; K=${2:?k}; V=${3:?v}; NCPU=${4:?ncpu}; MODE=${5:-boot}; B=${6:-128}; UB=${7:-1024}
LABEL="candidate-${CTX}-${K}-${V}-m${NCPU}-b${B}ub${UB}-fitoff"
SLOG="logs/server-${LABEL}.log"
RLOG="logs/results-${LABEL}.tsv"
KEYFILE=/tmp/gemma4-test-key
MODEL=/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
MMPROJ=/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf
GPU=GPU-349f1acc-02f8-51c4-f834-b47c94e2983b
: > "$RLOG"
[[ -s "$KEYFILE" ]] || { grep '^LLAMA_API_KEY=' .env | cut -d= -f2- > "$KEYFILE"; chmod 600 "$KEYFILE"; }
SRV= SAMPLER= RESTORED=0
bash vram-sampler.sh "v-${LABEL}" > /dev/null 2>&1 &
SAMPLER=$!
stop_test() { [[ -n "$SRV" ]] && kill "$SRV" 2>/dev/null || true; pkill -f -- '--port 18001' 2>/dev/null || true; sleep 5; }
restore() {
  [[ "$RESTORED" == 1 ]] && return
  RESTORED=1; [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null || true
  stop_test
  sudo -n systemctl restart gemma4.service >/dev/null 2>&1 || systemctl restart gemma4.service >/dev/null 2>&1 || true
  for i in $(seq 1 180); do curl -sf --max-time 3 http://127.0.0.1:8001/health >/dev/null && { echo PROD_RESTORE_PASS; return; }; sleep 2; done
  echo PROD_RESTORE_FAIL
}
trap restore EXIT INT TERM
sudo -n systemctl stop gemma4.service >/dev/null 2>&1 || systemctl stop gemma4.service >/dev/null 2>&1 || true
sleep 5; stop_test
CUDA_VISIBLE_DEVICES="$GPU" setsid /home/rugent/llama.cpp/build/bin/llama-server -m "$MODEL" \
 --host 127.0.0.1 --port 18001 --alias gemma-4 --api-key-file "$KEYFILE" -ngl all \
 --n-cpu-moe "$NCPU" -np 2 --kv-unified --kv-unified-per-slot "$CTX" \
 -fa on -ctk "$K" -ctv "$V" -b "$B" -ub "$UB" --cont-batching --load-mode auto --fit off \
 --mmproj "$MMPROJ" --no-mmproj-offload > "$SLOG" 2>&1 &
SRV=$!
for i in $(seq 1 180); do
  kill -0 "$SRV" 2>/dev/null || break
  curl -sf --max-time 2 http://127.0.0.1:18001/health >/dev/null && break
  sleep 2
done
if ! curl -sf --max-time 3 http://127.0.0.1:18001/health >/dev/null; then
  echo -e "BOOT\tFAIL" | tee -a "$RLOG"; grep -aiE 'out of memory|cudaMalloc failed|sizing KV pool' "$SLOG" | tail -8; exit 2
fi
grep -a 'sizing KV pool' "$SLOG" | head -1 | tee -a "$RLOG"
printf 'BOOT\tPASS\n' | tee -a "$RLOG"
python3 exact.py bench | tee -a "$RLOG"
python3 exact.py vision | tee -a "$RLOG"
if [[ "$MODE" == full ]]; then
  python3 exact.py long 125000 | tee -a "$RLOG"
  python3 exact.py dual 120000 | tee -a "$RLOG"
  python3 exact.py dual 125000 | tee -a "$RLOG"
  python3 exact.py retrieval 125000 3 | tee -a "$RLOG"
elif [[ "$MODE" == final ]]; then
  python3 exact.py long 128000 | tee -a "$RLOG"
  python3 exact.py dual 128000 | tee -a "$RLOG"
  python3 exact.py retrieval 125000 3 | tee -a "$RLOG"
fi
exit 0
