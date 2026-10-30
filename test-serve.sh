#!/usr/bin/env bash
# Test-profile launcher (NOT production). Production entry is serve.sh.
# Starts llama-server on port 18001 so the production server (8001) is untouched.
# Env overrides (all optional):
#   TEST_CTX=196608 TEST_PAR=2 TEST_CTK=q8_0 TEST_CTV=q8_0
#   TEST_NCPU=3 TEST_MMPROJ=<path> TEST_MMPROJ_OFFLOAD=0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"  # LLAMA_API_KEY
LLAMA_SERVER="${LLAMA_SERVER:-/home/rugent/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
PORT=18001
CTX="${TEST_CTX:-196608}"
PAR="${TEST_PAR:-2}"
CTK="${TEST_CTK:-q8_0}"
CTV="${TEST_CTV:-q8_0}"
NCPU="${TEST_NCPU:-3}"
MMPROJ="${TEST_MMPROJ:-/home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf}"
OFFLOAD="${TEST_MMPROJ_OFFLOAD:-0}"
[[ -x "$LLAMA_SERVER" ]] || { echo "no server binary" >&2; exit 1; }
[[ -f "$MODEL" && -f "$MMPROJ" ]] || { echo "missing model/mmproj" >&2; exit 1; }
if ss -ltn | grep -q ":${PORT} "; then echo "port ${PORT} busy — kill test server first" >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES="GPU-349f1acc-02f8-51c4-f834-b47c94e2983b"
KEYFILE="$(mktemp /tmp/gemma4-test-key.XXXXXX)"; chmod 600 "$KEYFILE"
printf '%s\n' "$LLAMA_API_KEY" > "$KEYFILE"
MM=(--mmproj "$MMPROJ")
[[ "$OFFLOAD" == "1" ]] || MM+=(--no-mmproj-offload)
LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"
PROFILE="ctx${CTX}_par${PAR}_${CTK}_${CTV}_moe${NCPU}_mm$(basename "$MMPROJ" .gguf)_off${OFFLOAD}"
LOG="$LOGDIR/test-$PROFILE.log"
echo "starting profile: $PROFILE"
# shellcheck disable=SC2024
setsid "$LLAMA_SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --alias gemma-4 \
  --api-key-file "$KEYFILE" -ngl all --n-cpu-moe "$NCPU" \
  -c "$CTX" -np "$PAR" --kv-unified-per-slot "$CTX" \
  -fa on -ctk "$CTK" -ctv "$CTV" -b 128 -ub 64 --cont-batching --kv-unified \
  --load-mode auto "${MM[@]}" > "$LOG" 2>&1 &
echo $! > "$LOGDIR/test-$PROFILE.pid"
echo "pid=$(cat "$LOGDIR/test-$PROFILE.pid") log=$LOG"