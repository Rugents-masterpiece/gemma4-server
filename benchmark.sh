#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
LLAMA_BENCH="${LLAMA_BENCH:-/home/rugent/llama.cpp/build/bin/llama-bench}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
[[ -x "$LLAMA_BENCH" ]] || { echo "ERROR: llama-bench not found: $LLAMA_BENCH" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-349f1acc-02f8-51c4-f834-b47c94e2983b}"
mkdir -p "$SCRIPT_DIR/logs"
LOG="$SCRIPT_DIR/logs/benchmark-$(date +%Y%m%d-%H%M%S).log"
{ echo '=== BEFORE ==='; nvidia-smi; free -h; echo '=== LLAMA BENCH ==='; "$LLAMA_BENCH" -m "$MODEL" -ngl 99 -ncmoe 99 -fa on -p 128 -n 128 -b 128 -ub 64; echo '=== AFTER ==='; nvidia-smi; free -h; } 2>&1 | tee "$LOG"
