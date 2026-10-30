#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE="gemma4.service"
PORT="${LLAMA_PORT:-8001}"
GPU_UUID="GPU-349f1acc-02f8-51c4-f834-b47c94e2983b"
if [[ -f "$SCRIPT_DIR/.env" ]]; then set -a; source "$SCRIPT_DIR/.env"; set +a; fi
: "${LLAMA_API_KEY:?LLAMA_API_KEY missing from .env or environment}"
pass=0; fail=0
check() { local name="$1"; shift; if "$@" >/dev/null 2>&1; then printf '%s: PASS\n' "$name"; ((pass+=1)); else printf '%s: FAIL\n' "$name"; ((fail+=1)); fi; }
check SERVICE systemctl is-active --quiet "$SERVICE"
check PORT bash -c "ss -ltn | grep -q ':${PORT} '"
check HEALTH curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/health"
check API curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/v1/models" -H "Authorization: Bearer ${LLAMA_API_KEY}"
check GPU bash -c "nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep -F '${GPU_UUID}' | grep -F 'llama-server'"
printf 'OVERALL: %s\n' "$([[ $fail -eq 0 ]] && echo PASS || echo FAIL)"
(( fail == 0 ))
