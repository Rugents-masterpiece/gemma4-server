#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LLAMA_PORT:-8001}"
if [[ -f "$SCRIPT_DIR/.env" ]]; then set -a; source "$SCRIPT_DIR/.env"; set +a; fi
: "${LLAMA_API_KEY:?LLAMA_API_KEY missing from .env or environment}"
python3 - "$PORT" "$LLAMA_API_KEY" <<'PY'
import json,sys,urllib.request
port,key=sys.argv[1:]
base=f'http://127.0.0.1:{port}'
def get(path):
 r=urllib.request.Request(base+path,headers={'Authorization':'Bearer '+key})
 with urllib.request.urlopen(r,timeout=15) as x: return x.status,json.load(x)
s,o=get('/health'); assert s==200 and o.get('status')=='ok', (s,o)
s,o=get('/v1/models'); assert s==200 and any(x.get('id')=='gemma-4' for x in o.get('data',[])), o
payload={'model':'gemma-4','messages':[{'role':'user','content':'Respond exactly with: GEMMA4_SMOKE_OK'}],'temperature':0,'max_tokens':8,'chat_template_kwargs':{'enable_thinking':False}}
r=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
with urllib.request.urlopen(r,timeout=120) as x: s=x.status; o=json.load(x)
content=o['choices'][0]['message'].get('content',''); assert s==200 and 'GEMMA4_SMOKE_OK' in content, content
print('HEALTH: PASS\nAPI: PASS\nCHAT: PASS\nOVERALL: PASS')
PY
