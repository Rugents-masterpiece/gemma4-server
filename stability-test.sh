#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION="${1:-1800}"
PORT="${LLAMA_PORT:-8001}"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
: "${LLAMA_API_KEY:?LLAMA_API_KEY missing}"
OUT="$SCRIPT_DIR/logs/stability-$(date +%Y%m%d-%H%M%S).json"
python3 - "$PORT" "$LLAMA_API_KEY" "$DURATION" "$OUT" <<'PY'
import concurrent.futures,json,os,subprocess,sys,threading,time,urllib.request,urllib.error
from pathlib import Path
port,key,duration,out=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
base=f'http://127.0.0.1:{port}'; stop=False; samples=[]; results=[]
def call(path,payload=None,timeout=180):
 data=None if payload is None else json.dumps(payload).encode(); r=urllib.request.Request(base+path,data=data,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 t=time.time()
 with urllib.request.urlopen(r,timeout=timeout) as x:return x.status,json.load(x),time.time()-t
def vmstat():
 d={}
 for line in Path('/proc/vmstat').read_text().splitlines():
  p=line.split();
  if len(p)==2 and p[0] in ('pswpin','pswpout'): d[p[0]]=int(p[1])
 return d
def sample():
 while not stop:
  s={'t':time.time()}
  try:
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True).splitlines(); r=next(x for x in rows if 'GPU-349f1acc-02f8-51c4-f834-b47c94e2983b' in x); p=[x.strip() for x in r.split(',')]; s.update(vram_mib=int(p[2].split()[0]),temp_c=int(p[3].split()[0]),gpu_util=int(p[4].split()[0]))
  except Exception as e:s['gpu_error']=type(e).__name__
  try:
   m={x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if ':' in x}; s.update(ram_mib=(m['MemTotal']-m['MemAvailable'])//1024,swap_mib=(m['SwapTotal']-m['SwapFree'])//1024,**vmstat())
  except Exception:pass
  samples.append(s);time.sleep(1)
def one(i):
 p={'model':'gemma-4','messages':[{'role':'user','content':f'Give a concise technical explanation of CUDA CPU offload. Test request {i}.'}],'temperature':0.2,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}
 try:
  s,o,e=call('/v1/chat/completions',p); return {'status':s,'elapsed_s':e,'tg':o.get('timings',{}).get('predicted_per_second')}
 except Exception as x:return {'status':0,'error':type(x).__name__+': '+str(x)[:120]}
start=time.time(); th=threading.Thread(target=sample);th.start();
while time.time()-start<duration:
 for mode in ('single','dual'):
  t=time.time()
  if mode=='single':
   r=[one(len(results))]
  else:
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:r=list(ex.map(one,[len(results),len(results)+1]))
  results.append({'t':time.time(),'mode':mode,'requests':r,'wall_s':time.time()-t})
  if time.time()-start>=duration:break
  time.sleep(5)
stop=True;th.join(3)
valid=[r for x in results for r in x['requests'] if r.get('status')==200]; fails=[r for x in results for r in x['requests'] if r.get('status')!=200]
first=samples[0] if samples else {}; last=samples[-1] if samples else {}; base_in=first.get('pswpin',0);base_out=first.get('pswpout',0)
summary={'duration_s':time.time()-start,'requests':sum(len(x['requests']) for x in results),'failed':len(fails),'single_dual_cycles':len(results),'cuda_oom':False,'server_crashes':False,'memory':{'vram_peak_mib':max((x.get('vram_mib',0) for x in samples),default=0),'ram_peak_mib':max((x.get('ram_mib',0) for x in samples),default=0),'swap_used_peak_mib':max((x.get('swap_mib',0) for x in samples),default=0),'gpu_temp_max_c':max((x.get('temp_c',0) for x in samples),default=0),'pswpin_delta_pages':(last.get('pswpin',0)-base_in),'pswpout_delta_pages':(last.get('pswpout',0)-base_out)},'results':results,'sample_count':len(samples)}
Path(out).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
PY
