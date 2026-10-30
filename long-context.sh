#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${LLAMA_SERVER:-/home/rugent/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
SLOT0="${1:-200000}"; SLOT1="${2:-200000}"; NAME="${3:-q4-ncpu3-long}"
PORT="${LONG_PORT:-8001}"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
LOG="$SCRIPT_DIR/logs/long-${NAME}.log"; JSON="$SCRIPT_DIR/logs/long-${NAME}.json"
python3 - "$SERVER" "$MODEL" "$LLAMA_API_KEY" "$PORT" "$SLOT0" "$SLOT1" "$LOG" "$JSON" <<'PY'
import concurrent.futures,json,os,subprocess,sys,tempfile,time,threading,urllib.request,urllib.error
from pathlib import Path
server,model,key,port,t0,t1,log_path,json_path=sys.argv[1:]; port=int(port); targets=[int(t0),int(t1)]
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=env.get('CUDA_VISIBLE_DEVICES','GPU-349f1acc-02f8-51c4-f834-b47c94e2983b')
keyfile=tempfile.NamedTemporaryFile(prefix='gemma4-long-key-',delete=False); keyfile.write((key+'\n').encode()); keyfile.close(); os.chmod(keyfile.name,0o600)
cmd=[server,'-m',model,'--host','127.0.0.1','--port',str(port),'--alias','gemma-4','--api-key-file',keyfile.name,'-ngl','all','--n-cpu-moe','3','-c','262144','-np','2','--kv-unified-per-slot','262144','-fa','on','-ctk','q4_1','-ctv','q4_1','-b','128','-ub','64','--cont-batching','--kv-unified','--load-mode','auto']
log=open(log_path,'w'); log.write('COMMAND '+json.dumps(cmd)+'\n'); log.flush(); p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
def req(path,payload=None,timeout=3600):
 data=None if payload is None else json.dumps(payload).encode(); r=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}); 
 with urllib.request.urlopen(r,timeout=timeout) as x: return x.status,json.loads(x.read().decode())
ready=False
for _ in range(360):
 if p.poll() is not None: break
 try:
  s,o=req('/health',None,10)
  if s==200 and o.get('status')=='ok': ready=True; break
 except Exception: pass
 time.sleep(1)
result={'name':Path(json_path).stem.replace('long-',''),'ready':ready,'targets':targets,'actual_prompt_tokens':None,'requests':None,'slots_samples':[]}
samples=[]; stop=False
def monitor():
 while not stop:
  row={}
  try:
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True,stderr=subprocess.DEVNULL).splitlines(); r=next((x for x in rows if 'RTX 4060 Ti' in x),None)
   if r:
    z=[x.strip() for x in r.split(',')]; row.update(vram_mib=int(z[1].split()[0]),gpu_util=int(z[3].split()[0]))
   mem=Path('/proc/meminfo').read_text().splitlines(); v={x.split(':')[0]:int(x.split()[1]) for x in mem if ':' in x}; row.update(ram_mib=(v['MemTotal']-v['MemAvailable'])//1024,swap_mib=(v['SwapTotal']-v['SwapFree'])//1024)
  except Exception: pass
  try:
   s,o=req('/slots',None,5); row['slots']=o
  except Exception: pass
  row['t']=time.time(); samples.append(row); time.sleep(1)
def make_text(target):
 phrase='The quick brown fox studies CUDA memory management and CPU expert offload. '
 # tokenize candidate text and adjust repeats by binary search
 def tok(n):
  s,o=req('/tokenize',{'content':phrase*n,'add_special':False},300); return len(o.get('tokens',[])), phrase*n
 lo,hi=1,max(2,target//5+100)
 while tok(hi)[0]<target: hi*=2
 while lo<hi:
  mid=(lo+hi)//2
  if tok(mid)[0]<target: lo=mid+1
  else: hi=mid
 n,text=tok(lo); return text,n
if ready:
 texts=[]; counts=[]
 for target in targets:
  text,n=make_text(target); texts.append(text); counts.append(n)
 result['actual_prompt_tokens']=counts
 th=threading.Thread(target=monitor); th.start();
 def one(i):
  t=time.time(); s,o=req('/completion',{'prompt':texts[i]+'\nReturn exactly LONG_OK.','n_predict':32,'temperature':0,'cache_prompt':False},3600); return {'slot':i,'status':s,'elapsed_s':time.time()-t,'timings':o.get('timings',{}),'content':o.get('content','')[-100:]}
 t=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: result['requests']=list(ex.map(one,[0,1]))
 result['wall_s']=time.time()-t
 stop=True; th.join(3)
else: stop=True
if not ready: stop=True
if p.poll() is None: p.terminate()
try: p.wait(20)
except subprocess.TimeoutExpired: p.kill(); p.wait()
try: os.unlink(keyfile.name)
except FileNotFoundError: pass
result['exit_code']=p.returncode
result['memory']={'idle_vram_mib':samples[0].get('vram_mib') if samples else None,'peak_vram_mib':max((x.get('vram_mib',0) for x in samples),default=None),'peak_ram_mib':max((x.get('ram_mib',0) for x in samples),default=None),'peak_swap_mib':max((x.get('swap_mib',0) for x in samples),default=None),'samples':len(samples)}
result['slot_observations']=[{'t':x.get('t'),'slots':[{'id':s.get('id'),'n_ctx':s.get('n_ctx'),'is_processing':s.get('is_processing'),'n_prompt_tokens':s.get('n_prompt_tokens'),'n_prompt_tokens_processed':s.get('n_prompt_tokens_processed')} for s in (x.get('slots') or [])]} for x in samples if 'slots' in x][::max(1,len([x for x in samples if 'slots' in x])//20)]
result['slots_sample_count']=sum(1 for x in samples if 'slots' in x)
Path(json_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
PY
