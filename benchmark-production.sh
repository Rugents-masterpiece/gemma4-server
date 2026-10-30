#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${LLAMA_SERVER:-/home/rugent/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
NAME="${1:?usage: $0 NAME NCPU CTK CTV BATCH UBATCH [LOAD_MODE]}"
NCPU="${2:?missing NCPU}"; CTK="${3:?missing CTK}"; CTV="${4:?missing CTV}"
BATCH="${5:?missing BATCH}"; UBATCH="${6:?missing UBATCH}"; LOAD_MODE="${7:-auto}"
PORT="${PROD_BENCH_PORT:-8001}"
[[ -f "$SCRIPT_DIR/.env" ]] && { set -a; source "$SCRIPT_DIR/.env"; set +a; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-349f1acc-02f8-51c4-f834-b47c94e2983b}"
LOG="$SCRIPT_DIR/logs/production-${NAME}.log"; JSON="$SCRIPT_DIR/logs/production-${NAME}.json"
[[ -x "$SERVER" && -f "$MODEL" ]] || { echo 'missing server or model' >&2; exit 1; }
if ss -ltn | grep -q ":${PORT} "; then echo "ERROR: port ${PORT} is occupied" >&2; exit 1; fi
python3 - "$SERVER" "$MODEL" "$LLAMA_API_KEY" "$PORT" "$NCPU" "$CTK" "$CTV" "$BATCH" "$UBATCH" "$LOAD_MODE" "$LOG" "$JSON" <<'PY'
import concurrent.futures,json,os,subprocess,sys,tempfile,threading,time,urllib.error,urllib.request
from pathlib import Path
server,model,key,port,ncpu,ctk,ctv,batch,ubatch,load_mode,log_path,json_path=sys.argv[1:]
port=int(port); ncpu=int(ncpu); batch=int(batch); ubatch=int(ubatch)
keyfile=tempfile.NamedTemporaryFile(prefix='gemma4-bench-key-',delete=False); keyfile.write((key+'\n').encode()); keyfile.close(); os.chmod(keyfile.name,0o600)
cmd=[server,'-m',model,'--host','127.0.0.1','--port',str(port),'--alias','gemma-4','--api-key-file',keyfile.name,'-ngl','all','--n-cpu-moe',str(ncpu),'-c','262144','-np','2','--kv-unified-per-slot','262144','-fa','on','-ctk',ctk,'-ctv',ctv,'-b',str(batch),'-ub',str(ubatch),'--cont-batching','--kv-unified','--load-mode',load_mode]
log=open(log_path,'w'); log.write('COMMAND '+json.dumps(cmd)+'\n'); log.flush()
p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
stop=False; samples=[]
def sample():
  while not stop:
    try:
      rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True,stderr=subprocess.DEVNULL).splitlines()
      row=next((r for r in rows if 'RTX 4060 Ti' in r),None)
      if row:
        parts=[x.strip() for x in row.split(',')]
        vram=int(parts[1].split()[0]); util=int(parts[3].split()[0])
        mem=Path('/proc/meminfo').read_text().splitlines(); vals={x.split(':')[0]:int(x.split()[1]) for x in mem if ':' in x}
        ram=(vals['MemTotal']-vals['MemAvailable'])//1024; swap=(vals['SwapTotal']-vals['SwapFree'])//1024
        samples.append({'t':time.time(),'vram_mib':vram,'gpu_util':util,'ram_mib':ram,'swap_mib':swap})
    except Exception: pass
    time.sleep(.25)
th=threading.Thread(target=sample); th.start()
def call(path,payload,timeout=300):
  data=None if payload is None else json.dumps(payload).encode()
  req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=data,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
  t=time.time()
  with urllib.request.urlopen(req,timeout=timeout) as r: body=r.read().decode(); status=r.status
  return status,time.time()-t,json.loads(body)
ready=False
for _ in range(240):
  if p.poll() is not None: break
  try:
    s,_,o=call('/health',None,10); ready=(s==200 and o.get('status')=='ok');
    if ready: break
  except Exception: pass
  time.sleep(1)
result={'name':Path(json_path).stem.replace('production-',''),'ready':ready,'profile':{'n_cpu_moe':ncpu,'kv_k':ctk,'kv_v':ctv,'batch':batch,'ubatch':ubatch,'load_mode':load_mode},'single':None,'dual':None}
if ready:
  payload={'model':'gemma-4','messages':[{'role':'user','content':'Explain CUDA CPU offload in llama.cpp in technical detail.'}],'temperature':0.2,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False}}
  s,elapsed,o=call('/v1/chat/completions',payload); result['single']={'status':s,'elapsed_s':elapsed,'timings':o.get('timings',{})}
  def one(i):
    q=dict(payload); q['messages']=[{'role':'user','content':f'Explain CUDA CPU offload in llama.cpp in technical detail. Request {i}.'}]
    s,e,o=call('/v1/chat/completions',q); return {'status':s,'elapsed_s':e,'timings':o.get('timings',{})}
  t0=time.time()
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: pair=list(ex.map(one,[0,1]))
  result['dual']={'wall_s':time.time()-t0,'requests':pair,'aggregate_tg_tps':sum(x.get('timings',{}).get('predicted_per_second',0) for x in pair)}
stop=True; th.join(timeout=2)
if p.poll() is None: p.terminate()
try: p.wait(15)
except subprocess.TimeoutExpired: p.kill(); p.wait()
try: os.unlink(keyfile.name)
except FileNotFoundError: pass
result['exit_code']=p.returncode
result['memory']={'idle_vram_mib':samples[0]['vram_mib'] if samples else None,'peak_vram_mib':max((x['vram_mib'] for x in samples),default=None),'peak_ram_mib':max((x['ram_mib'] for x in samples),default=None),'peak_swap_mib':max((x['swap_mib'] for x in samples),default=None),'sample_count':len(samples)}
Path(json_path).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
PY
