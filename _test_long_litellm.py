import json, urllib.request, sys
key=''
for line in open('/home/rugent/litellm/.env'):
    if line.startswith('LITELLM_MASTER_KEY'): key=line.strip().split('=',1)[1]
n=int(sys.argv[1]) if len(sys.argv)>1 else 10500
unit="The quick brown fox jumps over the lazy dog. "
text = unit*n
body={"model":"gemma-4","max_tokens":16,"messages":[{"role":"user","content":"FILLER_START " + text + " FILLER_END What is the capital of Slovakia? Answer with one short sentence."}]}
req=urllib.request.Request('http://127.0.0.1:4000/v1/chat/completions', data=json.dumps(body).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
import time; t0=time.time()
try:
    resp=urllib.request.urlopen(req, timeout=1000)
    d=json.loads(resp.read())
    print(f"PASS prompt={d['usage']['prompt_tokens']} completion={d['usage']['completion_tokens']} elapsed={time.time()-t0:.1f}s content={d['choices'][0]['message']['content'][:80]}")
except Exception as e:
    print("FAIL", e, getattr(e,'read',lambda:b'')()[:400], f"elapsed={time.time()-t0:.1f}s")
