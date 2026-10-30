#!/usr/bin/env python3
"""Profile test client for test-pool.sh (port 18001)."""
import json, os, sys, time, base64, urllib.request
PORT = 18001
def api_key():
    k = os.environ.get("LLAMA_TEST_KEY")
    if k: return k
    p = "/home/rugent/gemma4-server/.env"
    for line in open(p):
        if line.startswith("LLAMA_API_KEY="): return line.split("=",1)[1].strip()
    raise SystemExit("no api key")
KEY = api_key()
URL = f"http://127.0.0.1:{PORT}"
def req(path, payload, timeout=3600):
    r = urllib.request.Request(URL+path, data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())
def msg_text(m):
    c = m.get("content") or ""
    if not c and m.get("reasoning_content"):
        rc = m["reasoning_content"]
        c = rc.split("\n\n")[-1] if "\n\n" in rc else rc
    return c.strip()
def cmd_smoke():
    t0=time.time()
    r = req("/v1/chat/completions", {"model":"gemma-4",
        "messages":[{"role":"user","content":"Povzved jedine cislo: 2+2=?"}],
        "max_tokens":16,"temperature":0.0}, timeout=180)
    print(f"SMOKE ok {time.time()-t0:.1f}s text={msg_text(r['choices'][0]['message'])[:80]!r}")
def cmd_vision(imgpath="/tmp/test_image.png", question="Co je na obrazku? Kratce."):
    if not os.path.exists(imgpath): imgpath="/tmp/vision_test.png"
    b64 = base64.b64encode(open(imgpath,"rb").read()).decode()
    t0=time.time()
    r = req("/v1/chat/completions", {"model":"gemma-4","messages":[{"role":"user","content":[
        {"type":"text","text":question},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}],
        "max_tokens":128}, timeout=600)
    txt = msg_text(r["choices"][0]["message"])
    print(f"VISION ok {time.time()-t0:.1f}s text={txt[:120]!r}")
def mk_long(n):
    filler = "testovaci text pre overenie kontextu. "
    return (filler * (max(1, n)))[:n*3]
def cmd_long(n, mt=32):
    prompt = "Tento text je dlhy. Zhrn ho jednou vetou.\n\n" + mk_long(int(n))
    t0=time.time()
    r = req("/v1/chat/completions", {"model":"gemma-4","messages":[{"role":"user","content":prompt}],
        "max_tokens":int(mt),"temperature":0.0}, timeout=7200)
    u=r["usage"]; dt=time.time()-t0
    ptx=u["prompt_tokens"]; ppg=r.get("timings",{}).get("prompt_per_second",0)
    tgs=r.get("timings",{}).get("predicted_per_second",0)
    print(f"LONG {n}t ok {dt:.1f}s prompt_tokens={ptx} pp_tok_s={ppg:.1f} tg_tok_s={tgs:.1f} text={msg_text(r['choices'][0]['message'])[:60]!r}")
def cmd_dual(n, mt=32):
    import threading
    base = mk_long(int(n))
    prompts = ["Slot A.\n\n"+base, "Slot B.\n\n"+base]
    res=[None,None]; err=[None,None]; tim=[None,None]
    def one(i):
        t0=time.time()
        try:
            r = req("/v1/chat/completions", {"model":"gemma-4","messages":[{"role":"user","content":prompts[i]}],
                "max_tokens":int(mt),"temperature":0.0}, timeout=7200)
            res[i]=r["usage"]["prompt_tokens"]; tim[i]=time.time()-t0
        except Exception as e:
            err[i]=f"{type(e).__name__}: {str(e)[:160]}"
    ths=[]; t0=time.time()
    for i in range(2):
        t=threading.Thread(target=one,args=(i,)); t.start(); ths.append(t)
    for t in ths: t.join()
    print(f"DUAL {n}t {time.time()-t0:.1f}s prompts={res} errors={err}")
    if any(err) or None in res: sys.exit(1)
def cmd_bench(mt=300):
    t0=time.time()
    r = req("/v1/chat/completions", {"model":"gemma-4","messages":[{"role":"user","content":"Napocit 300 viet o prirode, kazdu novym riadkom."}],
        "max_tokens":int(mt),"temperature":0.8,"seed":42}, timeout=3600)
    u=r["usage"]; dt=time.time()-t0
    tgs=r.get("timings",{}).get("predicted_per_second",0)
    print(f"BENCH completion_tokens={u['completion_tokens']} dt={dt:.1f}s tg_tok_s={tgs:.1f}")
if __name__=="__main__":
    if len(sys.argv)<2: print(__doc__); sys.exit(2)
    c=sys.argv[1]; a=sys.argv[2:]
    {"smoke":cmd_smoke,"vision":lambda:cmd_vision(*a),"long":lambda:cmd_long(*a),
     "dual":lambda:cmd_dual(*a),"bench":cmd_bench}.get(c,lambda:None)()
