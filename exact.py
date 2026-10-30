#!/usr/bin/env python3
"""Exact-token long/dual/retrieval driver for llama-server (port 18001).

Truth source for prompt size: templated prompt measured via server-side
/apply-template + /tokenize. Never sizes prompts by characters or file bytes.
Payloads sent to /v1/chat/completions with enable_thinking=false.
"""
import json, os, sys, time, base64, threading, urllib.request, urllib.error, argparse

KEY = None
def get_key():
    global KEY
    if KEY: return KEY
    k = os.environ.get("LLAMA_TEST_KEY")
    if k:
        KEY = k; return KEY
    for line in open("/home/rugent/gemma4-server/.env"):
        if line.startswith("LLAMA_API_KEY="):
            KEY = line.split("=",1)[1].strip(); return KEY
    raise SystemExit("no api key")

URL = "http://127.0.0.1:18001"
def req(path, payload, timeout=7200, quiet_err=True):
    r = urllib.request.Request(URL+path, data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+get_key()}, method="POST")
    try:
        return json.load(urllib.request.urlopen(r, timeout=timeout)), None
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"

# deterministic filler; never contains code-like tokens
FILLER_UNIT = "Tento odsek je vyplnovaci text bez vyznamu sluzaci na tlak na kontekst. "
def filler(units):
    return FILLER_UNIT * units

def templated_prompt(messages):
    a, err = req("/apply-template", {"messages": messages, "add_generation_prompt": True}, timeout=120)
    if a is None: raise RuntimeError(f"apply-template: {err}")
    return a["prompt"]

def count_tokens(prompt):
    t, err = req("/tokenize", {"content": prompt, "add_special": True, "parse_special": False}, timeout=120)
    if t is None: raise RuntimeError(f"tokenize: {err}")
    n = t["tokens"]
    return n if isinstance(n, int) else len(n)

def build_calibration(target, extra_messages):
    """Return units s.t. templated token count ~target (tol 100), via server /tokenize."""
    lo, hi = 0, 40000
    # probe per-unit cost
    probe = filler(1) + " X"
    msgs = list(extra_messages)
    units = 1
    def prompt_for(u):
        m = list(extra_messages); m[-1]["content"] = filler(u) + m[-1]["content"][m[-1]["content"].index("\n"):] if False else m[-1]["content"]
        return None
    # simpler: caller supplies base message template with marker {FILLER}
    return lo, hi

def count_for_units(units, messages_tmpl):
    """messages_tmpl: list of dicts; last user content is marker-replaced by filler*units."""
    m = json.loads(json.dumps(messages_tmpl))
    m[-1]["content"] = m[-1]["content"].replace("{FILLER}", filler(units), 1)
    p = templated_prompt(m)
    return count_tokens(p), m, p

def calibrate(target, messages_tmpl, tol=100):
    n0, _, _ = count_for_units(1, messages_tmpl)
    n1, _, _ = count_for_units(100, messages_tmpl)
    cpl = max((n1-n0)/99.0, 1)  # tokens per filler unit
    units = int((target-n0)/cpl)
    units = max(1, units)
    n, m, p = count_for_units(units, messages_tmpl)
    tries = 0
    while abs(n-target) > tol and tries < 25:
        units = max(0, units + int((target-n)/cpl))
        n, m, p = count_for_units(units, messages_tmpl)
        tries += 1
    return units, n

def chat(messages, max_tokens=32, temperature=0.0):
    return req("/v1/chat/completions",
               {"model":"gemma-4","messages":messages,"max_tokens":max_tokens,
                "temperature":temperature,"enable_thinking":False}, timeout=7200)

FACTS = [
    (2,  "ALPHA_CODE = KESTREL-49127"),
    (25, "BETA_CODE = MARBLE-82731"),
    (50, "GAMMA_CODE = ORBIT-35219"),
    (75, "DELTA_CODE = CEDAR-61984"),
    (98, "OMEGA_CODE = LANTERN-74326"),
]
QUESTIONS = "V docre som vložil 5 kódov. Na zaklade dokumentu odpovedz presne a LEN hodnotami:\nALPHA_CODE = ?\nBETA_CODE = ?\nGAMMA_CODE = ?\nDELTA_CODE = ?\nOMEGA_CODE = ?"
def retrieval_messages():
    # fillers distributed so codes land at ~2/25/50/75/98%
    parts = []
    prev = 0
    for pct, fact in FACTS:
        w = pct - prev
        parts.append("{FILLER}" + " " + fact)
        prev = pct
    parts.append("{FILLER}")
    body = "\n\n".join(parts)
    return [{"role":"user","content":body + "\n\n" + QUESTIONS}]

def cmd_long(target):
    msgs = [{"role":"user","content":"{FILLER}\n\nTento dokument je velmi dlhy. Zhrn ho v 30 slovach."}]
    units, actual = calibrate(target, msgs)
    m = json.loads(json.dumps(msgs)); m[-1]["content"] = m[-1]["content"].replace("{FILLER}", filler(units),1)
    t0=time.time()
    r, err = chat(m, 32)
    dt=time.time()-t0
    if r is None:
        print(f"LONG target={target} ERR {err}"); sys.exit(1)
    u=r["usage"]; ti=r.get("timings",{})
    print(f"LONG actual_input={u['prompt_tokens']} max_output=32 ok {dt:.1f}s pp={ti.get('prompt_per_second',0):.1f} tg={ti.get('predicted_per_second',0):.1f}")

def cmd_dual(target):
    # two payloads, equal exact tokens, launched simultaneously
    def mk(slot):
        msgs = [{"role":"user","content":"{FILLER}\n\nTento dokument pre "+slot+" je dlhy. Zhrn ho v 30 slovach."}]
        units, actual = calibrate(target, msgs)
        m = json.loads(json.dumps(msgs)); m[-1]["content"] = m[-1]["content"].replace("{FILLER}", filler(units),1)
        return m
    m0 = mk("A"); m1 = mk("B")
    res=[None,None]; err=[None,None]; tim=[None,None]
    def one(i):
        t0=time.time(); r,e = chat([m0,m1][i], 32)
        tim[i]=time.time()-t0
        if r: res[i]=r["usage"]["prompt_tokens"]
        else: err[i]=e
    ths=[threading.Thread(target=one,args=(i,)) for i in range(2)]
    t0=time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.time()-t0
    agg = ((res[0]+res[1])/wall) if res[0] and res[1] else 0
    print(f"DUAL target={target} actual=[{res[0]},{res[1]}] wall={wall:.1f}s agg_pp={agg:.0f} tok/s errors={err}")
    if any(err): sys.exit(1)

def cmd_vision():
    img = "/tmp/test_image.png"
    if not os.path.exists(img): img="/tmp/vision_test.png"
    b64 = base64.b64encode(open(img,"rb").read()).decode()
    t0=time.time()
    r,e = chat([{"role":"user","content":[
        {"type":"text","text":"Co je na obrazku? Kratke."},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}], 128)
    if r is None: print(f"VISION ERR {e}"); sys.exit(1)
    txt=(r["choices"][0]["message"].get("content") or "").strip()
    print(f"VISION ok {time.time()-t0:.1f}s text={txt[:90]!r}")

def cmd_bench():
    t0=time.time()
    r,e = chat([{"role":"user","content":"Napocit 300 viet o prirode, kazdu novym riadkom."}], 300, 0.8)
    ti=r.get("timings",{})
    print(f"BENCH short_tg={ti.get('predicted_per_second',0):.1f} tok/s dt={time.time()-t0:.1f}s")

def cmd_retrieval(target, runs=3):
    m = retrieval_messages()
    units, actual = calibrate(target, m)
    payload = json.loads(json.dumps(m)); payload[-1]["content"] = payload[-1]["content"].replace("{FILLER}", filler(units),1)
    print(f"RET target={target} actual_prompt_tokens~{actual}")
    for run in range(runs):
        t0=time.time()
        r,e = chat(payload, 512)
        if r is None:
            # capture diagnostics, do not blindly retry
            print(f"RET run{run+1} ERR {e}")
            continue
        msg = r["choices"][0]["message"]
        ans = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
        pt = r["usage"]["prompt_tokens"]
        scores=[]
        for key,val in [("ALPHA","KESTREL-49127"),("BETA","MARBLE-82731"),("GAMMA","ORBIT-35219"),("DELTA","CEDAR-61984"),("OMEGA","LANTERN-74326")]:
            ok = (val.lower() in ans.lower()) or (key.lower()+" " in ans.lower() and val.split("-")[1] in ans)
            scores.append(f"{key}:{'OK' if ok else 'BAD'}")
        n_ok = sum(1 for s in scores if s.endswith("OK"))
        print(f"RET run{run+1} pt={pt} {' '.join(scores)} score={n_ok}/5 {time.time()-t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["long","dual","vision","bench","retrieval","count"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "long": cmd_long(int(a.args[0]))
    elif a.cmd == "dual": cmd_dual(int(a.args[0]))
    elif a.cmd == "vision": cmd_vision()
    elif a.cmd == "bench": cmd_bench()
    elif a.cmd == "retrieval":
        tgt = int(a.args[0]) if a.args else 194000
        runs = int(a.args[1]) if len(a.args) > 1 else 3
        cmd_retrieval(tgt, runs)
    elif a.cmd == "count":
        msgs = [{"role":"user","content":"{FILLER}"}]
        units, actual = calibrate(int(a.args[0]), msgs)
        print(f"COUNT target={a.args[0]} actual={actual} units={units}")
