#!/usr/bin/env python3
"""Build an OpenAI chat payload whose *templated* input token count hits --target
(as measured by llama-server /tokenize). Deterministic filler (no code strings)."""
import json, urllib.request, urllib.error, os, sys, argparse

def req(base, path, payload, key):
    r = urllib.request.Request(base+path, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}","Content-Type":"application/json"}, method="POST")
    return json.load(urllib.request.urlopen(r, timeout=300))

FILLER = ("Tento odsek je vyplnovaci text bez vyznamu pre tlakovny test. " * 6)
def make(facts, filler_units):
    # facts: list of (pct, "CODE = LABEL-DIGITS"); filler_units = repeats
    text = ""
    prev = 0
    for pct, f in facts:
        n = max(1, int((pct - prev) * filler_units / 100))
        text += FILLER * n + " " + f + " "
        prev = pct
    text += FILLER * max(1, int((98 - prev) * filler_units / 100 + 1))
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18001")
    ap.add_argument("--keyfile", default="/tmp/gemma4-test-key")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--facts-pos", default="2,25,50,75,98")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = open(a.keyfile).read().strip()
    facts = [("ALPHA_CODE = KESTREL-49127", 2), ("BETA_CODE = MARBLE-82731", 25),
             ("GAMMA_CODE = ORBIT-35219", 50), ("DELTA_CODE = CEDAR-61984", 75),
             ("OMEGA_CODE = LANTERN-74326", 98)]
    qs = ("Na zaklade dokumentu odpovedz LEN hodnotami, nic ine. "
          "ALPHA_CODE? BETA_CODE? GAMMA_CODE? DELTA_CODE? OMEGA_CODE?")
    sysmsg = "Pamataj presne hodnoty. Pre kazdu premennu vrat len jej hodnotu."
    # binary search filler_units against server /tokenize of templated prompt
    def count(units):
        text = make([(f,p) for f,p in facts], units)
        msgs = [{"role":"system","content":sysmsg},
                {"role":"user","content":text+qs}]
        # measure templated tokens: build the chat prompt via apply-template then tokenize
        try:
            tmpl = req(a.base, "/apply-template", {"messages": msgs, "add_generation_prompt": True}, key)
            prompt = tmpl.get("prompt") or tmpl.get("content") or ""
            tk = req(a.base, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True}, key)
            return tk.get("tokens") if isinstance(tk, dict) else tk, prompt
        except Exception as e:
            # fallback: tokenize user text directly
            tk = req(a.base, "/tokenize", {"content": text+qs, "add_special": True}, key)
            return (tk.get("tokens") if isinstance(tk, dict) else tk), (text+qs)
    lo, hi = 1, 60000
    last = (None, None)
    for _ in range(40):
        mid = (lo+hi)//2
        n, prompt = count(mid)
        last = (n, mid)
        if n == a.target or abs(n - a.target) <= 100:
            break
        if n < a.target: lo = mid+1
        else: hi = mid-1
        if hi <= lo: break
    n, units = last
    text = make([(f,p) for f,p in facts], units)
    payload = {"model":"gemma-4",
               "messages":[{"role":"system","content":sysmsg},{"role":"user","content":text+qs}],
               "max_tokens":32,"temperature":0.0}
    json.dump(payload, open(a.out,"w"))
    print(f"requested_target={a.target}")
    print(f"actual_input_tokens={n}")
    print(f"filler_units={units}")
    print(f"max_output_tokens=32")
    print(f"total_theoretical_max={n+32}")

if __name__=="__main__":
    main()
