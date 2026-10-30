#!/usr/bin/env python3
"""Debug retrieval: build 190k prompt, ask ONE question, dump raw answer to /tmp/ret_answer.txt."""
import json, sys, time
sys.path.insert(0, "/home/rugent/gemma4-server")
from exact import filler, calibrate, chat, templated_prompt, count_for_units, FILLER_UNIT
import urllib.request, os

KEYFILE = "/tmp/gemma4-test-key"
def get_key():
    k = os.environ.get("LLAMA_TEST_KEY")
    if k: return k
    return open(KEYFILE).read().strip()

# reuse from exact but inline to control raw output
target = int(sys.argv[1]) if len(sys.argv)>1 else 190000
q = sys.argv[2] if len(sys.argv)>2 else "ALPHA_CODE"
FACTS = [(2,"ALPHA_CODE = KESTREL-49127"),(25,"BETA_CODE = MARBLE-82731"),(50,"GAMMA_CODE = ORBIT-35219"),(75,"DELTA_CODE = CEDAR-61984"),(98,"OMEGA_CODE = LANTERN-74326")]

def build_msg():
    parts=[]
    prev=0
    for pct,f in FACTS:
        parts.append("{FILLER} "+f); prev=pct
    parts.append("{FILLER}")
    body="\n\n".join(parts)
    return [{"role":"user","content": body + f"\n\nV docre su 5 5-znackovych kody. Odpovedz LEN hodnotou: {q} = ?"}]

msgs=build_msg()
import exact
units, actual = exact.calibrate(target, msgs)
m = json.loads(json.dumps(msgs)); m[-1]["content"] = m[-1]["content"].replace("{FILLER}", filler(units),1)
t0=time.time()
r, err = exact.chat(m, 128)
print(f"target={target} actual_tokens~{actual} units={units}")
if r:
    msg=r["choices"][0]["message"]
    print("keys:", list(msg.keys()))
    print("content:", repr(msg.get("content")))
    if msg.get("reasoning_content"): print("reasoning[:300]:", repr(msg["reasoning_content"][:300]))
    print("usage:", r["usage"])
else:
    print("ERR:", err)
