#!/usr/bin/env python3
"""Long-context retrieval: 5 facts inserted at 0/25/50/75/100% of ~190k-token
filler, then ONE request asks all 5 questions (keeps KV pool under dual ceiling)."""
import os, json, urllib.request

BASE = os.environ.get("RET_BASE", "http://127.0.0.1:18001")
KEY = open("/tmp/gemma4-test-key").read().strip()
FILL = "Tento odstavec je vyplnovy text, ktory nema vyznam. " * 48  # ~512 chars
FACTS = [
    ("fakt1", "Modre zvierata maju meno ZORIN a ziju v doline KOPA"),
    ("fakt2", "Cervena rastlina sa vola BLUMAR a kvitne kazdy styriasty den"),
    ("fakt3", "Cislo tajneho kodu je 4719 a heslo je VASKY-ROCK"),
    ("fakt4", "Most v hlavnom meste ma 2244 mostnikov a stavia ho ALFRED"),
    ("fakt5", "Posledny oddiel knihy bol nazov TICY a cas bol 23:59"),
]
QS = [
    ("Ako sa volaju modre zvierata a v ktorej doline ziju?", ["ZORIN", "KOPA"]),
    ("Ako sa vola cervena rastlina a kedy kvitne?", ["BLUMAR", "styriasty"]),
    ("Ake je cislo tajneho kodu a ake je heslo?", ["4719", "VASKY-ROCK"]),
    ("Kolko mostnikov ma most a kto ho stava?", ["2244", "ALFRED"]),
    ("Aky bol nazov posledneho oddielu knihy a aky bol cas?", ["TICY", "23:59"]),
]

def post(msgs, max_tokens=256):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps({"model": "gemma-4", "messages": msgs, "max_tokens": max_tokens,
                         "temperature": 0.0}).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=1800))

# gemma tokenizer for this filler ~1.92 chars/token
TARGET = int(os.environ.get("RET_TOKENS", "190000"))
total_chars = int(TARGET * 1.92)
per = (total_chars - 64) // (len(FACTS) + 1)
parts = []
for i, (_, fact) in enumerate(FACTS):
    reps = max(1, per // 264)
    parts.append("".join((FILL + " ") * reps) + fact + " ")
context = ("Tieto udaje su dolezite pre test. " + "".join(parts) + " Koniec dokumentu.")
qtext = "\n\n".join(f"OTAZKA {i+1}: {q}" for i, (q, _) in enumerate(QS))
system = ("Pamataj presne hodnoty z dokumentu. Pre kazdu OTAZKU odpovedz na novy riadok"
          " cislu otazky a len spravnou hodnotou. Dokument je velmi dlhy.")
try:
    r = post([{"role": "system", "content": system},
              {"role": "user", "content": context + "\n\n" + qtext}], max_tokens=256)
    c = r["choices"][0]["message"].get("content") or ""
    pt = r.get("usage", {}).get("prompt_tokens", "?")
    print(f"RET	ctx_tokens={pt}")
    for i, (q, expects) in enumerate(QS):
        line = ""
        for ln in c.splitlines():
            if f"OTAZKA {i+1}" in ln:
                line = ln
        ok = all(str(e).lower() in line.lower() for e in expects)
        print(f"RET	Q{i+1}	{'OK' if ok else 'FAIL'}	{line[:150]}")
except Exception as e:
    print(f"RET	ERROR	{e}")
