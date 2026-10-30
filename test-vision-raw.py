#!/usr/bin/env python3
"""Direct vision request vs test server on 18001. Checks that content is not empty."""
import json, sys, base64, urllib.request
KEY = open("/tmp/gemma4-test-key").read().strip()
img = base64.b64encode(open("/tmp/vision_test.png", "rb").read()).decode()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
for i in range(n):
    req = urllib.request.Request("http://127.0.0.1:18001/v1/chat/completions",
        data=json.dumps({"model": "gemma-4", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Co je na obrazku? Strucne."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}],
            "max_tokens": 256}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        txt = r["choices"][0]["message"]["content"].strip()
        print(f"[{i+1}] len={len(txt)} tokens={r['usage']['completion_tokens']} text={txt[:120]!r}")
    except Exception as e:
        print(f"[{i+1}] ERR {type(e).__name__} {str(e)[:120]}")