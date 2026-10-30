#!/usr/bin/env python3
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"

def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    if n > 100_000_000: raise ValueError(f"insane string length {n}")
    return f.read(n).decode("utf-8", "replace")

def read_val(f, t):
    if t == 0: return f.read(1)[0]
    if t == 1: return struct.unpack("<b", f.read(1))[0]
    if t == 2: return struct.unpack("<H", f.read(2))[0]
    if t == 3: return struct.unpack("<h", f.read(2))[0]
    if t == 4: return struct.unpack("<I", f.read(4))[0]
    if t == 5: return struct.unpack("<i", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return f.read(1)[0] != 0
    if t == 8: return read_string(f)
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    if t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        al = struct.unpack("<Q", f.read(8))[0]
        if at not in (4,5,6,8,10,11): raise ValueError(f"array elem type {at}")
        return [read_val(f, at) for _ in range(al)]
    raise ValueError(f"type {t}")

kv = {}
with open(path, "rb") as f:
    magic = f.read(4)
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0] if ver < 2 else 0
    n_kv = struct.unpack("<Q", f.read(8))[0]
    for _ in range(n_kv):
        k = read_string(f)
        t = struct.unpack("<I", f.read(4))[0]
        kv[k] = read_val(f, t)

for key in ["llama.n_layer","llama.embedding_length","llama.attention.head_count","llama.attention.head_count_kv","llama.attention.layer_types","llama.rope.dimension_count","llama.rope.dimension_sections","tokenizer.ggml.bos_token_id"]:
    v = kv.get(key)
    if key == "llama.attention.layer_types" and isinstance(v, list) and v and isinstance(v[0], (list,int)):
        from collections import Counter
        v = dict(Counter(v if v and not isinstance(v[0], list) else [str(x) for x in v]))
    print(key, "=", v)

# KV bytes per token calculation
n_layer = kv.get("llama.n_layer", 0)
hd = kv.get("llama.embedding_length",0) // max(1,kv.get("llama.attention.head_count",1))
kvh = kv.get("llama.attention.head_count_kv", [hd])
if isinstance(kvh, list):
    kvh = max(kvh)
lt = kv.get("llama.attention.layer_types", [])
full_layers = sum(1 for x in lt if x == 1) if lt and lt[0] in (0,1) else sum(1 for x in lt if x=="full") if lt else n_layer
swa_layers = n_layer - full_layers
print(f"\nfull-attn layers={full_layers} swa layers={swa_layers} head_dim={hd} head_kv={kvh}")
# q8_0 K = head_dim*1B, q5_1 V = 1.25*head_dim? no — q5_1 block per head_dim/32: 1B/32 + 0.5? approximate
# llama.cpp cell bytes: K size = n_cell * n_layer_full * (head_kv*head_dim) * bps
# q8_0: 1 byte / elem; q5_1: (0.5 + 1/64? ) — llama.cpp q5_1 = 1/32 scale + 5/8 = 0.66 bytes/elem approx. Use actual ggml block sizes:
# q8_0 block: 32 elems, 32*0.5 + ... actually block q8_0: 32 elems * 1B = 32B + 0? scale f16: 34/32=1.06
# q5_1 block: 32 elems *5/8 +16 bit scale = 20+2=22/32=0.688
# q4_1: 32*0.5 + 2 = 16+2 = 18/32 = 0.5625
for name, ksize, vsize in [("q8_0",1.0625,1.0625), ("q5_1",1.0625,0.6875), ("q4_1",1.0625,0.5625)]:
    K = full_layers*kvh*hd*ksize
    V = full_layers*kvh*hd*vsize
    print(f"per-token (full layers, K={name} V={name}): K={K:.0f}B V={V:.0f}B total={K+V:.0f}B")
