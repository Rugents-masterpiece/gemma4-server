#!/usr/bin/env python3
"""KV pool sizing calc. Reads model config for n_layer/head counts."""
import json, subprocess, sys, re

# Get model metadata via llama-cli --list or gguf dump using llama-quantize info
# Simplest: use llama-cli from build dir with --help? No. Use the server log KV info lines.
# But we can compute from GGUF metadata:
path = "/home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
import struct

def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def read_gguf_kv(path):
    kv = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic not in (b"GGUF", b"ggUF", b"ggUf"):
            raise ValueError("not gguf")
        ver = struct.unpack("<I", f.read(4))[0]
        if ver < 2:
            n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            k = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            # types: 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,8 string,9 array,10 u64,11 i64,12 f64
            if vtype == 0: v = f.read(1)[0]
            elif vtype == 1: v = struct.unpack("<b", f.read(1))[0]
            elif vtype == 2: v = struct.unpack("<H", f.read(2))[0]
            elif vtype == 3: v = struct.unpack("<h", f.read(2))[0]
            elif vtype == 4: v = struct.unpack("<I", f.read(4))[0]
            elif vtype == 5: v = struct.unpack("<i", f.read(4))[0]
            elif vtype == 6: v = struct.unpack("<f", f.read(4))[0]
            elif vtype == 7: v = f.read(1)[0] != 0
            elif vtype == 8: v = read_string(f)
            elif vtype == 10: v = struct.unpack("<Q", f.read(8))[0]
            elif vtype == 11: v = struct.unpack("<q", f.read(8))[0]
            elif vtype == 12: v = struct.unpack("<d", f.read(8))[0]
            elif vtype == 9:
                arr_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                # read array (simple ints/strings for now)
                vals = []
                for _ in range(arr_len):
                    if arr_type == 8: vals.append(read_string(f))
                    elif arr_type == 4: vals.append(struct.unpack("<I", f.read(4))[0])
                    elif arr_type == 5: vals.append(struct.unpack("<i", f.read(4))[0])
                    elif arr_type == 6: vals.append(struct.unpack("<f", f.read(4))[0])
                    else: raise ValueError("arr type not impl " + str(arr_type))
                v = vals
            else: raise ValueError("type " + str(vtype))
            kv[k] = v
    return kv

kvs = read_gguf_kv(path)
n_layer = kvs.get("llama.n_layer")
n_head = kvs.get("llama.attention.head_count")
head_kv = kvs.get("llama.attention.head_count_kv")
embd = kvs.get("llama.embedding_length")
swa = kvs.get("llama.attention.layer_types", [])
swa_n = sum(1 for t in swa if t == "swa" or t == 1) if swa else 0
print(f"layers={n_layer} heads={n_head} head_kv={head_kv} embd={embd} layer_types={list(set(swa))} swa_count={swa_n}")
# For full-attn layers: cells = n_slots? Actually unified pool cells shared across all layers? No.
# KV cache cells = tokens_capacity * n_kv_layers. Bytes = cells*layer*per-cell-k*bytes(k) + *v.
# per layer per token: head_kv * head_dim for K (q8: head_dim*1B? q8_0 block quant). head_dim = embd/head_count.
head_dim = embd // n_head
print(f"head_dim={head_dim}")
# unified pool: n_cells total across full layers + swa separate? Check later from logs.
