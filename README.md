# Gemma 4 26B-A4B QAT Server

**llama.cpp server configurations, benchmarks, and production scripts** for running **Gemma 4 26B-A4B QAT** (Quantization-Aware Training) models on consumer GPUs (RTX 4060 Ti, RTX 3050).

## Features

- **Optimized Quantization**: Q8_K / Q5_K_M KV cache, UD-Q4_K_XL weights
- **MoE Support**: `-np 2` experts, `--n-cpu-moe 6` CPU offload
- **Long Context**: Up to 131K context with MTP (Multi-Token Prediction)
- **Vision Support**: mmproj F16 projector for multimodal
- **Flash Attention**: `-fa on` for memory efficiency
- **Continuous Batching**: `--cont-batching` for throughput
- **Production Scripts**: systemd, healthcheck, monitoring

## Model Variants Tested

| Config | KV Cache | Weights | Context | Notes |
|--------|----------|---------|---------|-------|
| Production | Q8_0 / Q5_1 | UD-Q4_K_XL | 128K | Stable, low VRAM |
| 131K Candidate | Q8_0 / Q5_1 | UD-Q4_K_XL | 131K | MTP enabled |
| Vision | Q8_0 / Q5_1 | UD-Q4_K_XL | 128K | mmproj F16 |

## Prerequisites

- llama.cpp (built with CUDA, `GGML_CUDA=1`)
- RTX 4060 Ti (16GB) or RTX 3050 (8GB)
- CUDA 12.x
- Model: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`
- mmproj: `mmproj-F16.gguf` (for vision)

## Installation

```bash
git clone https://github.com/Rugents-masterpiece/gemma4-server.git
cd gemma4-server

# Build llama.cpp (if not installed)
cd /path/to/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Copy model files
cp /path/to/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf .
cp /path/to/mmproj-F16.gguf .
```

## Quick Start

```bash
# Safe serve (conservative VRAM)
./serve-safe.sh

# Production (131K context, MTP)
./run-131k-candidate.sh

# Vision enabled
./serve-vision-safe.sh

# Health check
./healthcheck.sh
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `serve.sh` | Main production server |
| `serve-safe.sh` | Conservative VRAM (8GB) |
| `serve-mtp.sh` | MTP enabled |
| `serve-vision-safe.sh` | Vision multimodal |
| `healthcheck.sh` | Liveness probe |
| `status.sh` | Server status |
| `monitor-memory.sh` | VRAM/RAM monitoring |
| `benchmark.sh` | Quick benchmark |
| `benchmark-production.sh` | Full benchmark suite |
| `stability-test.sh` | Long-running stability |
| `tune.sh` | Parameter tuning |

## Systemd Service

```ini
# /etc/systemd/system/gemma4.service
[Unit]
Description=Gemma 4 26B-A4B llama.cpp Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rugent
WorkingDirectory=/home/rugent/gemma4-server
Environment=CUDA_VISIBLE_DEVICES=GPU-<uuid>
ExecStart=/home/rugent/llama.cpp/build/bin/llama-server \
  -m /home/rugent/models/gemma-4-26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8001 \
  --alias gemma-4 --api-key-file /tmp/gemma4-api-key \
  -ngl all --n-cpu-moe 6 -np 2 \
  --kv-unified --kv-unified-per-slot 131072 \
  -fa on -ctk q8_0 -ctv q5_1 \
  -b 128 -ub 1024 --cont-batching \
  --mmproj /home/rugent/models/gemma-4-26B-A4B-QAT/mmproj-F16.gguf \
  --no-mmproj-offload
Restart=on-failure
RestartSec=10
TimeoutStartSec=1800

[Install]
WantedBy=multi-user.target
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-ngl all` | All layers on GPU | |
| `--n-cpu-moe 6` | 6 CPU threads for MoE | |
| `-np 2` | 2 experts parallel | MoE parallelism |
| `--kv-unified` | Unified KV cache | Memory efficient |
| `--kv-unified-per-slot 131072` | 128K per slot | Long context |
| `-fa on` | Flash Attention | Speed + memory |
| `-ctk q8_0 -ctv q5_1` | KV quantization | Q8 keys, Q5 values |
| `-b 128 -ub 1024` | Batch sizes | Throughput tuning |
| `--cont-batching` | Continuous batching | Higher throughput |

## Benchmarks

```bash
# Quick test
./benchmark.sh

# Production benchmark
./benchmark-production.sh

# KV cache analysis
python3 kv-calc.py
python3 kv-meta.py
```

## Monitoring

```bash
# VRAM/RAM
./monitor-memory.sh

# Server status
./status.sh

# Logs
journalctl -u gemma4 -f
```

## Security

- No credentials in repository
- API key via `/tmp/gemma4-api-key` (file, not env)
- Runs on specific GPU via `CUDA_VISIBLE_DEVICES`
- Localhost only by default

## License

MIT
