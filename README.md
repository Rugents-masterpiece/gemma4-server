# Gemma 4 26B-A4B IT QAT GGUF server

Deployment directory: `/home/rugent/gemma4-server`

## Verified hardware

- OS: Ubuntu 26.04.1 LTS, kernel 7.0.0-31-generic
- CPU: AMD Ryzen 7 3700X, 8 cores / 16 threads
- RAM: 45 GiB
- Swap: 8 GiB
- Target GPU: NVIDIA GeForce RTX 4060 Ti, 16380 MiB
- NVIDIA driver: 610.57.04
- CUDA UMD: 13.3; nvcc toolkit: 12.4.131
- A second RTX 3050 is present and occupied by another service. All Gemma launchers select the RTX 4060 Ti explicitly by UUID `GPU-349f1acc-02f8-51c4-f834-b47c94e2983b`; the 3050 is not used.

## Model and runtime

- Hugging Face repository: `unsloth/gemma-4-26B-A4B-it-qat-GGUF`
- GGUF: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`
- Size: 14,249,047,104 bytes (13.27 GiB)
- MTP draft downloaded: `mtp-gemma-4-26B-A4B-it.gguf` (251,939,328 bytes)
- llama.cpp upstream commit: `58367713a6935c0810103378144008df32e3d5db`
- Build: `0.4.1-dev`, CUDA enabled, `GGML_CUDA_FA_ALL_QUANTS=ON`, static build, Ninja

## Final production profile

- Server: `0.0.0.0:8001`
- API alias/model: `gemma-4`
- Parallel slots: 2
- Context per slot: 262144 tokens
- Context pool: 524288 tokens
- Log proof: `n_slots = 2, n_ctx_slot = 262144, kv_unified = 'true'`
- Flash Attention: on
- KV cache: K=`q4_1`, V=`q4_1`
- CPU MoE: first 3 of 30 layers (`--n-cpu-moe 3`)
- GPU MoE: remaining 27 layers
- Batch / microbatch: 128 / 64
- Load mode: `auto`
- No CUDA OOM in the final profile or the 200k + 200k long-context test

The earlier Q5_1/`n-cpu-moe=28` profile was functional but slow. In the second tuning pass, Q5_1 passed down to `n-cpu-moe=3`; `n-cpu-moe=2` failed compute-buffer allocation and `n-cpu-moe=3` had too little VRAM reserve. Q4_1/`n-cpu-moe=3` was the best measured stable production profile. Q4_1/`n-cpu-moe=2` booted but reached approximately 15.86 GiB VRAM and was rejected for insufficient reserve.

The model metadata reports `n_ctx_train = 262144`; the deployment keeps each slot at that native maximum. The requested total pool of 524288 is achieved by two slots, not by making one sequence 524288 tokens long.

## Scripts

- `serve.sh`: optimized production profile above. Supports runtime overrides such as `LLAMA_N_CPU_MOE`, `LLAMA_CTK`, `LLAMA_CTV`, `LLAMA_BATCH`, `LLAMA_UBATCH`, `LLAMA_CTX`, and `LLAMA_PORT`.
- `serve-safe.sh`: fallback with all MoE weights on CPU, Q8 KV, two slots, conservative 131072-token context per slot by default.
- `serve-mtp.sh`: optional MTP test profile on port 8003, 32768 context, one slot. Not used by the main service.
- `benchmark.sh`: legacy/non-production `llama-bench` benchmark; captures before/after GPU/RAM.
- `benchmark-production.sh`: starts the exact server profile and measures authenticated single-slot and dual-slot API throughput plus sampled VRAM/RAM/swap.
- `tune.sh`: reproducible boot/OOM boundary test for `n-cpu-moe`, KV types, batch, ubatch, and load mode.
- `long-context.sh`: exact-token concurrent completion test with `/slots` sampling.
- `healthcheck.sh`: service, port, health, authenticated API, and GPU UUID checks.
- `quick-smoke.sh`: short authenticated chat completion smoke test.
- `stability-test.sh`: configurable repeated single/dual request test with VRAM, temperature, RAM, and `/proc/vmstat` swap counters.
- `status.sh`: GPU/RAM status and authenticated `/v1/models` probe.

Start optimized server:

```bash
cd /home/rugent/gemma4-server
./serve.sh
```

The real API key is in `.env` mode 600 and is not included here. Copy `.env.example` only as a template; never commit `.env`.

## API

```bash
set -a; . /home/rugent/gemma4-server/.env; set +a
curl http://127.0.0.1:8001/v1/models \
  -H "Authorization: Bearer $LLAMA_API_KEY"
```

Gemma 4's reasoning template is enabled by default. For a direct non-thinking smoke test, use:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

The verified smoke response was exactly `GEMMA4_OK` with this option. Two simultaneous requests returned `SLOT_0` and `SLOT_1`; server logs show both `id 0` and `id 1`.

## Measured performance and memory

The old `llama-bench` result below is legacy/non-production because it used `-ngl 99 -ncmoe 99`:

- Prompt processing: 94.61 tok/s (pp128)
- Generation: 33.09 tok/s (tg128)

Exact production API benchmark, RTX 4060 Ti UUID selected:

| Profile | Single PP | Single TG | Dual TG slot 0/1 | Aggregate TG | VRAM peak | RAM peak |
|---|---:|---:|---:|---:|---:|---:|
| Q5_1, ncpu=28 | 45.75 | 30.58 | 20.00 / 20.19 | 40.19 | 5610 MiB | 17.2 GiB |
| Q5_1, ncpu=4 | 122.24 | 61.83 | 49.46 / 49.87 | 99.33 | 15412 MiB | 17.2 GiB |
| Q4_1, ncpu=4 | 118.00 | 65.60 | 49.85 / 49.43 | 99.28 | 15062 MiB | 17.3 GiB |
| Q4_1, ncpu=3 | 134.14 | 70.86 | 52.97 / 53.42 | 106.40 | 15470 MiB | 17.0 GiB |

`load-mode=none` on Q4_1/ncpu=3 measured 70.68 single TG and 104.79 aggregate TG, with approximately 20.2 GiB RAM peak. It was not selected because the improvement was within measurement noise while RAM use was higher. The production default remains `auto`.

## Long-context test

The Q4_1/ncpu=3 profile ran two concurrent completions with tokenizer-generated prompts of exactly 200005 content tokens each. The API timings reported 200013 prompt tokens per request, HTTP 200 for both, and no CUDA OOM or server crash. Wall time was 1010.99 seconds. Slot 0 completed in 417.04 seconds; slot 1 completed in 1010.98 seconds. During the run, `/slots` showed both slots with `n_ctx=262144`; slot 0 reached 200013 processed prompt tokens and slot 1 reached 200013. Maximum sampled RTX 4060 Ti VRAM was 15488 MiB, RAM 22259 MiB, swap 624 MiB.

The test confirms the requested minimum 200k + 200k concurrent operation. It does not claim that 240k + 240k or 250k + 250k was tested.

## Memory allocation and context sizing

The runtime startup log explicitly reports `n_slots = 2`, `n_ctx_slot = 262144`, and `kv_unified = 'true'`. With `--kv-unified-per-slot 262144` and no explicit `-c`, the same commit prints `sizing KV pool to n_parallel * kv_unified_per_slot = 2 * 262144 = 524288`; the model then boots with the same per-slot context. The production launcher keeps explicit `-c 262144` for reproducibility.

The verbose startup log (`logs/startup-verbose-allocation.log`) reports for Q4_1/ncpu=3: CUDA total 15974 MiB, free 523 MiB; CUDA self/model/context/compute = 12348 / 1740 / 1198 MiB; CUDA unaccounted = 163 MiB; Host total = 1741 MiB with model/context/compute = 1706 / 0 / 34 MiB. The normal runtime log does not report these byte totals unless `--verbose` is used. The available `nvidia-smi` readback is the authoritative total VRAM measurement. The tuning/benchmark scripts sample VRAM, RAM, and swap every 0.25–1 second.

## MTP result

The repository's MTP draft was downloaded and loaded successfully with `--spec-type draft-mtp --spec-draft-n-max 4`. A direct API test reported `draft_n=192`, `draft_n_accepted=79`, but generation was 30.77 tok/s versus 33.24 tok/s on the base profile in the measured tests. MTP is therefore disabled in `serve.sh`; use `serve-mtp.sh` only for experiments.

## Stop and troubleshooting

Stop only the process started for this deployment, for example by locating it with `ss -ltnp | grep ':8001'` and sending TERM to its PID. The launcher refuses to start if the port is already occupied.

If the optimized profile fails, use:

```bash
LLAMA_SAFE_PORT=8001 ./serve-safe.sh
```

Check logs under `logs/`. The launcher prints its effective CUDA device, model, context, slot, MoE, KV, batch, Flash Attention, and port settings at startup.

## SYSTEMD SERVICE

Production lifecycle is managed by:

```text
gemma4.service
```

The unit runs as `rugent:rugent`, uses `/home/rugent/gemma4-server/serve.sh`, has `Restart=on-failure`, `RestartSec=5`, `KillSignal=SIGINT`, `TimeoutStopSec=60`, and `LimitNOFILE=65535`.

```bash
sudo systemctl enable --now gemma4.service
sudo systemctl status gemma4.service
sudo systemctl restart gemma4.service
sudo systemctl stop gemma4.service
```

The API key is loaded by `serve.sh` from `.env` and passed to llama.cpp through a mode-600 temporary `--api-key-file`, removed at shutdown. It is not stored in the unit file, printed in the service journal, or exposed in the llama-server command line. `serve-safe.sh`, `serve-mtp.sh`, and the long/production benchmark launchers use the same file-based approach.

## HEALTH CHECK

```bash
./healthcheck.sh
./quick-smoke.sh
```

`healthcheck.sh` verifies systemd active state, port 8001, `/health`, authenticated `/v1/models`, and the RTX 4060 Ti UUID. `quick-smoke.sh` additionally performs a short authenticated chat completion without a long-context prompt.

## LOGS

The production service uses the systemd journal, not an unbounded application log:

```bash
journalctl -u gemma4.service -f
journalctl -u gemma4.service -n 100 --no-pager
```

The deployment's `logs/` directory contains bounded test artifacts and benchmark output. No logrotate rule is required for the systemd service itself.

## VRAM REQUIREMENTS

The RTX 4060 Ti is dedicated to Gemma 4. Do not schedule another CUDA model on this GPU while Gemma is running. The Q4_1/ncpu=3 production profile reaches approximately 15.49 GiB of 16.38 GiB VRAM, leaving only about 523 MiB in the verbose startup allocation. `n-cpu-moe=2` is not an accepted production profile.

The RTX 3050 is used by another service and is intentionally excluded from all Gemma launchers by explicit GPU UUID.

## SWAP BEHAVIOUR

The host has 8 GiB swap. Long-context testing previously showed up to 624 MiB swap used, which alone does not prove active swapping. Use `vmstat 1` or `sar -W 1` and inspect `si`/`so` during inference. `stability-test.sh` also records `/proc/vmstat` `pswpin`/`pswpout` deltas.

The 100k + 100k systemd-service test produced non-zero system-wide swap I/O: 265 KiB `si` and 16831 KiB `so` over 421 samples, with peaks of 40 KiB/s and 16712 KiB/s. The llama-server process itself had `VmSwap: 0 kB`. The largest existing swapped processes were Hermes agent, Elasticsearch, several workers, LiteLLM, and OpenViking; this is host-wide pressure from other services rather than Gemma expert weights being swapped. The test completed both requests with HTTP 200.

The 30-minute stability test ran for 1804.6 seconds with 483 successful requests, 0 failures, 0 CUDA OOMs, and 0 server crashes. It measured 15472 MiB VRAM peak, 19112 MiB RAM peak, 53 °C maximum GPU temperature, 669 MiB maximum swap used, and only 241 `pswpin` plus 249 `pswpout` pages over the full run. The llama process remained unswapped (`VmSwap: 0 kB`); system-wide swap occupancy is inherited host state and must not be interpreted as Gemma expert swapping.

## RECOVERY / FALLBACK

If the optimized service fails:

```bash
sudo systemctl stop gemma4.service
cd /home/rugent/gemma4-server
./serve-safe.sh
```

The safe profile keeps all MoE weights on CPU and uses Q8 KV. For a known-good llama.cpp rollback, use commit `58367713a6935c0810103378144008df32e3d5db`, then rebuild the existing CUDA build; old tuning results are not assumed valid after a runtime change.

## UPGRADE PROCEDURE

After an upgrade to llama.cpp, the driver, or the model runtime:

1. Stop or schedule maintenance for `gemma4.service`.
2. Run `./quick-smoke.sh` after restart.
3. Run `./benchmark-production.sh` with the final profile.
4. Compare startup memory allocation, API throughput, CUDA errors, and health results with this README.
5. Repeat MoE/long-context tuning only if allocation or performance materially changes.

## SECURITY AND NETWORK

The server listens on `0.0.0.0:8001`, but the current UFW policy has no allow rule for port 8001 and defaults to deny incoming traffic. This was not changed automatically. If LAN access is required later, add a narrow LAN-only rule and avoid exposing the raw llama-server directly to the Internet. Preferred architecture is llama-server behind a LAN/reverse proxy with TLS and access control.

## Mixed KV tests (2026-09-23)

### Unified pool vs per-slot -c (current llama.cpp commit)
- `--kv-unified-per-slot N` **without** `-c`: pool sized to `n_parallel*N` (log confirms `sizing KV pool to n_parallel * kv_unified_per_slot = 2 * 196608 = 393216`). All such configs **CUDA OOM at boot** on the 4060 Ti — tensors ~12.5 GB, KV pool ~13.5 GB at K=Q8 exceeds the card before inference.
- `-c N` **with** `--kv-unified` and `-np 2`: per-slot capacity N, shared pool of 2*N cells fits VRAM; long-context verified. This is the working mode.
- `--kv-unified-per-slot` combined with `-c`: per-slot limit only, no pool resize.

### Q8/Q5_1 and Q8/Q4_1 results (196608/slot target, vision CPU projector)
| K | V | b | ub | moe | Boot | Peak VRAM (MiB) |
|---|---|---|----|-----|------|-----------------|
| q8_0 | q5_1 | 64 | 32 | 3 | CUDA OOM (203.91 MiB) | 12484 |
| q8_0 | q5_1 | 32 | 16 | 3 | CUDA OOM | 12484 |
| q8_0 | q5_1 | 16 | 8 | 3 | CUDA OOM | 12484 |
| q8_0 | q5_1 | 64 | 32 | 4 | CUDA OOM | 12076 |
| q8_0 | q5_1 | 16 | 8 | 5 | CUDA OOM (fit_params abort) | 15927 |
| q8_0 | q5_1 | 16 | 8 | 6 | CUDA OOM | 11258 |
| q8_0 | q4_1 | 64 | 32 | 3 | CUDA OOM | 12484 |
| q8_0 | q4_1 | 32 | 16 | 3 | CUDA OOM (1551.81 MiB) | 12484 |
| q8_0 | q4_1 | 32 | 16 | 4 | CUDA OOM | 12609 |

Conclusion: no mixed-QV profile boots with the pure unified pool on 16 GB. Lowering batch/ubatch (down to 16/8) and MoE offload up to 6 layers changed nothing — V-quantization saves only ~5% of KV bytes, and the 393216-cell pool (~13.5 GB at Q8) against ~12.5 GB weights is the sole OOM driver. The unified-pool-only path is a dead end on this hardware.

### Working production profile (serve.sh defaults)
`K=q8_0 V=q8_0 -c 196608 -np 2 --kv-unified --kv-unified-per-slot 196608, fa=on, b=128 ub=1024, moe=3, mmproj F16 CPU`
- Single slot: 194,869-token prompt PASS
- Dual: 2 x 141,719 tokens PASS; 2 x 163k+ FAIL (KV pool exhaustion, not CUDA OOM)
- Peak VRAM 15,802 MiB (~578 MiB headroom); vision PASS (~4.4 s); Flash Attention on CUDA (FA_ALL_QUANTS build)
- `LLAMA_KV_NO_C=1` switches to unified-pool-only mode (OOCs at boot; kept for reference only)

### Final 2x131072 production profile (2026-09-24)

- Model: Gemma 4 26B-A4B QAT UD-Q4_K_XL
- Slots/context: `-np 2`, `--kv-unified`, `--kv-unified-per-slot 131072` (pool 262144)
- KV: `K=q8_0`, `V=q5_1`
- MoE: `--n-cpu-moe 6`
- Batch: `-b 128 -ub 1024`
- `--fit off`, Flash Attention on
- Vision: `mmproj-F16.gguf`, `--no-mmproj-offload`
- Exact 128k tests: single 127949 tokens PASS; dual 127950 + 127950 simultaneous PASS
- Retrieval: 5/5 on three runs at approximately 125k input tokens
- Cold starts: 5/5 with `--fit off`
- Measured RTX 4060 Ti peak headroom: 882 MiB; RAM peak 30587 MiB; swap peak 8038 MiB
- Short generation benchmark: 55.5 tok/s; long 128k generation: 23.4 tok/s
- Q8/Q8 comparison: m5 had 36 MiB headroom; m6 had 446 MiB, below the 500 MiB requirement. Q8/Q5_1 m6 was selected.
- Historical 196608 results remain in `/home/rugent/gemma-4-deploy-results-ctx-2x196608.md`.
- Previous profile: 2 × 196608, Q5_1/Q5_1, `n-cpu-moe=7`.
- New profile: 2 × 131072, Q8_0/Q5_1, `n-cpu-moe=6`.
- Do not reduce `n-cpu-moe` below 6 without a complete retest.
- The selected production profile intentionally keeps approximately 882 MiB peak VRAM headroom.

### Final memory / swap validation

- `monitor-memory.sh` records RAM, swap, `vmstat si/so`, and RTX 4060 Ti VRAM once per sample under `logs/memory-monitor/`.
- Idle baseline: RAM peak 22237 MiB; swap 6673 -> 6530 MiB; `si` peak 27400 KiB/s; `so` 0.
- Short LiteLLM request: RAM peak 22040 MiB; swap 6529 -> 6468 MiB; `si` peak 43572 KiB/s; `so` 0; request PASS.
- Vision LiteLLM request: RAM peak 22050 MiB; swap 6468 -> 6460 MiB; `si` peak 3188 KiB/s; `so` 0; request PASS.
- Long LiteLLM request: 102024 prompt tokens, 1.6 s warm/cache-assisted response; RAM peak 22339 MiB; swap 6408 -> 6394 MiB; `si` peak 948 KiB/s; `so` 0; request PASS.
- Dual LiteLLM request: 102025 + 102025 prompt tokens, 230.0 s; RAM peak 22533 MiB; swap 6439 -> 6415 MiB; `si` peak 276 KiB/s; `so` 0; both requests PASS.
- No significant swap pressure was observed: swap-in activity was intermittent/background reclaim, swap-out stayed zero, and available RAM remained above 24 GiB during the measured workloads.

### Operational warning

RTX 4060 Ti is dedicated to Gemma. Do not start additional CUDA workloads on this GPU. The selected profile maintains approximately 872–882 MiB peak VRAM headroom. RTX 3050 may continue its separate retrieval/STT workloads.

Frozen configuration snapshot: `final-profile-ctx-2x131072/`
