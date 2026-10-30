# Gemma 4 26B-A4B — Integrations Audit: AFTER
Date: 2026-09-24 (CEST)
Source of truth: Gemma 4 26B-A4B QAT UD-Q4_K_XL, llama.cpp, 196608 tokens per request/slot, 2 slots, internal unified KV pool 393216, q5_1 KV per tuning (runtime args per serve.sh: -np 2 -c 196608 --kv-unified --kv-unified-per-slot 196608), vision mmproj F16 (CPU offload), FA CUDA, text+vision+audio capable route.

Legend: context values are TOTAL per-request context. "output" is max generated tokens (never 196608).

| Component | Config | Uses LiteLLM/direct | Model alias | Old context | New total context | Input budget | Output budget | Compaction | Timeout | Concurrency | Changed | Test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma llama-server | /etc/systemd/system/gemma4.service + serve.sh | PRODUCER (port 8001) | gemma-4 | 196608 | 196608 (unchanged, runtime untouched) | — | — | — | — | np=2 internal | NO | health ok |
| LiteLLM | /home/rugent/litellm/proxy_server_config.yaml (docker litellm-litellm-1, port 4000, DB STORE_MODEL_IN_DB=True; PG LiteLLM_ProxyModelTable empty → file is source of truth) | gateway | gemma-4 → openai/gemma-4 @ http://192.168.16.42:8001/v1 | 262144 (wrong) | 196608 (max_input_tokens, both litellm_params + model_info) | implicit (no explicit input cap set; LiteLLM pre-call check uses max_input_tokens=196608 as total-window proxy) | 16384 max_output_tokens | n/a | request_timeout 3600 s (preserved); no per-model timeout set | no explicit per-model concurrency → LiteLLM default router queue, 192.168.16.42:8001 upstream | YES | text/vision/105k/182k PASS via :4000 |
| Hermes Agent (global ~/.hermes) | /home/rugent/.hermes/config.yaml | via LiteLLM local-litellm | gemma-4 | 262144 | 196608 (provider model entry) | implicit | 16384 max_tokens | context_engine none; codex_responses_compact_threshold untouched (only applies to codex provider) | agent.timeout 3600 | — | YES | config parses; gemma route PASS |
| Hermes profile rudo-vasky | /home/rugent/.hermes/profiles/rudo-vasky/config.yaml | via LiteLLM (127.0.0.1:4000/v1) | gemma-4 | 262144 | 196608 | implicit | 16384 | none | request_timeout 3600 (unchanged) | 1 gateway | YES | gateway active, no Invalid model errors |
| Hermes profile exabot | /home/rugent/.hermes/profiles/exabot/config.yaml | via LiteLLM | gemma-4-12b-it (dead alias) → gemma-4 | 262144 + bad alias | 196608 | implicit | 16384 | none | 3600 | — | YES | gemma-4 resolves in LiteLLM /v1/models |
| Hermes profile pc-groky | /home/rugent/.hermes/profiles/pc-groky/config.yaml | via LiteLLM (fallback; primary codex gpt) | gemma-4-12b-it → gemma-4 | 262144 + bad alias | 196608 | implicit | 16384 | codex path untouched | 3600 | — | YES | alias resolvable; no restart needed (no gateway running for this profile) |
| Hermes profile dashboard-nuc | /home/rugent/.hermes/profiles/dashboard-nuc/config.yaml | via LiteLLM (default qwen-prisma) | qwen-prisma (NOT gemma) | 196608 | 196608 unchanged — belongs to qwen-prisma route (coincidence), left alone | implicit | 16384 | none | 3600 | — | NO (reviewed) | — |
| Hermes profile laurincov-pes | /home/rugent/.hermes/profiles/laurincov-pes/config.yaml | via LiteLLM (default qwen-prisma) | qwen-prisma | 243952 (qwen, not gemma) | unchanged (not Gemma) | — | — | — | — | — | NO (not Gemma consumer) | — |
| Hermes profile patjo-web | /home/rugent/.hermes/profiles/patjo-web/config.yaml | via LiteLLM (qwen-prisma) | qwen-prisma | 262144/253952 (qwen entries) | unchanged (not Gemma) | — | — | threshold_tokens 180000 belongs to qwen memory config, left as-is | — | — | NO (not Gemma consumer) | — |
| Hermes profile rugent2rudo | config.yaml | via LiteLLM (qwen-prisma) | qwen-prisma | — | unchanged (not Gemma) | — | — | — | — | — | NO | — |
| agenticidentitytrust/agentsorchestrator/modelqaspecialist/orbit | config.yaml | codex gpt-5.6-luna primary; local-litellm entry lists qwen-prisma only (no gemma key) | no gemma | — | unchanged (no Gemma) | — | — | codex_responses_compact_threshold 200000 = codex-only, untouched | — | — | NO | — |
| OpenViking | /home/rugent/public_repo/openviking-runtime/ov.conf | vlm block: direct gemma-4-12b-it @127.0.0.1:8080 (DEAD) → via LiteLLM http://127.0.0.1:4000/v1 | gemma-4-12b-it → gemma-4 | 131072 (stale) + wrong port | 196608 (max_input_tokens) via gateway | implicit | max_tokens 16384 | n/a | request_timeout 1800 | per-server, low | YES | service healthy :1933; gateway+alias+vision e2e PASS (embeddings/rerank paths unchanged on :8081/:8082) |
| watchdog laurincov-pes | /home/rugent/workspace/laurincov-pes/endpoints.json + scripts (hermes cron 7ae916cfdd51 every 15m) | probes list: Gemma was 127.0.0.1:8080 (dead, false alarms) → 192.168.16.42:8001 (auth-free /health, deterministic) | Gemma | probe: 8080 (dead) | probe: 8001/health | n/a (deterministic health probe, no LLM budget) | n/a | n/a | watchdog request_timeout 30 s (short requests — kept) | sequential probe loop | YES (1 line) | live run: 6 services ok, 1 down (pre-existing: watchtower.service) |
| OpenCode | /home/rugent/.config/opencode/opencode.json | provider local-litellm baseURL :4000/v1 | gemma-4-12b-it → gemma-4 | 131072 + bad alias | 196608 (limit.context) | implicit | 16384 (limit.output) | compaction: none configured (uses context-based auto) | timeout 1800 (chunkTimeout 1800000 ms) | — | YES | alias present in gateway /v1/models |
| Docker | litellm-litellm-1 (restart done) | | | | | | | | | | | |
| cron/timers | hermes cron jobs; no other Gemma consumers found in /etc/cron*, systemd, user units, docker | | | | | | | | | | | |
| direct :8001 | gemma4.service itself (producer); code-intelligence-gateway → :1933/8181 (no gemma refs) | | | | | | | | | | | |

## Semantic diffs (old -> new)

LiteLLM (proxy_server_config.yaml):
  gemma-4 max_input_tokens: 262144 -> 196608 (both litellm_params and model_info)
  gemma-4 max_output_tokens: 16384 (unchanged)
  gemma-4 supports_vision: true (present)
  request_timeout: 3600 s (unchanged)
  api_base: http://192.168.16.42:8001/v1 (unchanged)

Hermes ~/.hermes/config.yaml:
  custom_providers[local-litellm].models.gemma-4.context_length: 262144 -> 196608
  max_tokens: 16384 (unchanged)

Hermes profile rudo-vasky:
  model.context_length: 262144 -> 196608
  default: gemma-4 (unchanged), max_tokens 16384 (unchanged)

Hermes profile exabot:
  default model alias: gemma-4-12b-it -> gemma-4 (dead alias fix)
  gemma-4 context_length: -> 196608

Hermes profile pc-groky:
  fallback alias: gemma-4-12b-it -> gemma-4
  gemma-4 context_length: -> 196608

OpenViking ov.conf:
  vlm: provider=openai api_base http://127.0.0.1:8080/v1 -> http://127.0.0.1:4000/v1 (LiteLLM gateway)
  vlm model: gemma-4-12b-it -> gemma-4
  vlm api_key: literal -> $OPENVIKING_VLM_API_KEY (master key, env)
  vlm max_input_tokens: 131072 -> 196608
  vlm max_tokens: 16384
  (vlm was pointed at a dead port 8080; embeddings/rerank services NOT touched)

watchdog laurincov-pes (endpoints.json):
  Gemma probe URL: http://127.0.0.1:8080/v1/models -> http://192.168.16.42:8001/health (auth-free, deterministic)

OpenCode opencode.json:
  gemma entry id/alias: gemma-4-12b-it -> gemma-4 (provider local-litellm)
  limit.context: 131072 -> 196608
  limit.output: 16384 (unchanged)

## Final health (read-back verified)

Gemma :8001 /health: {"status":"ok"}
LiteLLM :4000 liveliness: ok; /v1/models -> gemma-4 (max_input 196608 / max_output 16384), qwen-prisma, qwen-prisma-500k, qwen3-embedding, qwen3-reranker
/v1/responses via LiteLLM (gemma-4): completed
/v1/chat/completions via LiteLLM (gemma-4): PASS ("E2E-SMOKE")
Vision via LiteLLM (gemma-4, red 384x384): PASS ("Red", prompt_tokens 105)
Long context ~105k exact tokens via LiteLLM: PASS
Long context ~182k exact tokens via LiteLLM: PASS (~125 s)
Hermes rudo-vasky gateway: active, gemma-4 @ 196608, no model errors
OpenViking :1933: healthy, vlm routed via LiteLLM
Watchdog: live probe run clean (Gemma detected OK via 8001)

## Files changed
/home/rugent/litellm/proxy_server_config.yaml
/home/rugent/.hermes/config.yaml
/home/rugent/.hermes/profiles/rudo-vasky/config.yaml
/home/rugent/.hermes/profiles/exabot/config.yaml
/home/rugent/.hermes/profiles/pc-groky/config.yaml
/home/rugent/public_repo/openviking-runtime/ov.conf
/home/rugent/workspace/laurincov-pes/endpoints.json
/home/rugent/.config/opencode/opencode.json

## Services restarted
litellm-litellm-1 (docker) — once
openviking.service (user) — once
hermes-gateway-rudo-vasky.service (user) — once

## Rollbacks required
none

## Acceptance notes / caveats
- qwen-prisma route: qwen3-5 243952/500k entries are QWEN models, not Gemma; left unchanged (in scope only Gemma sync).
- codex_responses_compact_threshold: 200000 belongs to the openai-codex (gpt) compaction path, not Gemma; left unchanged.
- No client got 196608 as max OUTPUT anywhere; output caps kept at 16384.
- 393216 never propagated to any client config; only llama-server serve.sh computes it internally as 2x196608.
- pc-groky alias fixed but its primary is codex gpt; gateway for that profile not running (cron/profile only).
