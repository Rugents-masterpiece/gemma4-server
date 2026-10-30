# Gemma 4 26B-A4B – integration audit BEFORE (196k sync)

Gemerný kontext: Gemma 4 26B-A4B QAT, llama.cpp, context = 196 608 / request, 2 sloty,
interný KV pool 393 216 (NEPROPGOVÁ SA KNIENTOM). Host: pc-agent, GPU RTX 4060 Ti 16 GB.

## Prevádzková mapa (overené LIVE)

| Item | Hodnota |
|---|---|
| Gemma runtime | llama-server pid 2352692, port 8001 (0.0.0.0), alias `gemma-4`, `-c 196608 --kv-unified --kv-unified-per-slot 196608 -np 2 -fa on -ctk q8_0 -ctv q8_0 --n-cpu-moe 3` (spustené cez bash /home/rugent/gemma4-server/serve.sh, unitská cestou gemma4.service) |
| /v1/models read-back | `n_ctx: 196608`, capabilities `["completion","multimodal"]` – PASS |
| Poznámka runtime | aktívny proces má `-ngl all`, `--n-cpu-moe 3`, `q8_0/q8_0` (nie presne profil zo zadania 7/23 q5_1); slúži len ako kontext, meniť sa NEMÁ |
| LiteLLM | Docker compose /home/rugent/litellm, image local/litellm-switchyard:1.97.0, container litellm-litellm-1, host network, port 4000, config bind-mount: /home/rugent/litellm/proxy_server_config.yaml (ro mount v inspect, changeable cez rebuild – reálne je to zdrojový súbor v hosti), DB litellm-db-1 (postgres), redis |
| DB model entries | `LiteLLM_ModelTable` = 0 rows → source of truth je proxy_server_config.yaml (nie DB/Admin UI) |

## Audit tabuľka (pred zmenou)

| Component | Config path | How started | Gemma direct/LiteLLM | Model name | Current context | Current max input | Current max output | Timeout | Concurrency | Needs change |
|---|---|---|---|---|---|---|---|---|---|---|
| LiteLLM proxy | /home/rugent/litellm/proxy_server_config.yaml | docker compose (litellm) | LiteLLM → direct :8001 | gemma-4 (backend openai/gemma-4) | max_input 262144, max_output 16384 (model_info + per-model block) | 262144 | 16384 (litellm_params.max_tokens) | request_timeout 3600, timeout 3600 | default | ÁNO – 262144 je nesprávny (virtuálny, viac ako model), pre-call check by odmietal iba >262144; správne = 196608 total, input cap 180224 |
| Hermes Agent (global) | /home/rugent/.hermes/config.yaml | systemd --user hermes-gateway(+serve) | LiteLLM (custom:local-litellm) | gemma-4 alias v custom_providers | context_length 262144 | n/a (context_length = total) | max_tokens 16384 | timeout_seconds 3600 | n/a | ÁNO – 262144 → 196608 |
| Hermes profil rudo-vasky | /home/rugent/.hermes/profiles/rudo-vasky/config.yaml | systemd --user hermes-gateway-rudo-vasky | LiteLLM :4000 | default gemma-4 | context_length 262144 | n/a | max_tokens 16384 | (ded from provider) | max_turns 20 | ÁNO – 262144 → 196608; overiť že sa nedefinuje aj inde |
| Hermes profil exabot | .../profiles/exabot/config.yaml | systemd --user hermes-gateway-exabot | LiteLLM :4000 | default gemma-4-12b-it | context_length 262144 | n/a | 16384 | n/a | n/a | ÁNO – alias gemma-4-12b-it neexistuje v LiteLLM (fallback do model_cost mapy); alias na gemma-4 + context 196608 |
| Hermes profil pc-groky | .../profiles/pc-groky/config.yaml | user (ine gemma model) | LiteLLM (gemma-4-12b-it alias) | gemma-4-12b-it v custom_providers (262144) | 262144 | n/a | 16384 | 3600 | n/a | ÁNO – alias na existujúcu Gemmu (26B) + 196608, alebo označiť |
| Hermes profil patjo-web | .../profiles/patjo-web/config.yaml | user | LiteLLM :4000 (default qwen-prisma) | qwen-prisma (Hlavný model nie Gemma) | model.context_length 262144 (global default), qwen aliasy 196608/243952/253952 | n/a | 32768 | 3600/1200 | n/a | SKÖR – nie Gemma consumer primary; gemma entries neobsahuje (žiadna zmena) |
| Hermes profil dashboard-nuc | .../profiles/dashboard-nuc/config.yaml | user | LiteLLM :4000 | qwen-prisma (context 196608) | 196608 (už správne pre Qwen? nie Gemma) | n/a | 16384 | 14400/3600 | n/a | NIE – nepoužíva Gemmu |
| Ostatné profily (agenticidentitytrust, agentsorchestrator, modelqaspecialist, orbit, rugent2rudo) | .../config.yaml | systemd/user | openai-codex / iné (LiteLLM len ako secondary nemo/qwen aliasy) | nie Gemma | 243952 (nemo/qwen) | n/a | 16384 | 3600 | n/a | NIE – mení sa iba Gemma routes |
| OpenViking | /home/rugent/public_repo/openviking-runtime/ov.conf (vlm block) | systemd --user openviking | VOLANIE 127.0.0.1:8080 → NIKTO nepočúva na 8080 | gemma-4-12b-it (alias) | neblikované (vlm block: max_concurrent 1) | n/a | n/a | n/a | max_concurrent=1 (OK) | ÁNO – api_base 8080 je mŕtvy (8080 nebolo listening, health OFFLINE). Migrácia na LiteLLM :4000 s model gemma-4 |
| Watchdog laurincov-pes | /home/rugent/workspace/laurincov-pes/endpoints.json + ~/.hermes/profiles/laurincov-pes/config.yaml | Hermes cron interval 15m (script laurincov-pes-watchdog.py) | endpoints: HTTP health iba | n/a (watchdog nevolí LLM pri zdravom stave) | Gemma entry smeruje na 127.0.0.1:8080 → mŕtvy | n/a | n/a | n/a | n/a | ÁNO – presmerovať Gemma health na :8001/v1/models (alebo LiteLLM), inak falošný OFFLINE alert |
| OpenCode | /home/rugent/.config/opencode/opencode.json | človek-spúšťaný CLI (opencode-real), cwd projects | LiteLLM :4000 (provider litellm) | gemma-4-12b-it | limit.context 262144, output 16384 | n/a | 16384 | timeout 3600s, chunkTimeout 1800s | n/a | ÁNO – context 262144 → 196608, alias na gemma-4 |
| OpenWebUI | Docker container open-webui | docker (data/searxng? nie – samostatný) | LiteLLM via OPENAI_API_BASE_URL=host.docker.internal:4000 | model_ids: qwen-prisma, nemo-rugent (gemma NIE) | n/a | n/a | n/a | n/a | n/a | NIE – neobsahuje Gemmu (auditované) |
| Draft-legal agents | /home/rugent/draft-legal (user services) | systemd --user | podozrenie, overiť | — | — | — | — | — | — | preskúmať |

## Klbkávé nálezy

1. Starý context 262144 (falošný – model reálne max 196608) je na: LiteLLM gemma-4,
   Hermes global gemma-4, rudo-vasky, exabot, pc-groky, OpenCode gemma-4-12b-it.
2. Dva aliasy na Gemmu v LiteLLM neexistujú: `gemma-4-12b-it` (exabot, pc-groky, OpenCode)
   NENÍ v LiteLLM model_list → LiteLLM by ho riešil až z model_cost mapy, alebo by padol;
   reálne exabot/OpenCode tento model nepoužívajú aktívne? (exabot default = gemma-4-12b-it!)
   → pri testoch to vysvetlí vyššie.
3. OpenViking VLM mieri na 127.0.0.1:8080, ktorý nič nespúšťa (8080 nebolo listening).
   Watchdog likewise meria Gemmu na 8080 → trvalý OFFLINE stav.
4. LiteLLM backend `http://192.168.16.42:8001/v1` – toto je SELF (pc-agent má 16.42). OK.
5. API keys: len `configured: YES` (source: /home/rugent/litellm/.env env, gemma4-server/.env,
   viking ov.conf env-var odkazy).

## Poznámky sémantiky

- LiteLLM `model_info.max_input_tokens` = capping INPUT (pre-call check `input_tokens > max_input_tokens` → error).
  Nie je to total context. Správne: max_input_tokens = 196608 − 16384 (output) = 180224.
  Total context sa v LiteLLM metadáte explicitne neukladá; model_metadata `context_window`
  je odvozené ako max_input+max_output. Nastavíme preto aj max_output 16384 (už je).
- `litellm_params.max_tokens` = default output ceiling; zostáva 16384.
