# llm-budget-gateway

A small OpenAI-compatible proxy that sits between your tools and [OpenRouter](https://openrouter.ai) and keeps
AI spend under control: **one provider key on the server, model tiers you can edit without a restart, price
ceilings, a daily budget that agents cannot raise, free-model rotation when the budget runs out, and one failover
attempt when the upstream is down.**

I wrote it because several of my own tools (a coding agent, a web agent, automation workflows, batch scripts) each
called models with their own keys and nothing could cap the day's spend or say who spent it. This is a clean rewrite
of the gateway I run for them, extracted into a standalone package with its own tests.

## What it does

| Feature | Behaviour |
|---|---|
| **Tier aliases** | Clients ask for `tier/cheap`; the gateway resolves it from a JSON file (`[primary, backup]`) that is re-read when it changes, so you can swap models with no restart. The backup is sent as OpenRouter's native `models` fallback. |
| **Price guard** | Every model in the list must pass per-million-token ceilings (default 3 in / 6 out USD). Unknown models and variable-price routers (`-1/-1`) are refused with 403. If the price list cannot be fetched, the gateway fails closed (503). |
| **Daily budget** | Spend is booked from the provider's reported cost. Over the budget, paid requests are refused (429) or rerouted to free models. Only an **admin token** can raise *today's* budget (up to a ceiling); it reverts at midnight. Client tokens cannot raise their own limit. |
| **Free rotation** | Round-robin over free models, always with `allow_fallbacks=false` and `data_collection=deny`. A model that fails rests for 10 minutes (circuit breaker). A free model is never used as the backup of a paid one. |
| **Outage failover** | If the upstream is unreachable, out of credit (402) or answers 502/503/504, exactly **one** attempt goes to a second OpenAI-compatible provider. A read timeout is **never** retried: the paid model may still be generating, and retrying could bill twice. |
| **Images** | Only the last user message keeps its images (clients resend the whole history, which would multiply cost). If the last message has an image and the model cannot see it, the request moves to the vision tier. |
| **Accounting** | Per day, model, hour and calling app (`X-Client` header), plus provider-cache hit rate per model. |

## Run it

```bash
pip install -e .
cp .env.example .env && cp config.example.json config.json   # fill in your key and real model ids
set -a && . ./.env && set +a
python -m llm_budget_gateway          # http://127.0.0.1:8000
```

Point any OpenAI-compatible client at `http://127.0.0.1:8000/v1` with `Authorization: Bearer <a token from GATEWAY_TOKENS>`.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $CLIENT_TOKEN" -H "X-Client: my-script" -H "Content-Type: application/json" \
  -d '{"model": "tier/cheap", "messages": [{"role": "user", "content": "hello"}]}'

curl -s http://127.0.0.1:8000/usage -H "Authorization: Bearer $CLIENT_TOKEN"            # today's spend, by model and caller
curl -s http://127.0.0.1:8000/admin/budget -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"usd": 3}'                                                                       # raise today's budget (admin only)
```

The response carries `X-Resolved-Model`, the model that was asked for first.

## Configuration

Environment variables: `OPENROUTER_API_KEY`, `GATEWAY_TOKENS` (comma separated), `ADMIN_TOKEN`, `DAILY_BUDGET_USD`,
`BUDGET_CEILING_USD`, `MAX_IN_USD_PER_M`, `MAX_OUT_USD_PER_M`, `ALLOW_UNKNOWN_MODELS`, `GATEWAY_CONFIG`, `DATA_DIR`,
`GATEWAY_TZ` (the budget day rolls over at midnight in this zone), `UPSTREAM_BASE`.
The JSON file is documented in [`config.py`](src/llm_budget_gateway/config.py) and shown in [`config.example.json`](config.example.json).

## Honest limits

- **The budget is a soft cap.** It is checked before a call and booked after it, so parallel calls or a large `max_tokens`
  can overshoot it a little (the worst overshoot I measured on the original was +8.7%). If you need a hard cap, add a
  reservation step before the call.
- **Outage failover is covered by tests with a simulated upstream, not by a real outage.**
- Free models on OpenRouter come and go and rate-limit hard; the rotation exists to keep non-critical traffic alive,
  not to be a quality tier.
- Single process, single worker: the ledger is a JSON file with atomic writes. It is meant for one person or a small
  team, not for a fleet. LiteLLM, Portkey and OpenRouter itself cover similar ground at larger scale; this is a small,
  readable alternative you can audit in an afternoon.
- No streaming transformation: responses are passed through as bytes, and usage is read from the tail of the stream.

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

52 tests run with a fake upstream (no network, no keys). They cover auth, tier resolution and hot reload, the price guard,
budget exhaustion and rerouting, the admin override, the free-model circuit breaker, outage failover (one attempt,
no retry on read timeout), image trimming and cost accounting.

## License

MIT
