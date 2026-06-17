# @plurnk/plurnk-providers-google

Google Gemini provider for [plurnk-service](https://github.com/plurnk/plurnk-service). Routes `google/{model}` aliases through Gemini's OpenAI-compatible chat-completions endpoint at `generativelanguage.googleapis.com/v1beta/openai`.

## install

```
npm install @plurnk/plurnk-providers-google
```

Requires Node ≥ 25 (native TypeScript).

## use

```ts
import Google from "@plurnk/plurnk-providers-google";

const provider = await Google.fromEnv(process.env, "gemini-2.5-flash");
```

## env

No fallback defaults — required vars throw at `fromEnv` if missing or unparseable. Defaults belong in `plurnk-service`'s `.env.example` cascade, not in library code.

| Variable | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | yes | AI Studio key from aistudio.google.com/apikey |
| `PLURNK_PROVIDERS_REASONING_BUDGET` | yes | Universal reasoning budget (SPEC §4); sibling translates to `reasoning_effort` tier (low / medium / high) on the OpenAI-compat endpoint for 2.5+ thinking models. `0` disables |
| `PLURNK_FETCH_TIMEOUT` | yes | Universal fetch timeout in ms (SPEC §4) |
| `PLURNK_PROVIDER_RETRY_ATTEMPTS` | yes | Transient-failure retry budget (SPEC §4): `0` disables; `N` retries on 429/5xx/timeout/network with exponential backoff, honoring `Retry-After`. |

## context window

Real, pulled from `GET /v1beta/models/{model}?key={key}` at `fromEnv` time. Gemini exposes `inputTokenLimit` directly. The lookup uses the API key in the query string (AI Studio's `models.get` requires `?key=` auth; Bearer is rejected on this specific endpoint).

## pricing

`costFor` returns 0. Gemini has no documented runtime API that exposes per-model token pricing — the Cloud Billing SKU catalog (`cloudbilling.googleapis.com`) requires a separate GCP service account with `roles/billing.viewer` and the SKU→model mapping is fragile substring matching on free-text descriptions. Operationally non-viable for the AI-Studio-key use case.

Pass-2 may revisit with an opt-in `GEMINI_PRICING_SOURCE=cloud_billing` env for operators willing to provision the heavier credentials. Until then, cost_pico stays at zero.

## tokenization

Heuristic `~4 chars/token`. Gemini does not have a synchronous tokenizer on npm — its sentencepiece variant isn't published in the standard tokenizer ecosystem the way `gpt-tokenizer` (cl100k_base) or `llama-tokenizer-js` are.

Gemini's REST `countTokens` endpoint (`/v1beta/models/{model}:countTokens`) is real and exact, but every call is an async round-trip. The plurnk-service `Provider.countTokens(text): number` contract is synchronous and gets invoked 3-5 times per turn during packet assembly. Wiring real REST tokenization would either:

1. Require evolving the contract to async (cost: every other sibling pays a Promise wrap for no benefit; engine's packet-build hot path takes the trampoline)
2. Or pre-encode the entire packet ahead of time as a batch call (cost: complex caching, ordering, and invalidation logic)

Neither is in scope. Honest gap, documented.

**Practical workaround:** packet subtotals are slightly low for Gemini traffic (chars/4 vs Gemini's ~chars/3.5 reality). Operators tracking budgets tightly should consult Gemini's wire-reported `usage.prompt` on each completion, which IS exact and lands in `turns.usage_prompt` accurately. Don't rely on `packet.system.tokens` / `packet.user.tokens` for tight budgeting on Gemini routes.

## reasoning

Gemini 2.5+ models support `reasoning_effort: low|medium|high` on the OpenAI-compat endpoint. PLURNK_PROVIDERS_REASONING_BUDGET translates as:

| PLURNK_PROVIDERS_REASONING_BUDGET | reasoning_effort |
|---|---|
| `0` (default) | omit |
| `1`–`1000` | `low` |
| `1001`–`4000` | `medium` |
| `4001`+ | `high` |

Models without thinking support (1.5 family) ignore the field.

## license

MIT.
