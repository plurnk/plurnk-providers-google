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

| Variable | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | yes | AI Studio key from aistudio.google.com/apikey |
| `PLURNK_REASON` | no | Universal reasoning budget; sibling translates to `reasoning_effort` tier (low / medium / high) on the OpenAI-compat endpoint for 2.5+ thinking models |
| `PLURNK_PROVIDER_FETCH_TIMEOUT` | no | Universal fetch timeout in ms; default `600000` |

## context window

Real, pulled from `GET /v1beta/models/{model}?key={key}` at `fromEnv` time. Gemini exposes `inputTokenLimit` directly. The lookup uses the API key in the query string (AI Studio's `models.get` requires `?key=` auth; Bearer is rejected on this specific endpoint).

## pricing

`costFor` returns 0. Gemini has no documented runtime API that exposes per-model token pricing — the Cloud Billing SKU catalog (`cloudbilling.googleapis.com`) requires a separate GCP service account with `roles/billing.viewer` and the SKU→model mapping is fragile substring matching on free-text descriptions. Operationally non-viable for the AI-Studio-key use case.

Pass-2 may revisit with an opt-in `GEMINI_PRICING_SOURCE=cloud_billing` env for operators willing to provision the heavier credentials. Until then, cost_pico stays at zero.

## tokenization

Heuristic 4-chars-per-token. Gemini's REST `countTokens` endpoint (`/v1beta/models/{model}:countTokens`) is real and exact, but each call is an async round-trip; wiring it into the synchronous `provider.countTokens(text): number` contract has a latency cost that pass-2 will evaluate (likely behind an opt-in env once the precision matters).

## reasoning

Gemini 2.5+ models support `reasoning_effort: low|medium|high` on the OpenAI-compat endpoint. PLURNK_REASON translates as:

| PLURNK_REASON | reasoning_effort |
|---|---|
| `0` (default) | omit |
| `1`–`1000` | `low` |
| `1001`–`4000` | `medium` |
| `4001`+ | `high` |

Models without thinking support (1.5 family) ignore the field.

## license

MIT.
