// Google Gemini provider — a thin fromEnv over the shared OpenAICompatProvider.
// Gemini's only bespoke surface is the /v1beta/models/{model} probe (context
// window via inputTokenLimit) and its `?key=` auth on that one endpoint;
// everything else (the generate spine, usage mapping, the numeric-budget →
// reasoning_effort tier translation) is the framework's.

import {
    OpenAICompatProvider,
    computeCost,
    parseRequiredInt,
    reasoningFromEnv,
    dataCaptureFromEnv,
    parseRequiredFloat,
    providerSource,
    requireEnv,
    envelopeFromEnv,
    type Provider,
    type ProviderUsage,
    type ReserveSpec,
} from "@plurnk/plurnk-providers";

const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

// Gemini exposes no runtime pricing endpoint (unlike openrouter/xai/cloudflare),
// so rates are a static table — paid-tier text prices from
// ai.google.dev/gemini-api/docs/pricing, captured 2026-05-31. Stored as
// pico-USD per token ($/1M tokens × 1e6). `long` carries the >200k-prompt tier
// where it differs (Pro models); Flash/Flash-Lite are flat.
type Rate = { input: number; output: number; cached: number };
type GeminiPricing = Rate & { long?: Rate };

const M = 1e6; // $/1M-tokens → pico-USD/token
// Longest-prefix match wins, so flash-lite must precede flash.
const PRICING_BY_PREFIX: ReadonlyArray<[string, GeminiPricing]> = Object.freeze([
    ["gemini-2.5-flash-lite", { input: 0.10 * M, output: 0.40 * M, cached: 0.01 * M }],
    ["gemini-2.5-flash", { input: 0.30 * M, output: 2.50 * M, cached: 0.03 * M }],
    ["gemini-2.5-pro", { input: 1.25 * M, output: 10.0 * M, cached: 0.125 * M, long: { input: 2.50 * M, output: 15.0 * M, cached: 0.25 * M } }],
    ["gemini-3.1-flash-lite", { input: 0.25 * M, output: 1.50 * M, cached: 0.025 * M }],
    ["gemini-3.1-pro", { input: 2.00 * M, output: 12.0 * M, cached: 0.20 * M, long: { input: 4.00 * M, output: 18.0 * M, cached: 0.40 * M } }],
    ["gemini-3.5-flash", { input: 1.50 * M, output: 9.00 * M, cached: 0.15 * M }],
]);

const LONG_CONTEXT_THRESHOLD = 200_000;

const pricingForModel = (model: string): GeminiPricing | null => {
    let best: { prefix: string; rate: GeminiPricing } | null = null;
    for (const [prefix, rate] of PRICING_BY_PREFIX) {
        if (model.startsWith(prefix) && (best === null || prefix.length > best.prefix.length)) best = { prefix, rate };
    }
    return best?.rate ?? null;
};

// Cost from the static table, delegated to the framework's computeCost (which
// bills completion + reasoning at the output rate). Picks the >200k long tier
// where the model defines one. Unknown model → 0 (SPEC §2: no known rates).
export const geminiCostFor = (pricing: GeminiPricing | null, usage: ProviderUsage): number => {
    if (pricing === null) return 0;
    const tier = pricing.long !== undefined && usage.prompt > LONG_CONTEXT_THRESHOLD ? pricing.long : pricing;
    return computeCost(usage, { input: tier.input, output: tier.output, cached: tier.cached });
};

export default class Google {
    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<Provider> {
        // GEMINI_API_KEY wins over GOOGLE_API_KEY (the genai SDK does the reverse,
        // but preferring the Gemini-specific var avoids a stray gcloud GOOGLE_API_KEY
        // silently hijacking the intended key). Both are first-class in the wild.
        const apiKey = requireEnv(env.GEMINI_API_KEY || env.GOOGLE_API_KEY, "GEMINI_API_KEY or GOOGLE_API_KEY", "google");
        const fetchTimeoutMs = parseRequiredInt(env.PLURNK_PROVIDERS_FETCH_TIMEOUT, "PLURNK_PROVIDERS_FETCH_TIMEOUT", "google");
        const streamIdleTimeoutMs = parseRequiredInt(env.PLURNK_PROVIDERS_STREAM_IDLE_TIMEOUT, "PLURNK_PROVIDERS_STREAM_IDLE_TIMEOUT", "google");
        const reasoning = reasoningFromEnv(env, "google");

        const { contextWindow, maxOutput } = await fetchModelInfo({ apiKey, model, fetchTimeoutMs });
        const pricing = pricingForModel(model);

        // #507: completion cap — live outputTokenLimit caps the env percentage, same
        // logic as standardProviders' catalog maxOutput cap. An operator absolute pin
        // (COMPLETION_RESERVE=8192 not "25%") always wins. Gemini reports outputTokenLimit
        // on the same /v1beta/models/{model} call as inputTokenLimit.
        const { reasoningReserve: envReasoning, completionReserve: envCompletion } = envelopeFromEnv(env, "google");
        const completionReserve: ReserveSpec = (() => {
            if ("tokens" in envCompletion) return envCompletion; // operator absolute always wins
            if (maxOutput === undefined) return envCompletion;
            return { tokens: Math.min(maxOutput, Math.round(envCompletion.percent * contextWindow)) };
        })();
        // #568: reasoning reserve — exact REASONING_BUDGET > half-completion > env%.
        const resolvedCompletion = "tokens" in completionReserve
            ? completionReserve.tokens
            : Math.round(completionReserve.percent * contextWindow);
        const reasoningReserve: ReserveSpec = (() => {
            if ("tokens" in envReasoning) return envReasoning; // operator absolute always wins
            if (reasoning.budget !== null) return { tokens: reasoning.budget }; // exact budget when reasoning=on
            return { tokens: Math.round(resolvedCompletion / 2) }; // half-completion for adaptive
        })();

        return new OpenAICompatProvider({
            model,
            url: `${BASE_URL}/openai/chat/completions`,
            fetchTimeoutMs,
            streamIdleTimeoutMs,
            headers: { Authorization: `Bearer ${apiKey}` },
            contextWindow,
            reasoning,
            temperature: parseRequiredFloat(env.PLURNK_PROVIDERS_TEMPERATURE, "PLURNK_PROVIDERS_TEMPERATURE", "google", 0),
            repeatPenalty: parseRequiredFloat(env.PLURNK_PROVIDERS_REPEAT_PENALTY, "PLURNK_PROVIDERS_REPEAT_PENALTY", "google", 0),
            // No frequency_penalty on google: Gemini's OpenAI-compat endpoint REJECTS it
            // (400 -- measured against gemini-2.5-flash, capability-probe #568), along with
            // presence/repetition_penalty and top_k/min_p; it honors only temperature + top_p.
            // Sending the PLURNK_PROVIDERS_FREQUENCY_PENALTY floor (0.4) 400s every call. A
            // caller still passes a penalty via `sampling` if a future model accepts it;
            // omitting -> OpenAICompat's 0-default (nothing sent). Mirrors xai (providers-xai#2).
            reasoningReserve,
            completionReserve,
            retryDelayMs: parseRequiredInt(env.PLURNK_PROVIDERS_RETRY_DELAY, "PLURNK_PROVIDERS_RETRY_DELAY", "google"),
            retryAttempts: parseRequiredInt(env.PLURNK_PROVIDERS_RETRY_ATTEMPTS, "PLURNK_PROVIDERS_RETRY_ATTEMPTS", "google"),
            // Opt-in data capture (#36), off by default, per-alias-scopable.
            ...dataCaptureFromEnv(env, "google"),
            // Gemini 2.5+ thinking models honor reasoning_effort tiers; the
            // framework translates the numeric budget to low/medium/high.
            reasoningStyle: "effort",
            // Tokenizer left to the framework default (chars/4 heuristic):
            // Gemini's exact REST countTokens is async-per-call and the
            // Provider contract declares countTokens sync.
            costFor: (usage) => geminiCostFor(pricing, usage),
            source: providerSource("google"),
        });
    }
}

// /v1beta/models/{model} requires `?key=` (Bearer 401s on this endpoint
// per Google's docs). Returns inputTokenLimit (context window) and outputTokenLimit
// (the model's hard completion cap, used to bound completionReserve; #507/#568).
type ModelInfoResponse = { inputTokenLimit?: number; outputTokenLimit?: number };

const fetchModelInfo = async ({
    apiKey, model, fetchTimeoutMs,
}: { apiKey: string; model: string; fetchTimeoutMs: number }): Promise<{ contextWindow: number; maxOutput: number | undefined }> => {
    const url = `${BASE_URL}/models/${encodeURIComponent(model)}?key=${encodeURIComponent(apiKey)}`;
    const res = await fetch(url, { signal: AbortSignal.timeout(fetchTimeoutMs) });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`Gemini /v1beta/models/${model} returned ${res.status}: ${body}`);
    }
    const data = (await res.json()) as ModelInfoResponse;
    if (data.inputTokenLimit === undefined || data.inputTokenLimit <= 0) {
        throw new Error(`Gemini /v1beta/models/${model} has no inputTokenLimit`);
    }
    const maxOutput = typeof data.outputTokenLimit === "number" && data.outputTokenLimit > 0
        ? data.outputTokenLimit
        : undefined;
    return { contextWindow: data.inputTokenLimit, maxOutput };
};
