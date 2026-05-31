// Google Gemini provider — a thin fromEnv over the shared OpenAICompatProvider.
// Gemini's only bespoke surface is the /v1beta/models/{model} probe (context
// window via inputTokenLimit) and its `?key=` auth on that one endpoint;
// everything else (the generate spine, usage mapping, the numeric-budget →
// reasoning_effort tier translation) is the framework's.

import {
    OpenAICompatProvider,
    parseRequiredInt,
    requireEnv,
    type Provider,
} from "@plurnk/plurnk-providers";

const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

export default class Google {
    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<Provider> {
        const apiKey = requireEnv(env.GEMINI_API_KEY, "GEMINI_API_KEY", "google");
        const fetchTimeoutMs = parseRequiredInt(env.PLURNK_FETCH_TIMEOUT, "PLURNK_FETCH_TIMEOUT", "google");
        const reasonBudget = parseRequiredInt(env.PLURNK_REASON, "PLURNK_REASON", "google");

        const contextSize = await fetchContextSize({ apiKey, model, fetchTimeoutMs });

        return new OpenAICompatProvider({
            model,
            url: `${BASE_URL}/openai/chat/completions`,
            fetchTimeoutMs,
            headers: { Authorization: `Bearer ${apiKey}` },
            contextSize,
            reasonBudget,
            // Gemini 2.5+ thinking models honor reasoning_effort tiers; the
            // framework translates the numeric budget to low/medium/high.
            reasoningStyle: "effort",
            // Tokenizer left to the framework default (chars/4 heuristic):
            // Gemini's exact REST countTokens is async-per-call and the
            // Provider contract declares countTokens sync. costFor likewise
            // left to default 0 — Gemini exposes no runtime pricing API.
        });
    }
}

// /v1beta/models/{model} requires `?key=` (Bearer 401s on this endpoint
// per Google's docs). Returns model metadata including inputTokenLimit.
type ModelInfoResponse = { inputTokenLimit?: number };

const fetchContextSize = async ({
    apiKey, model, fetchTimeoutMs,
}: { apiKey: string; model: string; fetchTimeoutMs: number }): Promise<number> => {
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
    return data.inputTokenLimit;
};
