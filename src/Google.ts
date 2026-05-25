import { chatCompletionStream, OpenAiHttpError } from "./openaiStream.ts";

const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";
const COMPAT_URL = `${BASE_URL}/openai`;

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type ProviderUsage = {
    prompt: number;
    completion: number;
    cached: number;
    total: number;
};

export type ProviderAssistant = {
    content: string;
    reasoning: string | null;
    usage: ProviderUsage;
    finishReason: string | null;
    model: string;
};

export type ProviderResponse = {
    assistant: ProviderAssistant;
    assistantRaw: unknown;
};

export type GoogleConfig = {
    apiKey: string;
    model: string;
    contextSize: number;
    fetchTimeoutMs: number;
    // PROVIDERS.md §3.8: numeric budget → reasoning_effort tier translation
    // for Gemini 2.5+ thinking models. 1.5-family ignores the field.
    reasonBudget: number;
};

export default class Google {
    #apiKey: string;
    #model: string;
    #contextSize: number;
    #fetchTimeoutMs: number;
    #reasonBudget: number;

    constructor(config: GoogleConfig) {
        this.#apiKey = config.apiKey;
        this.#model = config.model;
        this.#contextSize = config.contextSize;
        this.#fetchTimeoutMs = config.fetchTimeoutMs;
        this.#reasonBudget = config.reasonBudget;
    }

    static async fromEnv(env: NodeJS.ProcessEnv, model: string): Promise<Google> {
        const apiKey = env.GEMINI_API_KEY;
        if (apiKey === undefined || apiKey.length === 0) {
            throw new Error("google provider: GEMINI_API_KEY must be set");
        }
        const fetchTimeoutMs = parseRequiredInt(env.PLURNK_FETCH_TIMEOUT, "PLURNK_FETCH_TIMEOUT");
        const reasonBudget = parseRequiredInt(env.PLURNK_REASON, "PLURNK_REASON");
        const contextSize = await fetchContextSize({ apiKey, model, fetchTimeoutMs });
        return new Google({
            apiKey, model, contextSize, fetchTimeoutMs, reasonBudget,
        });
    }

    get contextSize(): number { return this.#contextSize; }
    get model(): string { return this.#model; }

    // Heuristic. Gemini's REST countTokens API exists at
    // /v1beta/models/{model}:countTokens and would give exact counts,
    // but it's async per-call. The plurnk-service Provider contract
    // declares countTokens as sync (number), and packet assembly calls
    // it 3-5 times per turn for section subtotals — wiring real REST
    // tokenization would either (a) require evolving the contract to
    // async, or (b) pre-encode the entire packet ahead of time. Neither
    // is in scope today. Documenting the gap rather than papering over
    // it with a contract-violating hack.
    //
    // For Gemini-family content the heuristic's chars/4 ratio is a
    // moderate underestimate (Gemini's sentencepiece tokenizer tends
    // to chars/3.5 for English in practice). Operators tracking
    // budgets tightly should use Gemini's actual usage numbers (which
    // come back exact on every wire response) rather than relying on
    // the packet subtotals.
    countTokens(text: string): number {
        return text.length === 0 ? 0 : Math.ceil(text.length / 4);
    }

    // Gemini exposes no runtime pricing API. Cost stays 0 until pass-2.
    costFor(_usage: ProviderUsage): number { return 0; }

    async generate({ messages, signal }: { messages: ChatMessage[]; signal?: AbortSignal }): Promise<ProviderResponse> {
        const headers: Record<string, string> = {
            Authorization: `Bearer ${this.#apiKey}`,
        };
        const body: Record<string, unknown> = { model: this.#model, messages };
        const effort = reasoningEffortFromBudget(this.#reasonBudget);
        if (effort !== null) body.reasoning_effort = effort;

        const timeoutSignal = AbortSignal.timeout(this.#fetchTimeoutMs);
        const effectiveSignal = signal !== undefined ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;

        const raw = await chatCompletionStream({
            url: `${COMPAT_URL}/chat/completions`,
            headers,
            body,
            signal: effectiveSignal,
        });

        const usage: ProviderUsage = {
            prompt: raw.usage?.prompt_tokens ?? 0,
            completion: raw.usage?.completion_tokens ?? 0,
            cached: raw.usage?.cached_tokens ?? 0,
            total: raw.usage?.total_tokens ?? 0,
        };

        return {
            assistant: {
                content: raw.content,
                reasoning: raw.reasoning_content.length > 0 ? raw.reasoning_content : null,
                usage,
                finishReason: raw.finish_reason,
                model: raw.model ?? this.#model,
            },
            assistantRaw: raw,
        };
    }
}

const reasoningEffortFromBudget = (budget: number): "low" | "medium" | "high" | null => {
    if (budget <= 0) return null;
    if (budget <= 1000) return "low";
    if (budget <= 4000) return "medium";
    return "high";
};

const parseRequiredInt = (raw: string | undefined, name: string): number => {
    if (raw === undefined || raw.length === 0) {
        throw new Error(`google provider: ${name} must be set`);
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) {
        throw new Error(`google provider: ${name} must be a number (got "${raw}")`);
    }
    return n;
};

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

export { OpenAiHttpError };
