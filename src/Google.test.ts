import test, { type TestContext } from "node:test";
import assert from "node:assert/strict";
import Google from "./Google.ts";

// Minimum env that satisfies all required guards in fromEnv. Tests that need
// to exercise one specific knob override its key on top of this.
const baseEnv = Object.freeze({
    GEMINI_API_KEY: "k-test",
    PLURNK_FETCH_TIMEOUT: "600000",
    PLURNK_REASON: "0",
});

const stubFetchOk = (t: TestContext, inputTokenLimit = 1_048_576): void => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({ inputTokenLimit }),
    })) as unknown as typeof fetch;
};

test("fromEnv: throws when GEMINI_API_KEY is unset", async () => {
    await assert.rejects(
        () => Google.fromEnv({}, "gemini-2.5-flash"),
        /GEMINI_API_KEY must be set/,
    );
});

test("fromEnv: throws when PLURNK_FETCH_TIMEOUT is unset", async () => {
    await assert.rejects(
        () => Google.fromEnv({ GEMINI_API_KEY: "k-test", PLURNK_REASON: "0" }, "gemini-2.5-flash"),
        /PLURNK_FETCH_TIMEOUT must be set/,
    );
});

test("fromEnv: throws when PLURNK_FETCH_TIMEOUT is non-numeric", async () => {
    await assert.rejects(
        () => Google.fromEnv({ ...baseEnv, PLURNK_FETCH_TIMEOUT: "abc" }, "gemini-2.5-flash"),
        /PLURNK_FETCH_TIMEOUT must be a number/,
    );
});

test("fromEnv: throws when PLURNK_REASON is unset", async () => {
    await assert.rejects(
        () => Google.fromEnv({ GEMINI_API_KEY: "k-test", PLURNK_FETCH_TIMEOUT: "600000" }, "gemini-2.5-flash"),
        /PLURNK_REASON must be set/,
    );
});

test("fromEnv: throws when PLURNK_REASON is non-numeric", async () => {
    await assert.rejects(
        () => Google.fromEnv({ ...baseEnv, PLURNK_REASON: "lots" }, "gemini-2.5-flash"),
        /PLURNK_REASON must be a number/,
    );
});

test("fromEnv: resolves contextSize from /v1beta/models/{model}", async (t) => {
    stubFetchOk(t, 1_048_576);
    const p = await Google.fromEnv({ ...baseEnv }, "gemini-2.5-flash");
    assert.equal(p.contextSize, 1_048_576);
    assert.equal(p.model, "gemini-2.5-flash");
});

test("fromEnv: uses ?key=<apiKey> on the models endpoint", async (t) => {
    let captured: string | URL | null = null;
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async (url: string | URL) => {
        captured = url;
        return { ok: true, json: async () => ({ inputTokenLimit: 32768 }) };
    }) as unknown as typeof fetch;

    await Google.fromEnv({ ...baseEnv }, "gemini-1.5-pro");
    assert.match(String(captured), /\?key=k-test$/);
});

test("fromEnv: throws when inputTokenLimit absent", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: true,
        json: async () => ({}),
    })) as unknown as typeof fetch;

    await assert.rejects(
        () => Google.fromEnv({ ...baseEnv }, "unknown-model"),
        /no inputTokenLimit/,
    );
});

test("fromEnv: throws when models endpoint returns non-2xx", async (t) => {
    const originalFetch = globalThis.fetch;
    t.after(() => { globalThis.fetch = originalFetch; });
    globalThis.fetch = (async () => ({
        ok: false,
        status: 403,
        text: async () => "permission denied",
    })) as unknown as typeof fetch;

    await assert.rejects(
        () => Google.fromEnv({ ...baseEnv }, "gemini-2.5-flash"),
        /returned 403/,
    );
});

test("contextSize and model exposed on instance", () => {
    const p = new Google({
        apiKey: "k", model: "gemini-2.5-flash", contextSize: 1_048_576,
        fetchTimeoutMs: 1, reasonBudget: 0,
    });
    assert.equal(p.contextSize, 1_048_576);
    assert.equal(p.model, "gemini-2.5-flash");
});

test("costFor: returns 0 unconditionally", () => {
    const p = new Google({
        apiKey: "k", model: "m", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
    });
    assert.equal(p.costFor({ prompt: 10000, completion: 5000, cached: 1000, total: 16000 }), 0);
});

test("countTokens: heuristic returns 0 for empty, ceil(len/4) otherwise", () => {
    const p = new Google({
        apiKey: "k", model: "m", contextSize: 1, fetchTimeoutMs: 1, reasonBudget: 0,
    });
    assert.equal(p.countTokens(""), 0);
    assert.equal(p.countTokens("abcd"), 1);
    assert.equal(p.countTokens("abcde"), 2);
});
