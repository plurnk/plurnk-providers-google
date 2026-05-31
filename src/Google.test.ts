import test, { mock } from "node:test";
import assert from "node:assert/strict";
import Google from "./Google.ts";

// Minimum env that satisfies all required guards in fromEnv. Tests that need
// to exercise one specific knob override its key on top of this.
const baseEnv = Object.freeze({
    GEMINI_API_KEY: "k-test",
    PLURNK_FETCH_TIMEOUT: "600000",
    PLURNK_REASON: "0",
});

// Mock the /v1beta/models/{model} probe. Returns the model-info JSON and
// records the requested URLs for endpoint/auth assertions.
const mockModelInfo = (info: unknown) => {
    const calls: string[] = [];
    mock.method(globalThis, "fetch", async (url: string) => {
        calls.push(String(url));
        return new Response(JSON.stringify(info), { status: 200 });
    });
    return calls;
};
test.afterEach(() => mock.restoreAll());

// — fromEnv env guards —

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
    mockModelInfo({ inputTokenLimit: 1_048_576 });
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
    mockModelInfo({ inputTokenLimit: 1_048_576 });
    await assert.rejects(
        () => Google.fromEnv({ ...baseEnv, PLURNK_REASON: "lots" }, "gemini-2.5-flash"),
        /PLURNK_REASON must be a number/,
    );
});

// — model-info probe —

test("fromEnv: resolves contextSize from /v1beta/models/{model}", async () => {
    mockModelInfo({ inputTokenLimit: 1_048_576 });
    const p = await Google.fromEnv({ ...baseEnv }, "gemini-2.5-flash");
    assert.equal(p.contextSize, 1_048_576);
    assert.equal(p.model, "gemini-2.5-flash");
});

test("fromEnv: uses ?key=<apiKey> on the models endpoint (not Bearer)", async () => {
    const calls = mockModelInfo({ inputTokenLimit: 32768 });
    await Google.fromEnv({ ...baseEnv }, "gemini-1.5-pro");
    assert.match(calls[0]!, /\/v1beta\/models\/gemini-1\.5-pro\?key=k-test$/);
});

test("fromEnv: throws when inputTokenLimit absent", async () => {
    mockModelInfo({});
    await assert.rejects(
        () => Google.fromEnv({ ...baseEnv }, "unknown-model"),
        /no inputTokenLimit/,
    );
});

test("fromEnv: throws when models endpoint returns non-2xx", async () => {
    mock.method(globalThis, "fetch", async () => new Response("permission denied", { status: 403 }));
    await assert.rejects(
        () => Google.fromEnv({ ...baseEnv }, "gemini-2.5-flash"),
        /returned 403/,
    );
});

// — Provider surface on the constructed instance —

test("costFor: returns 0 unconditionally (no runtime pricing)", async () => {
    mockModelInfo({ inputTokenLimit: 1_048_576 });
    const p = await Google.fromEnv({ ...baseEnv }, "gemini-2.5-flash");
    assert.equal(p.costFor({ prompt: 10000, completion: 5000, cached: 1000, total: 16000 }), 0);
});

test("countTokens: heuristic returns 0 for empty, ceil(len/4) otherwise", async () => {
    mockModelInfo({ inputTokenLimit: 1_048_576 });
    const p = await Google.fromEnv({ ...baseEnv }, "gemini-2.5-flash");
    assert.equal(p.countTokens(""), 0);
    assert.equal(p.countTokens("abcd"), 1);
    assert.equal(p.countTokens("abcde"), 2);
});
