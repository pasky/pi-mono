/**
 * Tests for AgentSession auto-retry behavior.
 *
 * Covers the basic retry cycle (transient error → success, max retries
 * exhausted) and a regression test for the retry-promise race condition.
 *
 * Race condition: Agent.emit() calls _handleAgentEvent synchronously but does
 * not await the returned Promise. Without the fix, _retryPromise was created
 * inside _handleRetryableError — which runs after `await _emitExtensionEvent()`
 * — so waitForRetry() could see undefined and prompt() would resolve while the
 * retry loop was still pending. The fix eagerly creates _retryPromise in the
 * synchronous prefix of _handleAgentEvent (before the first await).
 *
 * The race is reproduced by monkey-patching _emitExtensionEvent to yield to
 * the macrotask queue on agent_end, widening the window between emit()
 * returning and _handleRetryableError executing.
 */

import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@mariozechner/pi-agent-core";
import { type AssistantMessage, type AssistantMessageEvent, EventStream, getModel } from "@mariozechner/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { createTestResourceLoader } from "./utilities.js";

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createAssistantMessage(text: string, overrides?: Partial<AssistantMessage>): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "mock",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
		...overrides,
	};
}

describe("AgentSession retry", () => {
	let session: AgentSession;
	let tempDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-retry-test-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(async () => {
		if (session) {
			session.dispose();
		}
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true });
		}
	});

	/**
	 * Build a session with a streaming function that fails on the first N calls
	 * with a retryable error, then succeeds. Set slowExtensionEmit to widen the
	 * race window (used only for the race condition test).
	 */
	function createSession(opts: { failCount?: number; maxRetries?: number; slowExtensionEmit?: boolean } = {}) {
		const { failCount = 1, maxRetries = 3, slowExtensionEmit = false } = opts;
		let callCount = 0;

		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFn: () => {
				callCount++;
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					if (callCount <= failCount) {
						const msg = createAssistantMessage("", {
							stopReason: "error",
							errorMessage: "overloaded_error",
						});
						stream.push({ type: "start", partial: msg });
						stream.push({ type: "error", reason: "error", error: msg });
					} else {
						const msg = createAssistantMessage("Success");
						stream.push({ type: "start", partial: msg });
						stream.push({ type: "done", reason: "stop", message: msg });
					}
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = new ModelRegistry(authStorage, tempDir);
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		settingsManager.applyOverrides({ retry: { enabled: true, maxRetries, baseDelayMs: 1 } });

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader(),
		});

		if (slowExtensionEmit) {
			// Monkey-patch _emitExtensionEvent to yield to the macrotask queue on
			// agent_end. This simulates extension async work (e.g., logging) and
			// ensures agent.prompt() can resolve — and waitForRetry() can be
			// called — before _handleRetryableError runs, reproducing the race.
			const orig = (session as any)._emitExtensionEvent.bind(session);
			(session as any)._emitExtensionEvent = async (event: { type: string }) => {
				if (event.type === "agent_end") await new Promise((r) => setTimeout(r, 0));
				return orig(event);
			};
		}

		return { session, getCallCount: () => callCount };
	}

	it("retries after a transient error and succeeds", async () => {
		const { session, getCallCount } = createSession({ failCount: 1 });
		const events: string[] = [];
		session.subscribe((e) => {
			if (e.type === "auto_retry_start") events.push(`start:${e.attempt}`);
			if (e.type === "auto_retry_end") events.push(`end:success=${e.success}`);
		});

		await session.prompt("Test");

		expect(getCallCount()).toBe(2);
		expect(events).toEqual(["start:1", "end:success=true"]);
		expect(session.isRetrying).toBe(false);
	});

	it("exhausts max retries and emits failure", async () => {
		// failCount > maxRetries so every call fails
		const { session, getCallCount } = createSession({ failCount: 99, maxRetries: 2 });
		const events: string[] = [];
		session.subscribe((e) => {
			if (e.type === "auto_retry_start") events.push(`start:${e.attempt}`);
			if (e.type === "auto_retry_end") events.push(`end:success=${e.success}`);
		});

		await session.prompt("Test");

		// 1 initial + 2 retries = 3 calls total
		expect(getCallCount()).toBe(3);
		expect(events).toContain("start:1");
		expect(events).toContain("start:2");
		expect(events).toContain("end:success=false");
		expect(session.isRetrying).toBe(false);
	});

	it("prompt() must not resolve before retry completes (race condition regression)", async () => {
		// The slow extension emit on agent_end guarantees agent.prompt() resolves
		// before _handleRetryableError runs. Without the fix (eager _retryPromise
		// creation), waitForRetry() would return immediately and prompt() would
		// resolve while the retry loop was still pending.
		const { session, getCallCount } = createSession({ failCount: 1, slowExtensionEmit: true });
		session.subscribe(() => {});

		await session.prompt("Test");

		expect(getCallCount()).toBe(2);
		expect(session.isRetrying).toBe(false);
	});
});
