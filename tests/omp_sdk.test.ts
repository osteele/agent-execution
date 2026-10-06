import { afterEach, expect, test } from "bun:test";
import {
	chmod,
	link,
	mkdtemp,
	mkdir,
	readFile,
	realpath,
	rm,
	stat,
	symlink,
	writeFile,
} from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import type { WriterCredentials } from "../src/agent_execution/omp_sdk";
import {
	antigravityOAuthCredential,
	confinedPath,
	readTools,
	requireAntigravityOAuth,
	SDK_VERSION,
	writerReadiness,
	writeTools,
} from "../src/agent_execution/omp_sdk";

const temporary: string[] = [];

/** SDK-registry stand-in answering exact lookups for only the listed models. */
function registryOf(...selectors: string[]) {
	const models = selectors.map((selector) => {
		const [provider, id] = selector.split("/");
		return { provider, id };
	});
	return {
		find: (provider: string, id: string) =>
			models.find((model) => model.provider === provider && model.id === id),
	};
}

/** SDK credential-store stand-in: OAuth entries also count as stored credentials. */
function credentialsOf({
	oauth = [],
	stored = [],
}: { oauth?: string[]; stored?: string[] }): WriterCredentials {
	return {
		hasOAuth: (provider) => oauth.includes(provider),
		has: (provider) => oauth.includes(provider) || stored.includes(provider),
	};
}

test("an exact model absent from the SDK registry is unavailable despite valid auth", () => {
	const selector = "openai-codex/gpt-6.1-sol";
	const [row] = writerReadiness(
		[selector],
		registryOf("openai-codex/gpt-6-sol"),
		credentialsOf({ oauth: ["openai-codex"] }),
		{},
	);
	expect(row).toMatchObject({
		selector,
		model_available: false,
		credential_available: true,
		available: false,
	});
	expect(row.detail.trim()).not.toBe("");
});

test("a registry answering with another provider or model is not the requested model", () => {
	const credentials = credentialsOf({ oauth: ["openai-codex"] });
	for (const substitute of [
		{ provider: "openai", id: "gpt-6.1-sol" },
		{ provider: "openai-codex", id: "gpt-6-sol" },
	]) {
		const [row] = writerReadiness(
			["openai-codex/gpt-6.1-sol"],
			{ find: () => substitute },
			credentials,
			{},
		);
		expect(row.model_available).toBe(false);
		expect(row.available).toBe(false);
	}
});

test("a registered model without execution-eligible credentials is unavailable", () => {
	const selector = "kimi-code/k3";
	const [row] = writerReadiness(
		[selector],
		registryOf(selector),
		credentialsOf({ stored: ["openai-codex"] }),
		{},
	);
	expect(row).toMatchObject({
		model_available: true,
		credential_available: false,
		available: false,
	});
	expect(row.detail.trim()).not.toBe("");
});

test("OAuth-only writer routes refuse an API-key-only identity", () => {
	const selectors = [
		"anthropic/claude-opus-5-5",
		"google-antigravity/gemini-3.1-pro",
	];
	const registry = registryOf(...selectors);
	const keyOnly = writerReadiness(
		selectors,
		registry,
		credentialsOf({ stored: ["anthropic", "google-antigravity"] }),
		{ ANTHROPIC_API_KEY: "metered-key" },
	);
	for (const row of keyOnly) {
		expect(row.model_available).toBe(true);
		expect(row.credential_available).toBe(false);
		expect(row.available).toBe(false);
	}
	const oauth = writerReadiness(
		selectors,
		registry,
		credentialsOf({ oauth: ["anthropic", "google-antigravity"] }),
		{},
	);
	expect(oauth.every((row) => row.available)).toBe(true);
});

test("coding-plan ZAI_API_KEY makes only the Zhipu coding-plan route eligible", () => {
	const zhipu = "zhipu-coding-plan/glm-5.3-flash";
	const kimi = "kimi-code/k3";
	const registry = registryOf(zhipu, kimi);
	const none = credentialsOf({});
	const [withoutKey] = writerReadiness([zhipu], registry, none, {});
	expect(withoutKey.available).toBe(false);
	const withKey = writerReadiness([zhipu, kimi], registry, none, {
		ZAI_API_KEY: "coding-plan-key",
	});
	expect(withKey.map((row) => [row.selector, row.available])).toEqual([
		[zhipu, true],
		[kimi, false],
	]);
});

test("writer readiness refuses inexact or duplicate selectors", () => {
	const registry = registryOf("kimi-code/k3");
	const credentials = credentialsOf({});
	for (const selectors of [
		[],
		["kimi-code"],
		["kimi-code/k3/extra"],
		["kimi-code/k3", "kimi-code/k3"],
	])
		expect(() => writerReadiness(selectors, registry, credentials, {})).toThrow();
});

afterEach(async () => {
	await Promise.all(
		temporary.splice(0).map((path) => rm(path, { recursive: true })),
	);
});
async function snapshot() {
	const parent = await realpath(await mkdtemp(join(tmpdir(), "omp-policy-")));
	temporary.push(parent);
	const root = join(parent, "snapshot");
	await mkdir(root);
	await writeFile(join(root, "source.txt"), "first\nneedle\nlast\n");
	await writeFile(join(parent, "secret.txt"), "outside snapshot");
	await symlink(join(parent, "secret.txt"), join(root, "escape"));
	return root;
}

test("Antigravity refuses key-only credentials and OAuth refresh failure", async () => {
	expect(() => requireAntigravityOAuth({ hasOAuth: () => false })).toThrow(
		"requires stored OAuth",
	);
	expect(() => requireAntigravityOAuth({ hasOAuth: () => true })).not.toThrow();
	await expect(antigravityOAuthCredential(async () => null)).rejects.toThrow(
		"API-key fallback prohibited",
	);
	await expect(
		antigravityOAuthCredential(async () => { throw new Error("refresh failed"); }),
	).rejects.toThrow("refresh failed");
	await expect(
		antigravityOAuthCredential(async () => ({ accessToken: "oauth-token" })),
	).rejects.toThrow("names no Cloud project");
	expect(
		JSON.parse(
			await antigravityOAuthCredential(async () => ({
				accessToken: "oauth-token",
				projectId: "project-1",
			})),
		),
	).toEqual({ token: "oauth-token", projectId: "project-1" });
});

test("URI devices, parent traversal and symlinks cannot escape the snapshot", async () => {
	const root = await snapshot();
	expect(await confinedPath(root, "source.txt")).toBe(join(root, "source.txt"));
	for (const path of [
		"../secret.txt",
		"escape",
		"xd://bash",
		"https://example.org/source",
		"source.txt\0",
	]) {
		await expect(confinedPath(root, path)).rejects.toThrow();
	}
});

test("read, glob and literal search expose served paths but neither execute nor write", async () => {
	const root = await snapshot();
	// Same pinned deployed SDK as execution, no auth discovery and no model call.
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const tools = readTools(sdk.z, root);
	const read = tools.find((tool) => tool.name === "execution_read")!;
	const result = await read.execute("read-1", {
		path: "source.txt",
		offset: 2,
		limit: 1,
	});
	expect(result.content[0].text).toContain("2: needle");
	expect(result.content[0].text).not.toContain("1: first");
	expect(result.details.paths).toEqual([join(root, "source.txt")]);
	const glob = tools.find((tool) => tool.name === "execution_glob")!;
	const listed = await glob.execute("glob-1", { path: ".", pattern: "**/*" });
	expect(listed.details.paths).toEqual([join(root, "source.txt")]);
	const grep = tools.find((tool) => tool.name === "execution_grep")!;
	const searched = await grep.execute("grep-1", {
		path: ".",
		pattern: "needle",
	});
	expect(searched.content[0].text).toContain(":2: needle");
	expect(searched.details.paths).toEqual([join(root, "source.txt")]);
	await expect(read.execute("escape-1", { path: "escape" })).rejects.toThrow();
	expect(await readFile(join(root, "source.txt"), "utf8")).toBe(
		"first\nneedle\nlast\n",
	);
});

test("ranged reads and literal search inspect large sources", async () => {
	const root = await snapshot();
	const source = "padding\n".repeat(70_000) + "large-source-marker\n";
	await writeFile(join(root, "large.txt"), source);
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	// The SDK location is selected by the runtime environment.
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const tools = readTools(sdk.z, root);
	const read = tools.find((tool) => tool.name === "execution_read")!;
	const ranged = await read.execute("large-read", {
		path: "large.txt",
		offset: 70_001,
		limit: 1,
	});
	expect(ranged.content[0].text).toContain("70001: large-source-marker");
	expect(ranged.content[0].text).not.toContain("padding");
	const grep = tools.find((tool) => tool.name === "execution_grep")!;
	const searched = await grep.execute("large-search", {
		path: "large.txt",
		pattern: "large-source-marker",
	});
	expect(searched.content[0].text).toContain(":70001: large-source-marker");
	expect(searched.details.paths).toEqual([join(root, "large.txt")]);
});

test("read and search bound emitted UTF-8 bytes even for one long line", async () => {
	const root = await snapshot();
	await writeFile(join(root, "long.txt"), "漢".repeat(200_000));
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	// The SDK location is selected by the runtime environment.
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const tools = readTools(sdk.z, root);
	const read = tools.find((tool) => tool.name === "execution_read")!;
	await expect(
		read.execute("long-read", {
			path: "long.txt",
			offset: 1,
			limit: 1,
		}),
	).rejects.toThrow();
	const grep = tools.find((tool) => tool.name === "execution_grep")!;
	await expect(
		grep.execute("long-search", { path: "long.txt", pattern: "漢" }),
	).rejects.toThrow();
});

test("workspace-write tools change snapshot bytes and refuse ambiguous edits", async () => {
	const root = await snapshot();
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const tools = writeTools(sdk.z, root);
	const write = tools.find((tool) => tool.name === "execution_write")!;
	const edit = tools.find((tool) => tool.name === "execution_edit")!;
	const written = await write.execute("write-1", {
		path: "created.txt",
		content: "alpha\n",
	});
	expect(written.details.paths).toEqual([join(root, "created.txt")]);
	expect(await readFile(join(root, "created.txt"), "utf8")).toBe("alpha\n");
	await edit.execute("edit-1", {
		path: "created.txt",
		old_text: "alpha",
		new_text: "beta",
	});
	expect(await readFile(join(root, "created.txt"), "utf8")).toBe("beta\n");
	await write.execute("write-2", {
		path: "source.txt",
		content: "same\nsame\n",
	});
	await expect(
		edit.execute("ambiguous", {
			path: "source.txt",
			old_text: "same",
			new_text: "changed",
		}),
	).rejects.toThrow("ambiguous");
});

test("workspace-write tools refuse escapes symlinks and metadata", async () => {
	const root = await snapshot();
	const metadataPaths = [".omp/config", ".OmP/config", ".GiT/config"];
	for (const path of metadataPaths) {
		await mkdir(join(root, path.split("/")[0]), { recursive: true });
		await writeFile(join(root, path), "metadata\n");
	}
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const tools = writeTools(sdk.z, root);
	const write = tools.find((tool) => tool.name === "execution_write")!;
	const edit = tools.find((tool) => tool.name === "execution_edit")!;
	for (const path of [
		"../secret.txt",
		"escape",
		".git/config",
		".omp/config",
		".OmP/config",
		".GiT/config",
		"./.omp/config",
		".omp/../source.txt",
		"https://example.org/source",
		"/tmp/outside",
		"source.txt\0",
	]) {
		await expect(
			write.execute(`blocked-${path}`, { path, content: "changed\n" }),
		).rejects.toThrow();
	}
	for (const path of metadataPaths) {
		await expect(
			edit.execute(`blocked-edit-${path}`, {
				path,
				old_text: "metadata",
				new_text: "changed",
			}),
		).rejects.toThrow();
		expect(await readFile(join(root, path), "utf8")).toBe("metadata\n");
	}
	expect(await readFile(join(root, "source.txt"), "utf8")).toBe(
		"first\nneedle\nlast\n",
	);
	expect(await readFile(join(root, ".omp", "config"), "utf8")).toBe(
		"metadata\n",
	);
});

test("a writer cannot overwrite another call's authoritative result", async () => {
	const root = await snapshot();
	const evidence = [
		".agent-execution/results/other-call.json",
		".Agent-Execution/results/other-call.json",
		"outputs/agent-execution-worker-result-other-call.json",
		"outputs/AGENT-EXECUTION-WORKER-RESULT.json",
	];
	for (const path of evidence) {
		await mkdir(join(root, path.substring(0, path.lastIndexOf("/"))), {
			recursive: true,
		});
		await writeFile(
			join(root, path),
			'{"model_call_id":"other-call","status":"completed"}',
		);
	}
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	// SDK location is selected by the runtime environment, not a project dependency.
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const tools = writeTools(sdk.z, root);
	const write = tools.find((tool) => tool.name === "execution_write")!;
	const edit = tools.find((tool) => tool.name === "execution_edit")!;
	for (const path of evidence) {
		const before = await readFile(join(root, path), "utf8");
		await expect(
			write.execute("writer-overwrite", {
				path,
				content: '{"status":"completed","forged":true}',
			}),
		).rejects.toThrow();
		await expect(
			edit.execute("writer-edit", {
				path,
				old_text: "completed",
				new_text: "forged",
			}),
		).rejects.toThrow();
		expect(await readFile(join(root, path), "utf8")).toBe(before);
	}
	await write.execute("ordinary-output", {
		path: "outputs/report.json",
		content: '{"report":true}',
	});
	expect(await readFile(join(root, "outputs/report.json"), "utf8")).toBe(
		'{"report":true}',
	);
});

test("workspace-write replacement does not mutate outside hardlink targets", async () => {
	const parent = await realpath(await mkdtemp(join(tmpdir(), "omp-policy-")));
	temporary.push(parent);
	const root = join(parent, "snapshot");
	await mkdir(root);
	await writeFile(join(parent, "shared.txt"), "outside\n");
	await link(join(parent, "shared.txt"), join(root, "linked.txt"));
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const write = writeTools(sdk.z, root).find(
		(tool) => tool.name === "execution_write",
	)!;
	await write.execute("replace-hardlink", {
		path: "linked.txt",
		content: "inside\n",
	});
	expect(await readFile(join(root, "linked.txt"), "utf8")).toBe("inside\n");
	expect(await readFile(join(parent, "shared.txt"), "utf8")).toBe("outside\n");
});

test("workspace-write replacement preserves existing executable mode", async () => {
	const root = await snapshot();
	const script = join(root, "run.sh");
	await writeFile(script, "#!/bin/sh\necho old\n");
	await chmod(script, 0o755);
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const tools = writeTools(sdk.z, root);
	const write = tools.find((tool) => tool.name === "execution_write")!;
	const edit = tools.find((tool) => tool.name === "execution_edit")!;
	const beforeMode = (await stat(script)).mode & 0o777;

	await write.execute("replace-executable", {
		path: "run.sh",
		content: "#!/bin/sh\necho new\n",
	});
	expect((await stat(script)).mode & 0o777).toBe(beforeMode);
	await edit.execute("edit-executable", {
		path: "run.sh",
		old_text: "new",
		new_text: "newer",
	});
	expect(await readFile(script, "utf8")).toBe("#!/bin/sh\necho newer\n");
	expect((await stat(script)).mode & 0o777).toBe(beforeMode);
});

test("workspace-write edits and writes UTF-8 files through the 2 MiB byte limit", async () => {
	const root = await snapshot();
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	// The SDK root is runtime-selected, so a static package import cannot select it.
	const sdk = await import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
	const tools = writeTools(sdk.z, root);
	const write = tools.find((tool) => tool.name === "execution_write")!;
	const edit = tools.find((tool) => tool.name === "execution_edit")!;

	const largeSource = "padding\n".repeat(70_000) + "unique-marker\n";
	await writeFile(join(root, "large-edit.txt"), largeSource);
	await edit.execute("large-splice", {
		path: "large-edit.txt",
		old_text: "unique-marker",
		new_text: "edited-marker",
	});
	const edited = await readFile(join(root, "large-edit.txt"));
	expect(edited).toEqual(Buffer.from(largeSource.replace("unique-marker", "edited-marker")));

	const exactLimit = "é".repeat(1024 * 1024);
	await write.execute("write-exact-limit", {
		path: "boundary.txt",
		content: exactLimit,
	});
	expect(await readFile(join(root, "boundary.txt"))).toEqual(Buffer.from(exactLimit));

	const oversized = exactLimit + "a";
	await expect(
		write.execute("write-over-limit", {
			path: "boundary.txt",
			content: oversized,
		}),
	).rejects.toThrow();
	expect(await readFile(join(root, "boundary.txt"))).toEqual(Buffer.from(exactLimit));
});
