import { afterEach, expect, test } from "bun:test";
import {
	chmod,
	link,
	lstat,
	mkdtemp,
	mkdir,
	readFile,
	readlink,
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

/** The pinned deployed SDK module that supplies zod for the tool schemas. */
async function sdkModule() {
	const sdkRoot =
		process.env.AGENT_EXECUTION_OMP_SDK_ROOT ??
		join(homedir(), ".local/share/agent-execution/omp-sdk", SDK_VERSION);
	return import(
		pathToFileURL(
			join(sdkRoot, "node_modules/@oh-my-pi/pi-coding-agent/src/index.ts"),
		).href
	);
}

/**
 * A snapshot seeded with the ordinary relative in-root links Offload carries
 * (file, directory, internal parent step, chain) beside every refused shape.
 */
async function linkedSnapshot() {
	const root = await snapshot();
	await mkdir(join(root, "realdocs"));
	await mkdir(join(root, "sub"));
	await mkdir(join(root, ".git"));
	await writeFile(join(root, "realdocs", "guide.txt"), "guide\nneedle-doc\n");
	await writeFile(join(root, ".git", "config"), "[core]\n");
	// Ordinary relative links: file, directory, parent step, chain, mid-target `..`.
	await symlink("source.txt", join(root, "alias.txt"));
	await symlink("realdocs", join(root, "docs"));
	await symlink("../source.txt", join(root, "sub", "up.txt"));
	await symlink("second.txt", join(root, "first.txt"));
	await symlink("alias.txt", join(root, "second.txt"));
	await symlink("realdocs/../source.txt", join(root, "zig.txt"));
	// Refused shapes.
	await symlink(join(root, "source.txt"), join(root, "abs-into-root.txt"));
	await symlink("../secret.txt", join(root, "rel-escape.txt"));
	await symlink("sub/../../secret.txt", join(root, "indirect-escape.txt"));
	await symlink("..", join(root, "parent-dir"));
	await symlink(".git/config", join(root, "meta-link.txt"));
	await symlink(".git", join(root, "meta-dir"));
	await symlink("cycle-b.txt", join(root, "cycle-a.txt"));
	await symlink("cycle-a.txt", join(root, "cycle-b.txt"));
	await symlink("self.txt", join(root, "self.txt"));
	await symlink("missing.txt", join(root, "ghost.txt"));
	await symlink(".", join(root, "loop"));
	return root;
}

test("confined reads resolve relative in-root file, directory, parent-step and chained links", async () => {
	const root = await linkedSnapshot();
	const sdk = await sdkModule();
	const tools = readTools(sdk.z, root);
	const read = tools.find((tool) => tool.name === "execution_read")!;

	const fileLink = await read.execute("read-file-link", { path: "alias.txt" });
	expect(fileLink.content[0].text).toContain("2: needle");
	expect(fileLink.details.paths).toEqual([join(root, "source.txt")]);

	const dirLink = await read.execute("read-dir-link", {
		path: "docs/guide.txt",
	});
	expect(dirLink.content[0].text).toContain("2: needle-doc");
	expect(dirLink.details.paths).toEqual([join(root, "realdocs", "guide.txt")]);

	const parentStep = await read.execute("read-parent-step", {
		path: "sub/up.txt",
	});
	expect(parentStep.content[0].text).toContain("2: needle");
	expect(parentStep.details.paths).toEqual([join(root, "source.txt")]);

	const chain = await read.execute("read-chain", { path: "first.txt" });
	expect(chain.content[0].text).toContain("2: needle");
	expect(chain.details.paths).toEqual([join(root, "source.txt")]);

	const midDotDot = await read.execute("read-mid-dotdot", { path: "zig.txt" });
	expect(midDotDot.content[0].text).toContain("2: needle");
	expect(midDotDot.details.paths).toEqual([join(root, "source.txt")]);

	const glob = tools.find((tool) => tool.name === "execution_glob")!;
	const listed = await glob.execute("glob-linked-base", {
		path: "docs",
		pattern: "*.txt",
	});
	expect(listed.details.paths).toEqual([join(root, "realdocs", "guide.txt")]);

	const grep = tools.find((tool) => tool.name === "execution_grep")!;
	const file = await grep.execute("grep-linked-file", {
		path: "alias.txt",
		pattern: "needle",
	});
	expect(file.content[0].text).toContain(":2: needle");
	expect(file.details.paths).toEqual([join(root, "source.txt")]);
	const directory = await grep.execute("grep-linked-dir", {
		path: "docs",
		pattern: "needle-doc",
	});
	expect(directory.content[0].text).toContain("guide.txt:2: needle-doc");
	expect(directory.details.paths).toEqual([join(root, "realdocs", "guide.txt")]);
});

test("confined reads preserve POSIX symlink component and directory semantics", async () => {
	const root = await linkedSnapshot();
	const sdk = await sdkModule();
	const read = readTools(sdk.z, root).find(
		(tool) => tool.name === "execution_read",
	)!;

	// There is a lookalike slash path, but POSIX readlink's backslash is literal.
	await mkdir(join(root, "data"));
	await writeFile(join(root, "data", "info.txt"), "wrong-file-marker\n");
	await symlink("data\\info.txt", join(root, "backslash-target"));
	await expect(
		read.execute("blocked-backslash-target", { path: "backslash-target" }),
	).rejects.toThrow();

	// POSIX file/. and file/ both require the target to be a directory.
	await writeFile(join(root, "agent-launcher"), "not a directory\n");
	await symlink("agent-launcher/", join(root, "trailing-file-target"));
	await symlink("agent-launcher/.", join(root, "dot-file-target"));
	for (const path of ["trailing-file-target", "dot-file-target"])
		await expect(read.execute(`blocked-${path}`, { path })).rejects.toThrow();

	// Internal dot/trailing separators on a directory are valid; a chain to it
	// remains resolvable rather than being rejected wholesale.
	await symlink("realdocs/./", join(root, "dot-dir-target"));
	await symlink("dot-dir-target", join(root, "dot-dir-chain"));
	const valid = await read.execute("valid-dot-directory-chain", {
		path: "dot-dir-chain/guide.txt",
	});
	expect(valid.content[0].text).toContain("2: needle-doc");
	expect(valid.details.paths).toEqual([join(root, "realdocs", "guide.txt")]);
});

test("confined reads refuse absolute, escaping, indirect, protected, cyclic and dangling links", async () => {
	const root = await linkedSnapshot();
	const sdk = await sdkModule();
	const read = readTools(sdk.z, root).find(
		(tool) => tool.name === "execution_read",
	)!;
	for (const path of [
		"abs-into-root.txt", // absolute target even though it points inside the root
		"rel-escape.txt", // relative `..` escape
		"indirect-escape.txt", // parent steps that leave the root
		"parent-dir/secret.txt", // directory-link escape through a child path
		"meta-link.txt", // link into VCS metadata
		"meta-dir/config", // directory link into VCS metadata
		"cycle-a.txt", // two-link cycle
		"self.txt", // self cycle
		"ghost.txt", // dangling link
		"docs/../../../secret.txt", // user-supplied traversal via a linked prefix
	]) {
		await expect(read.execute(`blocked-read-${path}`, { path })).rejects.toThrow();
	}
	await expect(
		read.execute("blocked-user-traversal", { path: "docs/../source.txt" }),
	).rejects.toThrow();
});

test("writes and edits refuse safe aliases and leave link identity and target bytes intact", async () => {
	const root = await linkedSnapshot();
	const sdk = await sdkModule();
	const tools = writeTools(sdk.z, root);
	const write = tools.find((tool) => tool.name === "execution_write")!;
	const edit = tools.find((tool) => tool.name === "execution_edit")!;

	const targetBefore = await readFile(join(root, "source.txt"), "utf8");
	const guideBefore = await readFile(join(root, "realdocs", "guide.txt"), "utf8");
	for (const path of ["alias.txt", "docs/guide.txt", "sub/up.txt", "first.txt"]) {
		await expect(
			write.execute(`blocked-alias-write-${path}`, {
				path,
				content: "forged\n",
			}),
		).rejects.toThrow();
		await expect(
			edit.execute(`blocked-alias-edit-${path}`, {
				path,
				old_text: "needle",
				new_text: "forged",
			}),
		).rejects.toThrow();
	}
	// Creating a new file through a linked directory parent is equally refused.
	await expect(
		write.execute("blocked-alias-create", {
			path: "docs/created.txt",
			content: "created\n",
		}),
	).rejects.toThrow();

	const expectedTargets: Record<string, string> = {
		"alias.txt": "source.txt",
		docs: "realdocs",
		"sub/up.txt": "../source.txt",
		"first.txt": "second.txt",
		"second.txt": "alias.txt",
	};
	for (const [link, target] of Object.entries(expectedTargets)) {
		const path = join(root, ...link.split("/"));
		expect((await lstat(path)).isSymbolicLink()).toBe(true);
		expect(await readlink(path)).toBe(target);
	}
	expect(await readFile(join(root, "source.txt"), "utf8")).toBe(targetBefore);
	expect(await readFile(join(root, "realdocs", "guide.txt"), "utf8")).toBe(
		guideBefore,
	);
	expect(await readFile(join(root, "alias.txt"), "utf8")).toBe(targetBefore);

	// Ordinary targets addressed canonically remain editable beside the alias.
	await edit.execute("canonical-edit", {
		path: "source.txt",
		old_text: "needle",
		new_text: "marker",
	});
	expect(await readFile(join(root, "source.txt"), "utf8")).toBe(
		"first\nmarker\nlast\n",
	);
	expect((await lstat(join(root, "alias.txt"))).isSymbolicLink()).toBe(true);
});

test("implicit glob and search traversal completes without following directory link loops", async () => {
	const root = await linkedSnapshot();
	const sdk = await sdkModule();
	const tools = readTools(sdk.z, root);
	const glob = tools.find((tool) => tool.name === "execution_glob")!;
	const grep = tools.find((tool) => tool.name === "execution_grep")!;

	const listed = await glob.execute("glob-loop", { path: ".", pattern: "**/*" });
	expect(listed.details.paths).toEqual([
		join(root, "realdocs", "guide.txt"),
		join(root, "source.txt"),
	]);

	const searched = await grep.execute("grep-loop", {
		path: ".",
		pattern: "needle",
	});
	expect(searched.details.paths).toEqual([
		join(root, "realdocs", "guide.txt"),
		join(root, "source.txt"),
	]);
	expect(searched.content[0].text).toContain("guide.txt:2: needle-doc");
	expect(searched.content[0].text).toContain("source.txt:2: needle");
});
