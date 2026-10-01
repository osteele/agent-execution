/** Restricted OMP SDK execution. Never imports configuration or tools from the reviewed tree. */
import { createHash, randomUUID } from "node:crypto";
import {
	chmod,
	lstat,
	open,
	readFile,
	realpath,
	readdir,
	rename,
	rm,
	stat,
} from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import type * as OmpSdk from "@oh-my-pi/pi-coding-agent";

export const SDK_VERSION = "18.4.4";
export const READ_TOOLS = [
	"execution_read",
	"execution_glob",
	"execution_grep",
];
export const WRITE_TOOLS = [...READ_TOOLS, "execution_write", "execution_edit"];
export const TOOLS = READ_TOOLS;
const MAX_BYTES = 512 * 1024;
// A source file can be larger than its bounded read or search result.
const MAX_READ_BYTES = 4 * 1024 * 1024;
const FORBIDDEN_METADATA = new Set([
	".git",
	".jj",
	".hg",
	".svn",
	".omp",
	".agent-execution",
	".agent-review",
	".agents",
	".codex",
	".claude",
	".gemini",
]);

class UnsupportedTextFile extends Error {}

function writeOutput(line: string): Promise<void> {
	const { promise, resolve, reject } = Promise.withResolvers<void>();
	process.stdout.write(line, (error) => (error ? reject(error) : resolve()));
	return promise;
}

function rejectUnsafePath(value: string, forbidMetadata: boolean): void {
	if (
		!value ||
		value.includes("\0") ||
		isAbsolute(value) ||
		/^[a-z][a-z0-9+.-]*:\/\//i.test(value)
	) {
		throw new Error(
			"Only relative filesystem paths inside the execution snapshot are permitted",
		);
	}
	for (const part of value.split(/[\\/]+/)) {
		if (part === "..") throw new Error("Path traversal is not permitted");
		if (forbidMetadata && FORBIDDEN_METADATA.has(part.toLowerCase()))
			throw new Error("VCS and harness metadata paths are not editable");
		if (
			forbidMetadata &&
			/^agent-execution-worker-result(?:-.*)?\.json$/i.test(part)
		)
			throw new Error("Legacy worker evidence paths are not editable");
	}
}

async function safeSnapshotPath(
	root: string,
	value: string,
	{
		mustExist,
		forbidMetadata = false,
	}: { mustExist: boolean; forbidMetadata?: boolean },
): Promise<string> {
	rejectUnsafePath(value, forbidMetadata);
	const rootReal = await realpath(root);
	const candidate = resolve(rootReal, value);
	const rel = relative(rootReal, candidate);
	if (rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel)) {
		throw new Error("Path escapes the execution snapshot");
	}
	const parts = rel ? rel.split(sep) : [];
	let cursor = rootReal;
	for (let index = 0; index < parts.length; index++) {
		cursor = join(cursor, parts[index]);
		try {
			const info = await lstat(cursor);
			if (info.isSymbolicLink())
				throw new Error("Symlink paths are not permitted");
			if (index < parts.length - 1 && !info.isDirectory())
				throw new Error("Path parent is not a directory");
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
			if (mustExist || index < parts.length - 1)
				throw new Error("Path parent does not exist safely");
		}
	}
	return candidate;
}

export async function confinedPath(
	root: string,
	value: string,
): Promise<string> {
	return safeSnapshotPath(root, value, { mustExist: true });
}

async function textFile(
	root: string,
	value: string,
): Promise<{ path: string; text: string }> {
	const path = await confinedPath(root, value);
	const info = await stat(path);
	if (!info.isFile() || info.size > MAX_READ_BYTES)
		throw new UnsupportedTextFile(
			"Read requires a regular file of at most 4 MiB",
		);
	const bytes = await readFile(path);
	if (bytes.length > MAX_READ_BYTES || bytes.includes(0))
		throw new UnsupportedTextFile("Binary or oversized file is not readable");
	try {
		return {
			path,
			text: new TextDecoder("utf-8", { fatal: true }).decode(bytes),
		};
	} catch (error) {
		if (!(error instanceof TypeError)) throw error;
		throw new UnsupportedTextFile("File does not contain valid UTF-8");
	}
}

async function files(root: string, base: string): Promise<string[]> {
	const start = await confinedPath(root, base);
	const found: string[] = [];
	const pending = [start];
	while (pending.length) {
		const directory = pending.pop()!;
		for (const entry of await readdir(directory, { withFileTypes: true })) {
			// Never follow directory symlinks or read a symlink as an implicit grep input.
			if (
				entry.isSymbolicLink() ||
				entry.name === ".git" ||
				entry.name === ".jj"
			)
				continue;
			const path = join(directory, entry.name);
			if (entry.isDirectory()) pending.push(path);
			else if (entry.isFile()) found.push(path);
			if (found.length + pending.length > 20000)
				throw new Error("Search exceeds 20000 snapshot entries; narrow path");
		}
	}
	return found.sort();
}

function checkResultSize(bytes: number): void {
	if (bytes > MAX_BYTES)
		throw new Error(
			"Tool result exceeds 512 KiB; narrow the path, pattern, or line range",
		);
}

function response(text: string, paths: string[]) {
	checkResultSize(Buffer.byteLength(text, "utf8"));
	return {
		content: [{ type: "text" as const, text }],
		details: { paths },
	};
}

async function assertSafeParent(root: string, path: string): Promise<void> {
	const rootReal = await realpath(root);
	const parent = dirname(path);
	const parentInfo = await lstat(parent);
	if (parentInfo.isSymbolicLink() || !parentInfo.isDirectory())
		throw new Error("Write parent is not a safe directory");
	const parentReal = await realpath(parent);
	const rel = relative(rootReal, parentReal);
	if (rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel))
		throw new Error("Write parent escapes the execution snapshot");
}

async function atomicTextReplace(
	root: string,
	path: string,
	text: string,
	mode: number = 0o600,
): Promise<void> {
	const encoded = new TextEncoder().encode(text);
	if (encoded.length > MAX_BYTES) throw new Error("Write exceeds 512 KiB");
	const permissions = mode & 0o777;
	await assertSafeParent(root, path);
	const temporary = join(
		dirname(path),
		`.agent-execution-write-${process.pid}-${randomUUID()}`,
	);
	let handle;
	try {
		handle = await open(temporary, "wx", permissions);
		await handle.writeFile(encoded);
		await handle.close();
		handle = undefined;
		await chmod(temporary, permissions);
		await assertSafeParent(root, path);
		await rename(temporary, path);
	} finally {
		if (handle) await handle.close();
		await rm(temporary, { force: true });
	}
}

export function writeTools(z: typeof OmpSdk.z, root: string) {
	return [
		...readTools(z, root),
		{
			name: "execution_write" as const,
			label: "Write snapshot",
			description:
				"Replace or create one UTF-8 file inside the execution snapshot. Relative paths only; symlinks, traversal, URLs and VCS metadata are refused.",
			parameters: z.object({
				path: z.string(),
				content: z.string(),
			}),
			async execute(_id: string, args: { path: string; content: string }) {
				const path = await safeSnapshotPath(root, args.path, {
					mustExist: false,
					forbidMetadata: true,
				});
				let before = "";
				let mode = 0o600;
				try {
					const info = await lstat(path);
					if (info.isSymbolicLink() || !info.isFile())
						throw new Error("Write target must be a plain file");
					mode = info.mode;
					before = await readFile(path, "utf8");
				} catch (error) {
					if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
				}
				if (before === args.content)
					throw new Error("Write would not change file bytes");
				await atomicTextReplace(root, path, args.content, mode);
				return response(`wrote ${path}`, [path]);
			},
		},
		{
			name: "execution_edit" as const,
			label: "Edit snapshot",
			description:
				"Replace exactly one matching UTF-8 span inside an existing snapshot file. Ambiguous, missing, no-op, symlink and metadata edits are refused.",
			parameters: z.object({
				path: z.string(),
				old_text: z.string().min(1),
				new_text: z.string(),
			}),
			async execute(
				_id: string,
				args: { path: string; old_text: string; new_text: string },
			) {
				const path = await safeSnapshotPath(root, args.path, {
					mustExist: true,
					forbidMetadata: true,
				});
				const info = await lstat(path);
				if (info.isSymbolicLink() || !info.isFile())
					throw new Error("Edit target must be a plain file");
				const before = await readFile(path, "utf8");
				const first = before.indexOf(args.old_text);
				if (first === -1) throw new Error("Edit match was not found");
				if (before.indexOf(args.old_text, first + args.old_text.length) !== -1)
					throw new Error("Edit match is ambiguous");
				const after =
					before.slice(0, first) +
					args.new_text +
					before.slice(first + args.old_text.length);
				if (after === before)
					throw new Error("Edit would not change file bytes");
				await atomicTextReplace(root, path, after, info.mode);
				return response(`edited ${path}`, [path]);
			},
		},
	];
}

export function readTools(z: typeof OmpSdk.z, root: string) {
	return [
		{
			name: "execution_read" as const,
			label: "Read snapshot",
			description:
				"Read a UTF-8 file of at most 4 MiB inside the execution snapshot. Results are limited to 512 KiB. No URLs, internal devices, shell, or document converters. Line offsets start at 1.",
			parameters: z.object({
				path: z.string(),
				offset: z.number().int().min(1).optional(),
				limit: z.number().int().min(1).max(2000).optional(),
			}),
			async execute(
				_id: string,
				args: { path: string; offset?: number; limit?: number },
			) {
				const file = await textFile(root, args.path);
				const offset = args.offset ?? 1;
				const lines = file.text.split("\n");
				const selected = lines.slice(
					offset - 1,
					offset - 1 + (args.limit ?? 300),
				);
				return response(
					`${file.path}\n${selected.map((line, i) => `${offset + i}: ${line}`).join("\n")}\n[${lines.length} total lines]`,
					[file.path],
				);
			},
		},
		{
			name: "execution_glob" as const,
			label: "List snapshot",
			description:
				"Find snapshot files by glob (relative to path, default snapshot root). Symlinks are not followed.",
			parameters: z.object({
				pattern: z.string(),
				path: z.string().optional(),
			}),
			async execute(_id: string, args: { pattern: string; path?: string }) {
				const base = await confinedPath(root, args.path ?? ".");
				const matcher = new Bun.Glob(args.pattern);
				const matches = (await files(root, args.path ?? ".")).filter((path) =>
					matcher.match(relative(base, path)),
				);
				if (matches.length > 2000)
					throw new Error("More than 2000 matches; narrow pattern");
				return response(matches.join("\n"), matches);
			},
		},
		{
			name: "execution_grep" as const,
			label: "Search snapshot",
			description:
				"Search UTF-8 snapshot files for a literal string (not a regex). File path or directory path is required. Symlinks, binary and files over 4 MiB are not scanned. Results are limited to 512 KiB.",
			parameters: z.object({ pattern: z.string().min(1), path: z.string() }),
			async execute(_id: string, args: { pattern: string; path: string }) {
				const base = await confinedPath(root, args.path);
				const inputs = (await stat(base)).isFile()
					? [base]
					: await files(root, args.path);
				const matches: string[] = [];
				const readPaths: string[] = [];
				const skipped: string[] = [];
				let resultBytes = 0;
				const appendResult = (destination: string[], line: string) => {
					resultBytes += Buffer.byteLength(line, "utf8") + 1;
					checkResultSize(resultBytes);
					destination.push(line);
				};
				for (const path of inputs) {
					let file;
					try {
						file = await textFile(root, relative(root, path));
					} catch (error) {
						if (!(error instanceof UnsupportedTextFile)) throw error;
						appendResult(skipped, `${path}: ${error.message}`);
						continue;
					}
					readPaths.push(path);
					for (const [index, line] of file.text.split("\n").entries()) {
						if (line.includes(args.pattern))
							appendResult(matches, `${path}:${index + 1}: ${line}`);
						if (matches.length > 1000)
							throw new Error("More than 1000 matching lines; narrow search");
					}
				}
				return response(
					matches.join("\n") +
						(skipped.length
							? `\nSkipped unreadable/binary/oversized files:\n${skipped.join("\n")}`
							: ""),
					readPaths,
				);
			},
		},
	];
}

async function main(): Promise<void> {
	const [sdkRoot, selector, policy, systemPrompt, snapshotCwd] =
		process.argv.slice(2);
	if (
		!sdkRoot ||
		!selector ||
		![
			"read-only-no-shell",
			"packet-only-no-tools",
			"workspace-write-no-shell",
			"--auth-status",
		].includes(policy)
	) {
		throw new Error("Invalid restricted OMP invocation");
	}
	const packageRoot = join(
		sdkRoot,
		"node_modules",
		"@oh-my-pi",
		"pi-coding-agent",
	);
	const metadata: unknown = JSON.parse(
		await readFile(join(packageRoot, "package.json"), "utf8"),
	);
	if (
		!metadata ||
		typeof metadata !== "object" ||
		!("version" in metadata) ||
		metadata.version !== SDK_VERSION
	) {
		throw new Error(`Restricted OMP requires SDK ${SDK_VERSION}`);
	}
	// Only this deployment-owned package is executable. Never resolve from cwd.
	const sdk: typeof OmpSdk = await import(
		pathToFileURL(join(packageRoot, "src", "index.ts")).href
	);
	if (!snapshotCwd || !isAbsolute(snapshotCwd))
		throw new Error("OMP snapshot cwd is missing");
	const cwd = await realpath(snapshotCwd);
	process.chdir(cwd);
	const [provider, modelId, extra] = selector.split("/");
	if (!provider || !modelId || extra) {
		throw new Error("Restricted OMP requires an exact provider/model selector");
	}
	const authStorage = await sdk.discoverAuthStorage();
	try {
		if (policy === "--auth-status") {
			const credentialType = authStorage.credentials.hasOAuth(provider)
				? "oauth"
				: authStorage.credentials.get(provider)?.type === "api_key"
					? "api_key"
					: authStorage.credentials.has(provider)
						? "unknown"
						: "missing";
			await writeOutput(
				`${JSON.stringify({
					schema_version: "agent-execution.omp-auth/v1",
					provider,
					model: modelId,
					credential_type: credentialType,
				})}\n`,
			);
			return;
		}
		if (provider === "anthropic" && !authStorage.credentials.hasOAuth(provider)) {
			throw new Error(
				"Restricted OMP Anthropic execution requires stored OAuth credentials",
			);
		}
		const prompt = await Bun.stdin.text();
		const settings = sdk.Settings.isolated({
			"advisor.enabled": false,
			"memory.backend": "off",
			"memories.enabled": false,
			"autolearn.enabled": false,
			"compaction.enabled": false,
			"branchSummary.enabled": false,
			"retry.modelFallback": false,
			"retry.usageAwareFallback": false,
			"retry.fallbackChains": {},
			"plan.enabled": false,
			"goal.enabled": false,
			"tools.xdev": false,
			externalThinking: false,
			includeWorkspaceTree: false,
			"secrets.enabled": false,
			"startup.checkUpdate": false,
			"task.maxRecursionDepth": 0,
			enabledModels: [selector],
		});
		const registry = new sdk.ModelRegistry(authStorage, undefined, {
			settings,
			ignoreLocalModelConfig: true,
		});
		const model = registry.find(provider, modelId);
		if (!model || model.provider !== provider || model.id !== modelId)
			throw new Error(`Unavailable exact OMP model: ${selector}`);
		const allowed =
			policy === "read-only-no-shell"
				? READ_TOOLS
				: policy === "workspace-write-no-shell"
					? WRITE_TOOLS
					: [];
		const manager = sdk.SessionManager.inMemory(cwd);
		const { session } = await sdk.createAgentSession({
			cwd,
			settings,
			authStorage,
			modelRegistry: registry,
			model,
			sessionManager: manager,
			// Effort.High; the SDK is imported for types only, so its const enum
			// value is written as the string it stands for.
			thinkingLevel: "high" as NonNullable<
				OmpSdk.CreateAgentSessionOptions["thinkingLevel"]
			>,
			autoApprove: true,
			getApiKey: (requestModel) => {
				if (requestModel.provider !== provider || requestModel.id !== modelId) {
					throw new Error(
						"OMP attempted an inference provider/model substitution",
					);
				}
				if (provider === "anthropic") {
					return async () => {
						// Public OAuth-only resolution refreshes stored tokens but cannot fall
						// through to API keys if the credential disappears or refresh fails.
						const access = await authStorage.oauth.access(
							provider,
							manager.getSessionId(),
							{ modelId },
						);
						if (!access)
							throw new Error(
								"OMP Anthropic OAuth unavailable; API-key fallback prohibited",
							);
						return access.accessToken;
					};
				}
				return registry.resolver(requestModel, manager.getSessionId());
			},
			restrictToolNames: true,
			allowRestrictedCustomTools: true,
			toolNames: allowed,
			customTools:
				policy === "read-only-no-shell"
					? readTools(sdk.z, cwd)
					: policy === "workspace-write-no-shell"
						? writeTools(sdk.z, cwd)
						: [],
			enableMCP: false,
			enableLsp: false,
			enableIrc: false,
			disableExtensionDiscovery: true,
			skills: [],
			rules: [],
			contextFiles: [],
			promptTemplates: [],
			slashCommands: [],
			extensions: [],
			additionalExtensionPaths: [],
			hasUI: false,
			systemPrompt:
				systemPrompt ||
				"Complete the supplied task using only the explicitly available snapshot tools. Return the requested response.",
		});
		let output = Promise.resolve();
		const emit = (event: unknown) => {
			const line = `${JSON.stringify(event)}\n`;
			output = output.then(() => writeOutput(line));
		};
		try {
			if (
				session.model?.provider !== provider ||
				session.model?.id !== modelId
			) {
				throw new Error("OMP changed the selected model during initialization");
			}
			const active = session.getEnabledToolNames().sort();
			if (JSON.stringify(active) !== JSON.stringify([...allowed].sort())) {
				throw new Error(`OMP tool allowlist mismatch: ${active.join(",")}`);
			}
			const header = manager.getHeader();
			if (!header) throw new Error("OMP lacks native session header");
			emit({
				type: "execution",
				schema_version: "agent-execution.omp-execution/v1",
				sdk_version: SDK_VERSION,
				harness: "omp",
				selector,
				tool_policy: policy,
				tools: allowed,
				session_id: header.id,
				cwd,
				prompt_sha256: createHash("sha256").update(prompt).digest("hex"),
				execution_id: randomUUID(),
			});
			emit(header);
			session.subscribe((event) => {
				// message_end is authoritative. Omit redundant/delta snapshots, not calls or results.
				if (
					[
						"message_start",
						"message_update",
						"tool_execution_update",
						"tool_stream_update",
					].includes(event.type)
				)
					return;
				emit(event);
			});
			await session.prompt(prompt, { expandPromptTemplates: false });
			const last = session.getLastAssistantMessage();
			if (
				!last ||
				last.stopReason === "error" ||
				last.stopReason === "aborted"
			) {
				throw new Error(
					last?.errorMessage ||
						"OMP produced no successful final assistant message",
				);
			}
			if (last.provider !== provider || last.model !== modelId)
				throw new Error("OMP final identity differs from selector");
			emit({ type: "execution_end", session_id: header.id });
			await output;
		} finally {
			await session.dispose();
		}
	} finally {
		authStorage.close();
	}
}

if (import.meta.main) {
	// This process ends after output is drained and the owned SDK resources are disposed.
	main().then(
		() => process.exit(0),
		(error) => {
			process.stderr.write(
				`${error instanceof Error ? error.message : String(error)}\n`,
				() => process.exit(1),
			);
		},
	);
}
