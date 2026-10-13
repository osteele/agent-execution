import { spawn } from "node:child_process";
import { mkdtemp, realpath, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";

export const DEFAULT_SHELL_TIMEOUT_SECONDS = 60;
export const MAX_SHELL_TIMEOUT_SECONDS = 900;
const MAX_RESULT_BYTES = 4 * 1024 * 1024;
const MAX_ERROR_BYTES = 64 * 1024;

export interface ShellResult {
	command: string;
	exit_code: number | null;
	timed_out: boolean;
	stdout: string;
	stderr: string;
	truncated: boolean;
}

interface SupervisorResult extends ShellResult {
	marker_seen: boolean;
}

function boundedText(limit: number) {
	let bytes = 0;
	const chunks: Buffer[] = [];
	return {
		append(chunk: Buffer) {
			if (bytes >= limit) return;
			const take = Math.min(chunk.length, limit - bytes);
			chunks.push(chunk.subarray(0, take));
			bytes += take;
		},
		get text() { return Buffer.concat(chunks, bytes).toString("utf8"); },
	};
}

async function runSupervisor(
	root: string,
	scratch: string,
	command: string,
	timeoutSeconds: number,
	pythonExecutable: string,
	probe = false,
): Promise<SupervisorResult> {
	if (!isAbsolute(pythonExecutable))
		throw new Error("OMP Python helper interpreter path is missing or not absolute");
	const helper = join(dirname(fileURLToPath(import.meta.url)), "omp_shell_supervisor.py");
	const child = spawn(
		pythonExecutable,
		[
			helper,
			root,
			scratch,
			String(timeoutSeconds),
			command,
			pythonExecutable,
			homedir(),
			...(probe ? ["--probe"] : []),
		],
		{
			cwd: root,
			env: {
				PATH: "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
				HOME: scratch,
				TMPDIR: scratch,
				XDG_CACHE_HOME: join(scratch, "cache"),
				UV_CACHE_DIR: join(scratch, "cache", "uv"),
			},
			stdio: ["pipe", "pipe", "pipe"],
			detached: true,
		},
	);
	const output = boundedText(MAX_RESULT_BYTES);
	const errors = boundedText(MAX_ERROR_BYTES);
	child.stdout?.on("data", (chunk: Buffer) => output.append(Buffer.from(chunk)));
	child.stderr?.on("data", (chunk: Buffer) => errors.append(Buffer.from(chunk)));
	const { promise, resolve, reject } = Promise.withResolvers<number | null>();
	child.once("error", reject);
	child.once("close", resolve);
	const code = await promise;
	if (code !== 0)
		throw new Error(errors.text.trim() || `macOS shell supervisor exited ${code}`);
	let result: unknown;
	try {
		result = JSON.parse(output.text);
	} catch (error) {
		throw new Error("macOS shell supervisor returned invalid result JSON", { cause: error });
	}
	if (!result || typeof result !== "object" || !("marker_seen" in result))
		throw new Error("macOS shell supervisor omitted its process-marker result");
	return result as SupervisorResult;
}

/** Verify both sandbox-exec and two-marker descendant discovery before model generation. */
export async function assertShellSandboxAvailable(
	root: string,
	pythonExecutable: string,
): Promise<void> {
	if (process.platform !== "darwin")
		throw new Error(`workspace-write-shell is unsupported on ${process.platform}; macOS sandbox-exec is required`);
	const canonicalRoot = await realpath(root);
	const scratch = await realpath(await mkdtemp(join(tmpdir(), "agent-execution-shell-probe-")));
	try {
		const result = await runSupervisor(
			canonicalRoot,
			scratch,
			"/bin/sleep 1",
			10,
			pythonExecutable,
			true,
		);
		if (result.exit_code !== 0 || result.timed_out || !result.marker_seen)
			throw new Error("macOS sandbox process discovery probe failed");
	} finally {
		await rm(scratch, { recursive: true, force: true });
	}
}

export async function executeShell(
	root: string,
	command: string,
	pythonExecutable: string,
	timeoutSeconds = DEFAULT_SHELL_TIMEOUT_SECONDS,
): Promise<ShellResult> {
	if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 1 || timeoutSeconds > MAX_SHELL_TIMEOUT_SECONDS)
		throw new Error(`timeout must be an integer from 1 to ${MAX_SHELL_TIMEOUT_SECONDS} seconds`);
	if (process.platform !== "darwin")
		throw new Error(`workspace-write-shell is unsupported on ${process.platform}; macOS sandbox-exec is required`);
	const canonicalRoot = await realpath(root);
	const scratch = await realpath(await mkdtemp(join(tmpdir(), "agent-execution-shell-")));
	try {
		const { marker_seen: _markerSeen, ...result } = await runSupervisor(
			canonicalRoot,
			scratch,
			command,
			timeoutSeconds,
			pythonExecutable,
		);
		return result;
	} finally {
		await rm(scratch, { recursive: true, force: true });
	}
}
