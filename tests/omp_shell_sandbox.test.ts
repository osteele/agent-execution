import { afterEach, expect, test } from "bun:test";
import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import { existsSync, watch } from "node:fs";
import { mkdtemp, mkdir, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { createServer, type Socket } from "node:net";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { assertShellSandboxAvailable, executeShell } from "../src/agent_execution/omp_shell_sandbox";

const temporary: string[] = [];

async function makeSnapshot() {
	const parent = await realpath(await mkdtemp(join(tmpdir(), "omp-shell-policy-")));
	temporary.push(parent);
	const root = join(parent, "snapshot");
	await mkdir(root);
	await writeFile(join(parent, "host-secret.txt"), "host secret");
	await writeFile(join(root, "source.txt"), "snapshot bytes\n");
	return { parent, root };
}

function processStatus(pid: number): string {
	const result = spawnSync("/bin/ps", ["-o", "stat=", "-p", String(pid)], { encoding: "utf8" });
	if (result.error) throw result.error;
	if (result.status === 1 && result.stdout.trim() === "") return "";
	if (result.status !== 0) throw new Error(`ps failed: ${result.stderr}`);
	return result.stdout.trim();
}

afterEach(async () => {
	await Promise.all(temporary.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

test.if(process.platform !== "darwin")("unsupported platforms fail closed before shell execution", async () => {
	const { root } = await makeSnapshot();
	await expect(assertShellSandboxAvailable(root, "/usr/bin/python3")).rejects.toThrow("sandbox-exec is required");
	await expect(executeShell(root, "echo should-not-run", "/usr/bin/python3")).rejects.toThrow("sandbox-exec is required");
});

test.if(process.platform === "darwin")("macOS sandbox executes commands and retains bounded structured results", async () => {
	const { root } = await makeSnapshot();
	await assertShellSandboxAvailable(root, "/usr/bin/python3");
	const success = await executeShell(root, "printf 'hello'; printf 'problem' >&2; exit 7", "/usr/bin/python3", 10);
	expect(success).toMatchObject({ exit_code: 7, timed_out: false, stdout: "hello", stderr: "problem", truncated: false });

	const large = await executeShell(root, "yes x | head -c 500000", "/usr/bin/python3", 10);
	expect(large.truncated).toBe(true);
	const binary = await executeShell(root, `python3 -c 'import os; os.write(1, b"\\xff"*200000)'`, "/usr/bin/python3", 10);
	expect(binary.truncated).toBe(true);
	expect(Buffer.byteLength(binary.stdout)).toBeLessThanOrEqual(200 * 1024);
});

test.if(process.platform === "darwin")("macOS sandbox refuses host reads and writes", async () => {
	const { parent, root } = await makeSnapshot();
	const outside = join(parent, "host-secret.txt");
	const result = await executeShell(root, `cat '${outside}'; printf changed > '${outside}'`, "/usr/bin/python3", 10);
	expect(result.exit_code).not.toBe(0);
	expect(result.stdout).not.toContain("host secret");
	expect(await readFile(outside, "utf8")).toBe("host secret");
});

test.if(process.platform === "darwin" && existsSync("/System/Volumes/Data"))("macOS sandbox refuses host data through the Data volume alias", async () => {
	const { parent, root } = await makeSnapshot();
	const alias = `/System/Volumes/Data${join(parent, "host-secret.txt")}`;
	expect(await readFile(alias, "utf8")).toBe("host secret");
	const result = await executeShell(root, `cat '${alias}'`, "/usr/bin/python3", 10);
	expect(result.exit_code).not.toBe(0);
	expect(result.stdout).not.toContain("host secret");
});

const brewPrefix = process.arch === "arm64" ? "/opt/homebrew" : "/usr/local";
test.if(process.platform === "darwin" && existsSync(join(brewPrefix, "etc")) && existsSync(join(brewPrefix, "var")))("macOS sandbox refuses Homebrew configuration and service directories", async () => {
	const { root } = await makeSnapshot();
	for (const directory of ["etc", "var"]) {
		const host = join(brewPrefix, directory);
		expect(spawnSync("/bin/ls", ["-A", host]).status).toBe(0);
		const result = await executeShell(root, `/bin/ls -A '${host}'`, "/usr/bin/python3", 10);
		expect(result.exit_code).not.toBe(0);
		expect(result.stdout).toBe("");
	}
});

test.if(process.platform === "darwin")("macOS sandbox lets runtimes read but not write system parameters", async () => {
	// Rust's std (uv, ruff) sizes its stack guard page and bun its allocator
	// from sysctl; without sysctl-read both abort at startup (ax18).
	const { root } = await makeSnapshot();
	const read = await executeShell(root, "/usr/sbin/sysctl -n hw.pagesize", "/usr/bin/python3", 10);
	expect(read.exit_code).toBe(0);
	expect(Number(read.stdout.trim())).toBeGreaterThan(0);
	const write = await executeShell(root, "/usr/sbin/sysctl -w kern.maxfiles=1", "/usr/bin/python3", 10);
	expect(write.exit_code).not.toBe(0);
});

const uv = [join(homedir(), ".local", "bin", "uv"), "/opt/homebrew/bin/uv"].find((path) => existsSync(path));
test.if(process.platform === "darwin" && uv !== undefined)("macOS sandbox runs a Rust toolchain binary", async () => {
	const { root } = await makeSnapshot();
	const result = await executeShell(root, "uv --version", "/usr/bin/python3", 10);
	expect(result.exit_code).toBe(0);
	expect(result.stdout).toStartWith("uv ");
});

test.if(process.platform === "darwin" && existsSync(join(brewPrefix, "etc", "openssl@3", "openssl.cnf")))("macOS sandbox reads Homebrew TLS configuration but not the rest of its etc", async () => {
	const { root } = await makeSnapshot();
	const tls = await executeShell(root, `/bin/cat '${join(brewPrefix, "etc", "openssl@3", "openssl.cnf")}' >/dev/null`, "/usr/bin/python3", 10);
	expect(tls.exit_code).toBe(0);
	const gitconfig = join(brewPrefix, "etc", "gitconfig");
	if (existsSync(gitconfig)) {
		const other = await executeShell(root, `/bin/cat '${gitconfig}'`, "/usr/bin/python3", 10);
		expect(other.exit_code).not.toBe(0);
		expect(other.stdout).toBe("");
	}
});

test.if(process.platform === "darwin")("macOS sandbox cannot expose host files through hard links", async () => {
	const { parent, root } = await makeSnapshot();
	const outside = join(parent, "host-secret.txt");
	const result = await executeShell(
		root,
		`ln '${outside}' linked-secret; cat linked-secret; printf changed > linked-secret`,
		"/usr/bin/python3",
		10,
	);
	expect(result.stdout).not.toContain("host secret");
	expect(await readFile(outside, "utf8")).toBe("host secret");
});


test.if(process.platform === "darwin")("macOS sandbox refuses reads, writes and renames of protected metadata", async () => {
	const { root } = await makeSnapshot();
	const git = join(root, ".git");
	await mkdir(git);
	await writeFile(join(git, "config"), "original");
	const result = await executeShell(root, "cat .git/config; printf forged > .git/config; mv .git/config .git/forged", "/usr/bin/python3", 10);
	expect(result.exit_code).not.toBe(0);
	expect(result.stdout).not.toContain("original");
	expect(await readFile(join(git, "config"), "utf8")).toBe("original");
	const evidence = join(root, "agent-execution-worker-result-copy.json");
	await writeFile(evidence, "retained evidence");
	const forged = await executeShell(root, "printf forged > agent-execution-worker-result-copy.json", "/usr/bin/python3", 10);
	expect(forged.exit_code).not.toBe(0);
	expect(await readFile(evidence, "utf8")).toBe("retained evidence");
});

test.if(process.platform === "darwin")("macOS sandbox denies network access", async () => {
	const { root } = await makeSnapshot();
	const server = createServer();
	let accepted = false;
	const sockets = new Set<Socket>();
	server.on("connection", (socket) => {
		accepted = true;
		sockets.add(socket);
		socket.on("close", () => sockets.delete(socket));
		socket.once("data", () => socket.end("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"));
	});
	server.listen(0, "127.0.0.1");
	await once(server, "listening");
	const address = server.address();
	if (!address || typeof address === "string") throw new Error("test server did not bind");
	const code = `import socket; client = socket.socket(); client.settimeout(2); client.connect(("127.0.0.1", ${address.port})); client.sendall(b"probe"); client.recv(1024)`;
	try {
		const control = spawn("/usr/bin/python3", ["-c", code], { stdio: "ignore" });
		const [controlExit] = await once(control, "close");
		expect(controlExit).toBe(0);
		expect(accepted).toBe(true);
		accepted = false;
		const result = await executeShell(root, `python3 -c '${code}'`, "/usr/bin/python3", 10);
		expect(result.exit_code).not.toBe(0);
		expect(accepted).toBe(false);
	} finally {
		for (const socket of sockets) socket.destroy();
		const closed = once(server, "close");
		server.close();
		await closed;
	}
});

test.if(process.platform === "darwin")("macOS sandbox command environment excludes harness session identity variables", async () => {
	// Issue ax17: a check inheriting the writer session's identity variables
	// binds its effects to that session and fails tests that pass in a clean
	// shell. The confined writer shell passes an allow-listed environment, so
	// these names must never reach the command even when the parent exports them.
	const { root } = await makeSnapshot();
	const names = ["CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID", "OMP_SESSION_ID"];
	const prior = names.map((name) => process.env[name]);
	names.forEach((name, index) => {
		process.env[name] = `ax17-${index}`;
	});
	try {
		const result = await executeShell(root, "env | sort", "/usr/bin/python3", 10);
		expect(result.exit_code).toBe(0);
		for (const name of names) expect(result.stdout).not.toContain(name);
		expect(result.stdout).toContain("PATH=");
	} finally {
		names.forEach((name, index) => {
			if (prior[index] === undefined) delete process.env[name];
			else process.env[name] = prior[index];
		});
	}
});

test.if(process.platform === "darwin")("timeout kills the complete command process group", async () => {
	const { root } = await makeSnapshot();
	const marker = join(root, "child.pid");
	const result = await executeShell(root, `sleep 30 & echo $! > '${marker}'; wait`, "/usr/bin/python3", 1);
	expect(result.timed_out).toBe(true);
	const pid = Number(await readFile(marker, "utf8"));
	// This is an integration assertion on the real macOS process table after the real timeout.
	const status = processStatus(pid);
	expect(status === "" || status.startsWith("Z")).toBe(true);
});
test.if(process.platform === "darwin")("timeout kills a double-forked session escape", async () => {
	const { root } = await makeSnapshot();
	const pidFile = join(root, "escaped.pid");
	const lateFile = join(root, "escaped-late.txt");
	const code = `import os, pathlib, time
p=os.fork()
if p == 0:
    g=os.fork()
    if g > 0: os._exit(0)
    os.setsid()
    pathlib.Path(${JSON.stringify(pidFile)}).write_text(str(os.getpid()))
    time.sleep(2)
    pathlib.Path(${JSON.stringify(lateFile)}).write_text("still alive")
    time.sleep(30)
else:
    time.sleep(30)`;
	const result = await executeShell(root, `python3 -c '${code}'`, "/usr/bin/python3", 1);
	expect(result.timed_out).toBe(true);
	const pid = Number(await readFile(pidFile, "utf8"));
	// Real process-table observation and a delayed marker prove cleanup beyond the direct process group.
	await Bun.sleep(2200);
	const status = processStatus(pid);
	expect(status === "" || status.startsWith("Z")).toBe(true);
	await expect(readFile(lateFile, "utf8")).rejects.toThrow();
});

test.if(process.platform === "darwin")("parent SIGKILL closes the watchdog lease and kills session escapes", async () => {
	const { root } = await makeSnapshot();
	const pidFile = join(root, "cancelled-escape.pid");
	const lateFile = join(root, "cancelled-late.txt");
	const code = `import os, pathlib, time
p=os.fork()
if p == 0:
    os.setsid()
    pathlib.Path(${JSON.stringify(pidFile)}).write_text(str(os.getpid()))
    time.sleep(2)
    pathlib.Path(${JSON.stringify(lateFile)}).write_text("still alive")
    time.sleep(30)
else:
    time.sleep(30)`;
	const moduleUrl = new URL("../src/agent_execution/omp_shell_sandbox.ts", import.meta.url).href;
	const driver = spawn(process.execPath, ["-e", `import { executeShell } from ${JSON.stringify(moduleUrl)}; await executeShell(${JSON.stringify(root)}, ${JSON.stringify(`python3 -c '${code}'`)}, "/usr/bin/python3", 60);`], { stdio: "ignore" });
	const appeared = Promise.withResolvers<void>();
	const watcher = watch(root, () => {
		void readFile(pidFile).then(() => appeared.resolve(), (error: NodeJS.ErrnoException) => {
			if (error.code !== "ENOENT") appeared.reject(error);
		});
	});
	const watchDeadline = setTimeout(() => appeared.reject(new Error("session escape pid was not written")), 10_000);
	let pid = 0;
	try {
		await appeared.promise;
		clearTimeout(watchDeadline);
		pid = Number(await readFile(pidFile, "utf8"));
		driver.kill("SIGKILL");
		await once(driver, "close");
		// Real process-table polling is necessary: the escapee is reparented and has no wait handle here.
		const deadline = Date.now() + 5_000;
		let status = "running";
		while (Date.now() < deadline && status !== "" && !status.startsWith("Z")) {
			status = processStatus(pid);
			if (status && !status.startsWith("Z")) await Bun.sleep(25);
		}
		expect(status === "" || status.startsWith("Z")).toBe(true);
		await Bun.sleep(2200);
		await expect(readFile(lateFile, "utf8")).rejects.toThrow();
	} finally {
		clearTimeout(watchDeadline);
		watcher.close();
		if (driver.exitCode === null && driver.signalCode === null) driver.kill("SIGKILL");
		if (pid > 1) {
			try { process.kill(pid, "SIGKILL"); } catch (error) {
				if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
			}
		}
	}
});
