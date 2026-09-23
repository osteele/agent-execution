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
import {
	confinedPath,
	readTools,
	SDK_VERSION,
	writeTools,
} from "../src/agent_execution/omp_sdk";

const temporary: string[] = [];
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

test("ranged reads and literal search inspect large sources without widening writes", async () => {
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
	const write = writeTools(sdk.z, root).find(
		(tool) => tool.name === "execution_write",
	)!;
	await expect(
		write.execute("large-write", {
			path: "large.txt",
			content: source + "changed\n",
		}),
	).rejects.toThrow();
	expect(await readFile(join(root, "large.txt"), "utf8")).toBe(source);
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
