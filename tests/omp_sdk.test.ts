import { afterEach, expect, test } from "bun:test";
import {
	mkdtemp,
	mkdir,
	readFile,
	realpath,
	rm,
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
		pattern: "",
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
	await expect(
		read.execute("escape-1", { path: "escape", pattern: "" }),
	).rejects.toThrow();
	expect(await readFile(join(root, "source.txt"), "utf8")).toBe(
		"first\nneedle\nlast\n",
	);
});
