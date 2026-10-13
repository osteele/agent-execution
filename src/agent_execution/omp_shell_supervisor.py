"""Run one macOS Seatbelt command and reap every process carrying its profile."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import select
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO

PROC_UID_ONLY = 4
PROC_PID_BUFFER_BYTES = 4 * 1024 * 1024
SANDBOX_FILTER_NONE = 0
SANDBOX_FILTER_PATH = 1
SANDBOX_CHECK_NO_REPORT = 0x40000000
OUTPUT_LIMIT = 200 * 1024
KILL_GRACE_SECONDS = 0.25


def sbpl_string(value: str) -> str:
    return json.dumps(value)


class ProcessMarkers:
    """Use two per-run SBPL decisions to identify only this sandbox's processes."""

    def __init__(self, directory: Path):
        self.denied = (directory / "denied").resolve()
        self.allowed = (directory / "allowed").resolve()
        for path in (self.denied, self.allowed):
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        self.denied_c = os.fsencode(self.denied)
        self.allowed_c = os.fsencode(self.allowed)
        self._libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        sandbox_path = ctypes.util.find_library("system_sandbox")
        if not sandbox_path:
            sandbox_path = "/usr/lib/system/libsystem_sandbox.dylib"
        self._sandbox = ctypes.CDLL(sandbox_path, use_errno=True)
        self._proc_listpids = self._libproc.proc_listpids
        self._proc_listpids.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
        self._proc_listpids.restype = ctypes.c_int
        self._sandbox_check = self._sandbox.sandbox_check
        self._sandbox_check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self._sandbox_check.restype = ctypes.c_int
        self._verify_api()

    def _check_path(self, pid: int, path: bytes) -> int:
        result = self._sandbox_check(
            pid,
            b"file-read-metadata",
            SANDBOX_FILTER_PATH | SANDBOX_CHECK_NO_REPORT,
            ctypes.c_char_p(path),
        )
        return result if result in (0, 1) else -1

    def _verify_api(self) -> None:
        if not self.denied.is_file() or not self.allowed.is_file():
            raise RuntimeError("sandbox process marker files are missing")
        current = os.getpid()
        if (
            self._check_path(current, self.denied_c) != 0
            or self._check_path(current, self.allowed_c) != 0
        ):
            raise RuntimeError("sandbox_check cannot positively distinguish this supervisor")
        if self.members():
            raise RuntimeError("unexpected process already matches the unique sandbox markers")

    def members(self) -> list[int]:
        if not self.denied.is_file() or not self.allowed.is_file():
            raise RuntimeError("sandbox process marker was removed")
        pids = (ctypes.c_int * (PROC_PID_BUFFER_BYTES // ctypes.sizeof(ctypes.c_int)))()
        size = self._proc_listpids(PROC_UID_ONLY, os.getuid(), pids, ctypes.sizeof(pids))
        if size <= 0:
            code = ctypes.get_errno()
            raise OSError(code, "proc_listpids(PROC_UID_ONLY) failed")
        count = min(size // ctypes.sizeof(ctypes.c_int), len(pids))
        own_pid = os.getpid()
        matched: list[int] = []
        for pid in pids[:count]:
            if pid <= 1 or pid == own_pid:
                continue
            sandboxed = self._sandbox_check(pid, None, SANDBOX_FILTER_NONE)
            if sandboxed not in (0, 1):
                continue  # The process exited or became inaccessible between enumeration and check.
            if sandboxed != 1:
                continue
            if (
                self._check_path(pid, self.denied_c) == 1
                and self._check_path(pid, self.allowed_c) == 0
            ):
                matched.append(pid)
        return matched

    def rules(self) -> str:
        return "\n".join(
            (
                f"(deny file-read-metadata (literal {sbpl_string(str(self.denied))}))",
                f"(allow file-read-metadata (literal {sbpl_string(str(self.allowed))}))",
                f"(deny file-write* (literal {sbpl_string(str(self.denied))}))",
                f"(deny file-write* (literal {sbpl_string(str(self.allowed))}))",
            )
        )

    def kill_members(self, sig: int) -> None:
        for pid in self.members():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue

    def terminate_all(self) -> None:
        deadline = time.monotonic() + 2
        while self.members():
            # Stop forking before killing; rescan to include children born
            # between enumeration and SIGSTOP.
            self.kill_members(signal.SIGSTOP)
            self.kill_members(signal.SIGKILL)
            if time.monotonic() >= deadline:
                raise RuntimeError("sandbox processes remain after forced cleanup")
            time.sleep(0.01)


def metadata_names() -> tuple[str, ...]:
    return (
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
    )


def metadata_regex_component(name: str) -> str:
    return "".join(
        f"[{char.lower()}{char.upper()}]" if char.isalpha() else "[.]" if char == "." else char
        for char in name
    )


def result_evidence_pattern() -> str:
    stem = metadata_regex_component("agent-execution-worker-result")
    return "/" + stem + "(-[^/]*)?" + metadata_regex_component(".json") + "(/|$)"


def runtime_paths(python_executable: Path, home: Path) -> tuple[set[Path], set[Path]]:
    roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    uv_python = home / ".local" / "share" / "uv" / "python"
    if uv_python.exists():
        roots.add(uv_python.resolve())
    for executable in (home / ".local" / "bin" / "uv", home / ".bun" / "bin" / "bun"):
        if executable.exists():
            roots.add(executable.resolve())
    roots.add(python_executable.resolve())
    developer = subprocess.run(
        ["/usr/bin/xcode-select", "-p"],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(home)},
    )
    if developer.returncode == 0 and developer.stdout.strip():
        selected = Path(developer.stdout.strip()).resolve()
        # Xcode's shims load sibling frameworks and Info.plist outside Developer.
        roots.add(next((path for path in selected.parents if path.suffix == ".app"), selected))
    executables = {
        Path("/usr/bin/python3"),
        Path("/usr/bin/xcrun"),
        Path("/usr/bin/xcode-select"),
        python_executable.resolve(),
    }
    for executable in (home / ".local" / "bin" / "uv", home / ".bun" / "bin" / "bun"):
        if executable.exists():
            executables.add(executable.resolve())
    return roots, executables


def sandbox_profile(
    root: Path,
    scratch: Path,
    markers: ProcessMarkers,
    roots: set[Path],
    executables: set[Path],
) -> str:
    root_path = sbpl_string(str(root))
    scratch_path = sbpl_string(str(scratch))
    brew_roots = {
        Path(prefix) / directory
        for prefix in ("/opt/homebrew", "/usr/local")
        for directory in ("bin", "sbin", "Cellar", "opt", "lib", "share", "Frameworks")
    }
    read_roots = (
        roots
        | brew_roots
        | {
            Path(path)
            for path in (
                "/System/Library",
                "/usr/bin",
                "/usr/sbin",
                "/usr/lib",
                "/usr/libexec",
                "/usr/share",
                "/bin",
                "/sbin",
            )
        }
    )
    exec_rules = " ".join(f"(literal {sbpl_string(str(path))})" for path in sorted(executables))
    openssl_rules = " ".join(
        f"(literal {sbpl_string(str(prefix / 'etc' / 'openssl@3' / 'openssl.cnf'))})"
        for prefix in (Path("/opt/homebrew"), Path("/usr/local"))
    )
    read_rules = " ".join(f"(subpath {sbpl_string(str(path))})" for path in sorted(read_roots))
    exec_roots = " ".join(
        f"(subpath {sbpl_string(str(path))})" for path in sorted(roots | brew_roots)
    )
    developer_links = {Path("/var/db/xcode_select_link"), Path("/private/var/db/xcode_select_link")}
    ancestry = {
        parent
        for path in read_roots
        | developer_links
        | {root, scratch, Path("/var/select"), Path("/private/var/select")}
        for parent in path.parents
    }
    metadata_rules = " ".join(f"(literal {sbpl_string(str(path))})" for path in sorted(ancestry))
    developer_rules = " ".join(
        f"(literal {sbpl_string(str(path))})" for path in sorted(developer_links)
    )
    metadata = (
        "/(" + "|".join(metadata_regex_component(name) for name in metadata_names()) + ")(/|$)"
    )
    # Tool caches in private scratch use names such as .git; snapshot and
    # installed-runtime metadata remains protected at every directory depth.
    metadata_denies = (
        f"(require-all (regex #{sbpl_string(metadata)}) (require-not (subpath {scratch_path})))"
    )
    evidence = result_evidence_pattern()
    protected = "\n".join(
        (
            f"(deny file-read* {metadata_denies} (regex #{sbpl_string(evidence)}))",
            f"(deny file-write* {metadata_denies} (regex #{sbpl_string(evidence)}))",
        )
    )
    return f"""
(version 1)
(deny default)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-exec (subpath "/bin") (subpath "/usr/bin") (subpath "/usr/sbin") (subpath "/sbin") (subpath {root_path}) (subpath {scratch_path}) {exec_roots} {exec_rules})
; Root-directory access permits process startup without exposing descendants.
(allow file-read* (literal "/"))
(allow file-read* (subpath "/private/var/select") (subpath "/var/select"))
(allow file-read* {developer_rules})
(allow file-read-metadata {metadata_rules})
(allow file-read* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/random") (literal "/dev/urandom") {read_rules} (subpath {root_path}) (subpath {scratch_path}))
; Homebrew-linked Node loads this OpenSSL configuration at startup.
(allow file-read* {openssl_rules})
; Runtimes read CPU/memory topology and OS version; named host identifiers
; and everything outside these names (process census, kern.proc.*) stay denied.
(allow sysctl-read (sysctl-name-prefix "hw.") (sysctl-name-prefix "kern.os")
    (sysctl-name "kern.version") (sysctl-name "kern.hostname"))
(deny sysctl-read (sysctl-name "hw.uuid") (sysctl-name "hw.serial"))
(allow file-write* (subpath {root_path}) (subpath {scratch_path}) (literal "/dev/null"))
(deny network*)
{markers.rules()}
{protected}
""".strip()


def collect(stream: IO[bytes], destination: bytearray, selector: selectors.BaseSelector) -> bool:
    chunk = os.read(stream.fileno(), 65536)
    if not chunk:
        selector.unregister(stream)
        stream.close()
        return False
    room = max(0, OUTPUT_LIMIT - len(destination))
    destination.extend(chunk[:room])
    return len(chunk) > room


def decode_output(raw: bytearray) -> tuple[str, bool]:
    text = raw.decode("utf-8", errors="replace")
    encoded = text.encode("utf-8")
    if len(encoded) > OUTPUT_LIMIT:
        return encoded[:OUTPUT_LIMIT].decode("utf-8", errors="ignore"), True
    return text, False


def supervise(
    root: Path,
    scratch: Path,
    timeout: int,
    command: str,
    probe: bool,
    python_executable: Path,
    home: Path,
) -> dict[str, object] | None:
    if sys.platform != "darwin":
        raise RuntimeError("workspace-write-shell requires macOS sandbox-exec")
    marker_dir = Path(tempfile.mkdtemp(prefix="agent-execution-markers-", dir=root.parent))
    markers: ProcessMarkers | None = None
    child: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    output_truncated = False
    timed_out = False
    parent_gone = False
    marker_seen = False
    terminate_at: float | None = None
    termination_done = False
    try:
        markers = ProcessMarkers(marker_dir)
        if marker_dir.is_relative_to(root) or marker_dir.is_relative_to(scratch):
            raise RuntimeError("sandbox process markers overlap a writable command directory")
        if select.select([sys.stdin.fileno()], [], [], 0)[0] and not os.read(sys.stdin.fileno(), 1):
            parent_gone = True
            return None
        roots, executables = runtime_paths(python_executable, home)
        tool_bin = scratch / ".agent-execution-bin"
        tool_bin.mkdir(mode=0o700)
        for name, target in (
            ("python3", Path(sys.executable).resolve()),
            ("uv", home / ".local" / "bin" / "uv"),
            ("bun", home / ".bun" / "bin" / "bun"),
        ):
            if target.exists():
                (tool_bin / name).symlink_to(target.resolve())
        profile = sandbox_profile(root, scratch, markers, roots, executables)
        child = subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-c", command],
            cwd=root,
            env={
                "PATH": f"{tool_bin}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME": str(scratch),
                "TMPDIR": str(scratch),
                "XDG_CACHE_HOME": str(scratch / "cache"),
                "UV_CACHE_DIR": str(scratch / "cache" / "uv"),
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        assert child.stdout is not None and child.stderr is not None
        deadline = time.monotonic() + timeout
        probe_deadline = time.monotonic() + 3
        selector.register(sys.stdin.buffer, selectors.EVENT_READ, "parent")
        selector.register(child.stdout, selectors.EVENT_READ, "stdout")
        selector.register(child.stderr, selectors.EVENT_READ, "stderr")
        open_outputs = {"stdout", "stderr"}
        while True:
            now = time.monotonic()
            if probe and not marker_seen and now < probe_deadline:
                marker_seen = bool(markers.members())
            if not timed_out and now >= deadline:
                timed_out = True
                markers.kill_members(signal.SIGTERM)
                terminate_at = now + KILL_GRACE_SECONDS
            if child.poll() is not None and not termination_done and terminate_at is None:
                if markers.members():
                    markers.kill_members(signal.SIGTERM)
                    terminate_at = now + KILL_GRACE_SECONDS
                else:
                    termination_done = True
            if terminate_at is not None and now >= terminate_at:
                markers.terminate_all()
                terminate_at = None
                termination_done = True
            if child.poll() is not None and not open_outputs and terminate_at is None:
                break
            remaining = max(0.0, deadline - now) if not timed_out else 0.05
            for key, _ in selector.select(min(0.05, remaining) if remaining else 0.01):
                if key.data == "parent":
                    if not os.read(sys.stdin.fileno(), 1):
                        parent_gone = True
                        markers.terminate_all()
                        break
                    raise RuntimeError("unexpected bytes on supervisor liveness pipe")
                if key.data == "stdout":
                    output_truncated |= collect(child.stdout, stdout, selector)
                    if child.stdout.closed:
                        open_outputs.discard("stdout")
                elif key.data == "stderr":
                    output_truncated |= collect(child.stderr, stderr, selector)
                    if child.stderr.closed:
                        open_outputs.discard("stderr")
            if parent_gone:
                return None
        markers.terminate_all()
        returncode = child.wait()
        stdout_text, stdout_cut = decode_output(stdout)
        stderr_text, stderr_cut = decode_output(stderr)
        return {
            "command": command,
            "exit_code": returncode if returncode >= 0 else 128 + -returncode,
            "timed_out": timed_out,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "truncated": output_truncated or stdout_cut or stderr_cut,
            "marker_seen": marker_seen,
        }
    finally:
        selector.close()
        try:
            if markers is not None:
                markers.terminate_all()
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait(timeout=1)
            shutil.rmtree(marker_dir)
            if parent_gone:
                shutil.rmtree(scratch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("timeout", type=int)
    parser.add_argument("command")
    parser.add_argument("python_executable", type=Path)
    parser.add_argument("home", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.timeout < 1 or args.timeout > 900:
        raise RuntimeError("timeout must be an integer from 1 to 900 seconds")
    result = supervise(
        args.root.resolve(),
        args.scratch.resolve(),
        args.timeout,
        args.command,
        args.probe,
        args.python_executable.resolve(),
        args.home.resolve(),
    )
    if result is not None:
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"sandbox supervisor: {error}", file=sys.stderr)
        raise SystemExit(1)
