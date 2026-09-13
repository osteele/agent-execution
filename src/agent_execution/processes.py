"""Spawn agent harnesses in their own process group and account for survivors.

A harness is not one process. ``claude``, ``codex``, and ``kimi`` each
spawn helpers — MCP servers, code-mode hosts, hook interpreters — that are
children of the harness rather than of this process. ``subprocess.run`` with a
timeout kills only the direct child, so an abandoned call leaves that helper tree
running, reparented to init, holding a model context resident until someone finds
it in Activity Monitor.

Two rules follow, and both are load-bearing:

Every harness starts in its own session, so it leads its own process group, and
the *group* is what gets signalled. Killing the direct child alone is what
created the leak.

What remains after the group is signalled is recorded as a survivor rather than
assumed absent. A leak that is cleaned up silently and a leak that is still
running look identical from the outside, so the census runs on every call —
including calls that exited cleanly, because a harness that returns a good answer
and leaves an MCP server behind is exactly the case no existing record could see.

Reaping after a clean exit is separated from force-killing a timeout
(``kill_kind``) so the telemetry answers where leaks are injected, not merely
whether any exist.
"""

from __future__ import annotations

import atexit
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, TextIO, cast

from agent_execution import env
from agent_execution.timing import current_boot_id

#: Seconds a signalled group is given to exit on SIGTERM before SIGKILL.
GRACE_SECONDS = 2.0

#: Seconds to let the kernel finish tearing a group down before recounting it.
SETTLE_SECONDS = 0.25

#: Seconds to keep collecting output after the direct child has exited.
#:
#: Bounded because descendants inherit the harness's pipes: reading to EOF would
#: wait for processes this call has already decided to kill.
DRAIN_SECONDS = 1.0

#: Seconds between process-group CPU and resident-memory observations.
RESOURCE_SAMPLE_SECONDS = 15.0

#: Programs that belong to an agent harness tree.
#:
#: Used only to scope the standalone orphan survey. The per-call census matches
#: on process group and needs no name list, which is why it cannot be defeated
#: by a helper this set has not heard of.
HARNESS_PROGRAMS = frozenset(
    {
        "claude",
        "codex",
        "kimi",
        "gemini",
        "codex-code-mode-host",
        "codex-mcp-server",
        "claude-code",
    }
)


@dataclass(frozen=True)
class Survivor:
    """One process still alive when it should not have been."""

    pid: int
    ppid: int
    pgid: int
    rss_kib: int
    elapsed: str
    command: str
    cpu_percent: float = 0.0

    @property
    def program(self) -> str:
        """Basename of the executable, which is what groups usefully in a report."""
        return os.path.basename(self.command.split()[0]) if self.command.strip() else ""

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "pgid": self.pgid,
            "rss_kib": self.rss_kib,
            "elapsed": self.elapsed,
            "cpu_percent": self.cpu_percent,
            "program": self.program,
            # The full command line can carry a prompt fragment or a path, so
            # the record keeps the program name and a bounded prefix rather
            # than an unbounded copy of whatever was on the command line.
            "command": self.command[:400],
        }


@dataclass(frozen=True)
class GroupTelemetry:
    """What happened to one harness process group.

    ``kill_kind`` distinguishes the injection points: ``None`` means the group
    was empty when the direct child exited, ``group-sweep`` means the call
    finished but left descendants behind, and ``group-term``/``group-kill`` mean
    the call was abandoned on timeout and the group was signalled.
    """

    pgid: int | None
    kill_kind: str | None
    survivors: tuple[Survivor, ...]
    survivors_reaped: bool
    resource_sample_count: int = 0
    peak_process_count: int = 0
    peak_rss_kib: int = 0
    peak_cpu_percent: float = 0.0
    resource_observation_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "pgid": self.pgid,
            "kill_kind": self.kill_kind,
            "survivor_count": len(self.survivors),
            "survivors_reaped": self.survivors_reaped,
            "survivors": [survivor.to_dict() for survivor in self.survivors],
            "resources": {
                "sample_count": self.resource_sample_count,
                "peak_process_count": self.peak_process_count,
                "peak_rss_kib": self.peak_rss_kib,
                "peak_cpu_percent": round(self.peak_cpu_percent, 3),
                "observation_error": self.resource_observation_error,
            },
        }


@dataclass(frozen=True)
class ProcessIdentity:
    """The three facts required to observe one owning process without PID reuse."""

    pgid: int
    boot_id: str | None
    process_started_at: float | None


def process_started_at(pid: int) -> float | None:
    """Read a process's kernel-reported start time through the public ``ps`` interface."""
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        local = datetime.strptime(result.stdout.strip(), "%a %b %d %H:%M:%S %Y").astimezone()
    except ValueError:
        return None
    return local.timestamp()


def _drain(stream: object, sink: list[str]) -> None:
    """Collect one pipe to EOF in the background.

    Runs in a thread so the caller can stop waiting on it, which is what keeps a
    descendant holding the pipe from extending a finished call.
    """
    reader = cast("IO[str]", stream)
    try:
        sink.append(reader.read())
    except (OSError, ValueError):
        pass
    finally:
        try:
            reader.close()
        except OSError:
            pass


def _ps_lines() -> list[str]:
    """Every process on the machine, including CPU and resident memory.

    ``ps`` is the portable interface to the process table; reading it is cheap
    enough at bounded intervals and avoids a native dependency. A ``ps`` that
    cannot be run is reported by raising, not by returning an empty census. An
    empty census means "nothing survived", which is the opposite of "unknown".
    """
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,ppid=,rss=,%cpu=,etime=,command="],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.splitlines()


def _parse(line: str) -> Survivor | None:
    fields = line.split(maxsplit=6)
    if len(fields) < 7:
        return None
    pid, pgid, ppid, rss, cpu_percent, elapsed, command = fields
    if not (pid.isdigit() and pgid.isdigit() and ppid.isdigit() and rss.isdigit()):
        return None
    try:
        cpu = float(cpu_percent)
    except ValueError:
        return None
    return Survivor(
        pid=int(pid),
        ppid=int(ppid),
        pgid=int(pgid),
        rss_kib=int(rss),
        elapsed=elapsed,
        command=command,
        cpu_percent=cpu,
    )


def census(*, pgid: int) -> tuple[Survivor, ...]:
    """Every live process in one process group, excluding this process."""
    if pgid <= 1:
        raise ValueError(f"refusing to census process group {pgid}")
    mine = os.getpid()
    found = [
        survivor
        for line in _ps_lines()
        if (survivor := _parse(line)) is not None
        if survivor.pgid == pgid and survivor.pid != mine
    ]
    return tuple(sorted(found, key=lambda survivor: survivor.pid))


def orphaned_harnesses(programs: Iterable[str] = HARNESS_PROGRAMS) -> tuple[Survivor, ...]:
    """Live harness processes reparented to init.

    A harness whose parent is init was not started by a shell that is still
    around to wait for it. That is the shape a leak takes once the session that
    spawned it is gone, and it is the only shape visible after the fact.
    """
    wanted = frozenset(programs)
    found = [
        survivor
        for line in _ps_lines()
        if (survivor := _parse(line)) is not None
        if survivor.ppid == 1 and survivor.program in wanted
    ]
    return tuple(sorted(found, key=lambda survivor: survivor.pid))


def signal_group(pgid: int, sig: int) -> bool:
    """Signal a whole process group. Returns whether the group still existed.

    Signalling group 0 would signal this process's own group, and group 1 is
    init's; both are refused rather than clamped, because reaching either means
    a pgid was lost somewhere upstream and silently doing nothing would hide it.
    """
    if pgid <= 1:
        raise ValueError(f"refusing to signal process group {pgid}")
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_group(pgid: int, grace_seconds: float) -> str:
    """Signal a group to stop, escalating to SIGKILL. Returns the kill kind."""
    if not signal_group(pgid, signal.SIGTERM):
        return "group-term"
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not census(pgid=pgid):
            return "group-term"
        time.sleep(0.05)
    signal_group(pgid, signal.SIGKILL)
    return "group-kill"


def reaping_enabled() -> bool:
    """Whether a clean call's leftover descendants get killed.

    On by default: a one-shot ``codex exec`` or ``claude -p`` has no sanctioned
    persistent child, so anything still running is a leak. The escape hatch
    exists because a future harness might legitimately share a long-lived helper
    across calls, and in that case the telemetry still records what was found.
    """
    return (env.variable("HARNESS_REAP") or "1").strip().lower() not in {"0", "false", "no"}


@dataclass
class _LiveGroups:
    """Process groups this interpreter has spawned and not yet cleaned up.

    Tracked so an interrupt can tear down every in-flight harness. Without this,
    Ctrl-C leaves each running harness — and its whole helper tree — orphaned,
    which is the leak that survives closing the terminal.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    pgids: set[int] = field(default_factory=set)

    def add(self, pgid: int) -> None:
        with self.lock:
            self.pgids.add(pgid)

    def discard(self, pgid: int) -> None:
        with self.lock:
            self.pgids.discard(pgid)

    def drain(self) -> tuple[int, ...]:
        with self.lock:
            drained = tuple(sorted(self.pgids))
            self.pgids.clear()
        return drained


_live = _LiveGroups()

_telemetry = threading.local()
_spawn_observer = threading.local()


@contextmanager
def observe_process_spawns(
    callback: Callable[[ProcessIdentity], None],
) -> Iterator[None]:
    """Attach one call record to harnesses spawned on the current thread."""
    previous = getattr(_spawn_observer, "callback", None)
    _spawn_observer.callback = callback
    try:
        yield
    finally:
        _spawn_observer.callback = previous


def kill_live_groups() -> tuple[int, ...]:
    """Kill every in-flight harness group. Returns the pgids that were signalled.

    Safe to call more than once and from an exit handler: draining the registry
    first means a second call has nothing to do.
    """
    drained = _live.drain()
    for pgid in drained:
        signal_group(pgid, signal.SIGKILL)
    return drained


def install_termination_guard() -> None:
    """Tear down in-flight harness groups on interpreter exit and on SIGTERM.

    Called from the CLI entry point rather than at import: installing a signal
    handler is a process-wide side effect that only works on the main thread,
    and a library import is the wrong place to take it.

    SIGINT is deliberately not handled here. Python already raises
    KeyboardInterrupt for it, and the guard around each dispatch converts that
    into the same teardown while letting the traceback surface normally.
    """
    atexit.register(kill_live_groups)

    def handle(signum: int, frame: object) -> None:
        # Raising rather than chaining to the previous handler: this runs from a
        # CLI entry point where nothing else has installed one, and SystemExit
        # both unwinds the dispatch and preserves the conventional exit status.
        kill_live_groups()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle)


_thread_failures: list[str] = []


def install_thread_failure_guard() -> None:
    """Make a background thread's failure visible in the exit status.

    Reported from weft: a dispatch printed a transition refusal on stderr,
    recorded nothing, and exited 0. Python's default hook prints an unhandled
    thread exception and leaves the process status untouched, so work performed
    in a daemon thread -- the spawn observer's recording callback, the detached
    reaper -- fails invisibly to any caller reading the exit code, which is how
    a wedged assignment looks like a successful dispatch.

    The hook records rather than exits: killing the process from a thread would
    skip the teardown the guard above exists to run.
    """

    # Cleared here so the record belongs to this invocation. Kept as a module
    # global rather than passed through: the hook is process-wide, and the
    # threads that fail are started far from the caller that reads the status.
    _thread_failures.clear()

    def hook(args: threading.ExceptHookArgs) -> None:
        name = getattr(args.thread, "name", "unknown")
        _thread_failures.append(f"{name}: {args.exc_type.__name__}: {args.exc_value}")
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)

    threading.excepthook = hook


def thread_failures() -> tuple[str, ...]:
    """Every background-thread failure recorded since the guard was installed."""
    return tuple(_thread_failures)


class guard_live_groups:
    """Kill in-flight harness groups if the enclosed block does not finish.

    Wraps a fan-out of concurrent dispatches. On a normal exit each dispatch has
    already cleaned up its own group and the registry is empty. On
    KeyboardInterrupt, SystemExit, or any other exception the registry still
    holds the groups whose dispatches never returned, and those are exactly the
    processes that would otherwise outlive this interpreter.
    """

    def __enter__(self) -> guard_live_groups:  # noqa: PYI034
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is not None:
            kill_live_groups()
        return False


def take_telemetry() -> GroupTelemetry | None:
    """Consume the telemetry recorded by the most recent spawn on this thread.

    Reading clears it, so a dispatch that used an injected runner records
    nothing instead of inheriting a previous call's reading. One dispatch runs
    per worker thread, which is what makes thread-local the right scope.
    """
    recorded: GroupTelemetry | None = getattr(_telemetry, "value", None)
    _telemetry.value = None
    return recorded


def run_in_process_group(
    command: Sequence[str],
    cwd: Path,
    prompt: str,
    timeout: float | None,
    *,
    grace_seconds: float = GRACE_SECONDS,
    on_spawn: Callable[[ProcessIdentity], None] | None = None,
    hold_before_exec: bool = False,
    on_resource_sample: Callable[[Mapping[str, object]], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a harness in its own process group, then account for the whole group.

    Raises ``subprocess.TimeoutExpired`` on timeout, as ``subprocess.run`` does,
    but only after the group has been signalled rather than just the child.
    ``None`` leaves completion unbounded while retaining resource sampling and cleanup.
    """
    release_read: int | None = None
    release_write: int | None = None
    launch_command = list(command)
    if hold_before_exec:
        release_read, release_write = os.pipe()
        launch_command = [
            sys.executable,
            str(Path(__file__).with_name("spawn_gate.py")),
            str(release_read),
            *launch_command,
        ]
    try:
        process = subprocess.Popen(
            launch_command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=(release_read,) if release_read is not None else (),
        )
    except (OSError, ValueError):
        if release_write is not None:
            os.close(release_write)
        raise
    finally:
        if release_read is not None:
            os.close(release_read)
    # start_new_session makes the child a session and group leader, so its pgid
    # equals its pid by construction. Reading it back with getpgid would race a
    # child that has already exited.
    pgid = process.pid
    _live.add(pgid)
    spawn_recorded = False
    try:
        observer = on_spawn or getattr(_spawn_observer, "callback", None)
        if observer is not None:
            observer(
                ProcessIdentity(
                    pgid=pgid,
                    boot_id=current_boot_id(),
                    process_started_at=process_started_at(pgid),
                )
            )
        if release_write is not None:
            os.write(release_write, b"1")
            os.close(release_write)
            release_write = None
        spawn_recorded = True
    finally:
        if not spawn_recorded:
            if release_write is not None:
                os.close(release_write)
            signal_group(pgid, signal.SIGKILL)
            process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            _live.discard(pgid)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    readers = (
        threading.Thread(target=_drain, args=(process.stdout, stdout_parts), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr_parts), daemon=True),
    )
    for reader in readers:
        reader.start()
    kill_kind: str | None = None
    timed_out = False
    resource_sample_count = 0
    peak_process_count = 0
    peak_rss_kib = 0
    peak_cpu_percent = 0.0
    resource_observation_error: str | None = None

    def sample_resources() -> None:
        nonlocal resource_sample_count
        nonlocal peak_process_count, peak_rss_kib, peak_cpu_percent
        nonlocal resource_observation_error
        try:
            processes = census(pgid=pgid)
        except (OSError, subprocess.SubprocessError) as error:
            if resource_observation_error is None:
                resource_observation_error = f"{type(error).__name__}: {error}"
            return
        resource_sample_count += 1
        process_count = len(processes)
        rss_kib = sum(item.rss_kib for item in processes)
        cpu_percent = sum(item.cpu_percent for item in processes)
        peak_process_count = max(peak_process_count, process_count)
        peak_rss_kib = max(peak_rss_kib, rss_kib)
        peak_cpu_percent = max(peak_cpu_percent, cpu_percent)
        if on_resource_sample is not None:
            try:
                on_resource_sample(
                    {
                        "sample_count": resource_sample_count,
                        "process_count": process_count,
                        "rss_kib": rss_kib,
                        "cpu_percent": round(cpu_percent, 3),
                        "peak_process_count": peak_process_count,
                        "peak_rss_kib": peak_rss_kib,
                        "peak_cpu_percent": round(peak_cpu_percent, 3),
                    }
                )
            except (OSError, ValueError) as error:
                if resource_observation_error is None:
                    resource_observation_error = f"{type(error).__name__}: {error}"

    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        if process.stdin is not None:
            try:
                # Prompt delivery shares the completion deadline, including
                # when the harness never reads its input pipe.
                if prompt:
                    descriptor = process.stdin.fileno()
                    os.set_blocking(descriptor, False)
                    with (
                        memoryview(prompt.encode(cast(TextIO, process.stdin).encoding)) as payload,
                        selectors.DefaultSelector() as selector,
                    ):
                        selector.register(descriptor, selectors.EVENT_WRITE)
                        offset = 0
                        while offset < len(payload):
                            remaining = None if deadline is None else deadline - time.monotonic()
                            if remaining is not None and remaining <= 0:
                                break
                            if not selector.select(remaining):
                                break
                            try:
                                offset += os.write(descriptor, payload[offset:])
                            except BlockingIOError:
                                continue
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()
        # Polling at a bounded interval preserves the direct-child timeout while
        # giving long calls resource measurements. Descendants may inherit the
        # output pipes, so completion is still defined by the direct child.
        sample_resources()
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                kill_kind = _terminate_group(pgid, grace_seconds)
                timed_out = True
                process.wait()
                break
            try:
                process.wait(
                    timeout=RESOURCE_SAMPLE_SECONDS
                    if remaining is None
                    else min(RESOURCE_SAMPLE_SECONDS, remaining)
                )
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    kill_kind = _terminate_group(pgid, grace_seconds)
                    timed_out = True
                    process.wait()
                    break
                sample_resources()
        for reader in readers:
            reader.join(DRAIN_SECONDS)
        survivors = census(pgid=pgid)
        reaped = True
        if survivors:
            if kill_kind is None:
                kill_kind = "group-sweep"
            if reaping_enabled():
                signal_group(pgid, signal.SIGKILL)
                time.sleep(SETTLE_SECONDS)
                reaped = not census(pgid=pgid)
                # Killing the holders closes the inherited pipes, so whatever
                # the bounded drain could not collect arrives now.
                for reader in readers:
                    reader.join(DRAIN_SECONDS)
            else:
                reaped = False
        _telemetry.value = GroupTelemetry(
            pgid=pgid,
            kill_kind=kill_kind,
            survivors=survivors,
            survivors_reaped=reaped,
            resource_sample_count=resource_sample_count,
            peak_process_count=peak_process_count,
            peak_rss_kib=peak_rss_kib,
            peak_cpu_percent=peak_cpu_percent,
            resource_observation_error=resource_observation_error,
        )
    except BaseException:
        # Resource observers are outside the harness contract. If one fails,
        # preserve the error but do not leave the process group running.
        if process.poll() is None:
            signal_group(pgid, signal.SIGKILL)
            process.wait()
        raise
    finally:
        _live.discard(pgid)
    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if timed_out:
        assert timeout is not None
        raise subprocess.TimeoutExpired(
            cmd=list(command), timeout=timeout, output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)
