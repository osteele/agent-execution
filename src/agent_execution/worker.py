"""Versioned remote harness execution and ctx evidence export.

The worker owns no review policy or state. It runs one already-resolved harness
command, asks ctx to publish and export that harness's exact native session, and
writes one durable result for the conductor to validate and consume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent_execution import __version__
from agent_execution.command import CommandResult, CommandRunner
from agent_execution.costs import estimate_execution_cost, require_cost_cap, validate_max_cost_usd
from agent_execution.identity import source_sha256
from agent_execution.omp_execution import (
    omp_transcript,
    require_omp_sdk,
    run_omp_command,
    validate_omp_command,
    validate_omp_transcript,
)
from agent_execution.processes import run_in_process_group
from agent_execution.worker_transcript import validate_ctx_transcript

WORKER_IDENTITY_SCHEMA = "agent-execution.worker-identity/v1"
WORKER_RESULT_SCHEMA = "agent-execution.worker-result/v1"
WORKER_RESULT_PATH = Path("outputs/agent-execution-worker-result.json")
WORKER_PROJECT = "agent-execution"
WORKER_PROTOCOL_VERSION = 1
"""Exact compatibility version for the shared worker/conductor contract.

Version 1 retains restricted SDK confinement, prompt/source identity and native
evidence validation from the extracted runtime. Consumers compare for equality.
"""
#: New execution excludes native Codex; historical evidence remains parseable.
SUPPORTED_WORKER_PROVIDERS = frozenset({"omp", "omp-packet"})
#: The binary each provider means. A provider is an adapter variant and need
#: not be a program name: `omp-packet` is OMP under a packet-only tool policy,
#: and the binary is `omp`. The consistency check below compares the command
#: against this rather than against the provider string, because what it exists
#: to prevent is a record naming one harness while another ran -- not a
#: spelling difference between a policy variant and its executable.
_PROVIDER_EXECUTABLES = {"codex": "codex", "omp": "omp", "omp-packet": "omp"}
#: Providers whose transcript is recoverable as a provider-owned session, which
#: is what the session-evidence path reads. Codex writes a `thread.started`
#: event and keeps a session; a packet-only OMP review is dispatched with
#: `--no-session` and has none by construction, so requiring one refused every
#: such review with "Codex output lacks a structured thread.started event" -- a
#: codex-shaped demand made of a harness told not to keep a session.
#:
#: For a session-less provider the harness output IS the evidence: the
#: conductor reconstructs the command result from the recorded stdout and
#: parses the response out of it, exactly as it does locally.
_SESSION_EVIDENCE_PROVIDERS = frozenset({"codex"})
WORKER_STATUSES = frozenset({"completed", "preflight_failed", "harness_failed", "evidence_failed"})

Which = Callable[[str], str | None]
Clock = Callable[[], float]


PromptCommandRunner = Callable[[list[str], Path, str, float | None], CommandResult]


@dataclass(frozen=True)
class WorkerIdentity:
    """Human-readable release identity plus exact installed-byte provenance.

    Compatibility is decided by ``protocol_version``.  ``package_version`` is
    the release humans can search for, while ``source_sha256`` distinguishes
    rebuilds and dirty installs with the same semver.  This repository is jj
    colocated with Git, so the recorded jj commit id is also the Git revision;
    the stable jj change id remains useful when that revision is rewritten.
    """

    package_version: str
    protocol_version: int
    source_sha256: str = ""
    jj_change_id: str = ""
    git_commit_id: str = ""
    dirty: bool | None = None
    schema_version: str = WORKER_IDENTITY_SCHEMA

    @property
    def version_spec(self) -> str:
        fields = [
            f"package={self.package_version}",
            f"protocol={self.protocol_version}",
        ]
        if self.git_commit_id:
            fields.append(f"git={self.git_commit_id}")
        if self.jj_change_id:
            fields.append(f"jj={self.jj_change_id}")
        if self.source_sha256:
            fields.append(f"sha256={self.source_sha256}")
        if self.dirty is not None:
            fields.append(f"dirty={str(self.dirty).lower()}")
        return ";".join(fields)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version_spec": self.version_spec,
            "package_version": self.package_version,
            "protocol_version": self.protocol_version,
            "source_sha256": self.source_sha256,
            "jj_change_id": self.jj_change_id,
            "git_commit_id": self.git_commit_id,
            "dirty": self.dirty,
        }

    @classmethod
    def from_dict(cls, raw: object) -> WorkerIdentity:
        if not isinstance(raw, dict):
            raise ValueError("worker identity must be an object")
        value = cast(dict[str, object], raw)
        if value.get("schema_version") != WORKER_IDENTITY_SCHEMA:
            raise ValueError(f"unsupported worker identity schema: {value.get('schema_version')!r}")
        package_version = value.get("package_version")
        protocol_version = value.get("protocol_version")
        source_sha256 = value.get("source_sha256", "")
        jj_change_id = value.get("jj_change_id", "")
        git_commit_id = value.get("git_commit_id", "")
        dirty = value.get("dirty")
        if not isinstance(package_version, str) or not package_version:
            raise ValueError("worker identity package_version must be a nonempty string")
        if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
            raise ValueError("worker identity protocol_version must be an integer")
        if not isinstance(source_sha256, str):
            raise ValueError("worker identity source_sha256 must be a string")
        if not isinstance(jj_change_id, str):
            raise ValueError("worker identity jj_change_id must be a string")
        if not isinstance(git_commit_id, str):
            raise ValueError("worker identity provenance fields must be strings")
        if source_sha256 and not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise ValueError("worker identity source_sha256 is malformed")
        if dirty is not None and not isinstance(dirty, bool):
            raise ValueError("worker identity dirty must be a boolean or null")
        identity = cls(
            package_version=package_version,
            protocol_version=protocol_version,
            source_sha256=source_sha256,
            jj_change_id=jj_change_id,
            git_commit_id=git_commit_id,
            dirty=dirty,
        )
        if value.get("version_spec") != identity.version_spec:
            raise ValueError("worker identity version_spec disagrees with its component fields")
        return identity


def protocol_mismatch_detail(
    conductor_protocol: int,
    worker_protocol: int,
    *,
    target: str | None = None,
) -> str:
    """Describe a protocol disagreement without recommending a downgrade."""
    mismatch = (
        f"worker protocol mismatch: conductor expects {conductor_protocol}, "
        f"remote worker reports {worker_protocol}"
    )
    if worker_protocol > conductor_protocol:
        return (
            mismatch + "; update or reinstall the local agent-execution consumer; "
            "do not redeploy the newer shared worker from this older checkout"
        )
    deployment = (
        f"install the matching agent-execution package on {target}"
        if target is not None
        else "install the matching agent-execution package on the worker"
    )
    return mismatch + "; " + deployment


def installed_source_sha256() -> str:
    """Return the exact installed shared-package identity, including normal wheels."""
    return source_sha256()


def installed_worker_identity(*, source_sha256: str | None = None) -> WorkerIdentity:
    """Describe this package independently of a consumer's deployment provenance."""
    return WorkerIdentity(
        package_version=__version__,
        protocol_version=WORKER_PROTOCOL_VERSION,
        source_sha256=installed_source_sha256() if source_sha256 is None else source_sha256,
    )


def run_command_with_prompt(
    command: list[str], cwd: Path, prompt: str, timeout: float | None
) -> CommandResult:
    """Run a harness with prompt bytes on stdin and process-group cleanup."""
    if command and Path(command[0]).name == "omp":
        completed = run_omp_command(command, cwd, timeout, prompt=prompt)
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
    completed = run_in_process_group(command, cwd, prompt, timeout)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def run_worker_command(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
    """Run a worker helper with the same process-group cleanup as the harness.

    A timed-out ctx process may leave descendants holding its output pipes. A
    direct-child timeout can then hang while collecting output and keep the
    enclosing Weft slot occupied after the evidence deadline.
    """
    completed = (
        run_omp_command(command, cwd, timeout)
        if command and Path(command[0]).name == "omp"
        else run_in_process_group(command, cwd, "", timeout)
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _text(raw: dict[str, object], name: str, *, allow_empty: bool = False) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise ValueError(f"worker result {name} must be {qualifier}")
    return value


def _number(raw: dict[str, object], name: str) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"worker result {name} must be a number")
    return float(value)


def _integer(raw: dict[str, object], name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"worker result {name} must be an integer")
    return value


@dataclass(frozen=True)
class HarnessOutcome:
    exit_status: int | None
    stdout: str
    stderr: str

    @classmethod
    def from_dict(cls, raw: object) -> HarnessOutcome:
        if not isinstance(raw, dict):
            raise ValueError("worker result harness must be an object")
        value = cast(dict[str, object], raw)
        exit_status = value.get("exit_status")
        if exit_status is not None and (
            isinstance(exit_status, bool) or not isinstance(exit_status, int)
        ):
            raise ValueError("worker result harness exit_status must be an integer or null")
        return cls(
            exit_status=cast(int | None, exit_status),
            stdout=_text(value, "stdout", allow_empty=True),
            stderr=_text(value, "stderr", allow_empty=True),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "exit_status": self.exit_status,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def _validated_ctx_import_receipt(receipt: dict[str, object]) -> dict[str, object]:
    version = receipt.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise ValueError("unsupported ctx import receipt schema")
    if receipt.get("outcome") not in {"success", "completed_with_rejections"}:
        raise ValueError(f"ctx import did not complete: {receipt.get('outcome')!r}")
    return receipt


@dataclass(frozen=True)
class WorkerEvidence:
    import_receipt: dict[str, object]
    transcript: dict[str, object]

    @classmethod
    def from_dict(cls, raw: object) -> WorkerEvidence:
        if not isinstance(raw, dict):
            raise ValueError("worker result evidence must be an object")
        value = cast(dict[str, object], raw)
        receipt = value.get("import_receipt")
        transcript = value.get("transcript")
        if not isinstance(receipt, dict) or not isinstance(transcript, dict):
            raise ValueError("worker evidence must contain object receipt and transcript fields")
        return cls(
            import_receipt=_validated_ctx_import_receipt(cast(dict[str, object], receipt)),
            transcript=cast(dict[str, object], transcript),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "import_receipt": self.import_receipt,
            "transcript": self.transcript,
        }


@dataclass(frozen=True)
class WorkerResult:
    model_call_id: str
    provider: str
    status: str
    worker_version: str
    worker_protocol_version: int
    ctx_version: str
    worker_cwd: str
    started_at: float
    completed_at: float
    model_call_started: bool
    session_id: str
    harness: HarnessOutcome
    evidence: WorkerEvidence | None = None
    failure: str = ""
    worker_source_sha256: str = ""
    prompt_sha256: str = ""
    worker_identity: WorkerIdentity | None = None
    #: The harness budget this run was given, so a later reader can tell which
    #: limit fired. A `harness-timeout` with no budget recorded is unreadable:
    #: the frozen policy, the worker's own default, and the job's ceiling are
    #: three different numbers, and the failure text names none of them.
    harness_timeout_seconds: float | None = None
    #: When the harness process actually started, which makes environment setup
    #: a measured interval rather than a gap inferred from two other numbers.
    #: The budget governs the harness call, so time spent before it is bounded
    #: by nothing here - a job can burn hours in dependency installation and
    #: exceed a budget that never began. Absent means unrecorded.
    harness_started_at: float | None = None
    schema_version: str = WORKER_RESULT_SCHEMA
    omp_evidence: dict[str, object] | None = None

    @classmethod
    def parse(cls, text: str) -> WorkerResult:
        try:
            decoded = cast(object, json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError("worker result is not JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError("worker result must be an object")
        raw = cast(dict[str, object], decoded)
        if raw.get("schema_version") != WORKER_RESULT_SCHEMA:
            raise ValueError(f"unsupported worker result schema: {raw.get('schema_version')!r}")
        status = _text(raw, "status")
        if status not in WORKER_STATUSES:
            raise ValueError(f"unknown worker result status: {status}")
        started_at = _number(raw, "started_at")
        completed_at = _number(raw, "completed_at")
        if completed_at < started_at:
            raise ValueError("worker result completed_at precedes started_at")
        model_call_started = raw.get("model_call_started")
        if not isinstance(model_call_started, bool):
            raise ValueError("worker result model_call_started must be a boolean")
        evidence_raw = raw.get("evidence")
        evidence = WorkerEvidence.from_dict(evidence_raw) if evidence_raw is not None else None
        identity_raw = raw.get("worker_identity")
        identity = WorkerIdentity.from_dict(identity_raw) if identity_raw is not None else None
        worker_protocol_version = _integer(raw, "worker_protocol_version")
        result = cls(
            model_call_id=_text(raw, "model_call_id"),
            provider=_text(raw, "provider"),
            status=status,
            worker_version=_text(raw, "worker_version"),
            worker_protocol_version=worker_protocol_version,
            worker_source_sha256=_text(raw, "worker_source_sha256", allow_empty=True)
            if "worker_source_sha256" in raw
            else "",
            prompt_sha256=_text(raw, "prompt_sha256", allow_empty=True)
            if "prompt_sha256" in raw
            else "",
            ctx_version=_text(raw, "ctx_version", allow_empty=True),
            worker_cwd=_text(raw, "worker_cwd"),
            started_at=started_at,
            completed_at=completed_at,
            model_call_started=model_call_started,
            session_id=_text(raw, "session_id", allow_empty=True),
            harness=HarnessOutcome.from_dict(raw.get("harness")),
            evidence=evidence,
            failure=_text(raw, "failure", allow_empty=True),
            worker_identity=identity,
            harness_timeout_seconds=(
                float(cast(int | float, raw["harness_timeout_seconds"]))
                if isinstance(raw.get("harness_timeout_seconds"), (int, float))
                and not isinstance(raw.get("harness_timeout_seconds"), bool)
                else None
            ),
            harness_started_at=(
                float(cast(int | float, raw["harness_started_at"]))
                if isinstance(raw.get("harness_started_at"), (int, float))
                and not isinstance(raw.get("harness_started_at"), bool)
                else None
            ),
            omp_evidence=(
                cast(dict[str, object], raw["omp_evidence"])
                if isinstance(raw.get("omp_evidence"), dict)
                else None
            ),
        )
        if "omp_evidence" in raw and not isinstance(raw["omp_evidence"], dict):
            raise ValueError("OMP evidence must be an object")
        if identity is not None and (
            identity.package_version != result.worker_version
            or identity.protocol_version != result.worker_protocol_version
            or identity.source_sha256 != result.worker_source_sha256
        ):
            raise ValueError("worker identity disagrees with legacy worker identity fields")
        # What a completed result must carry depends on whether the provider
        # keeps a session. Codex does, and its transcript is the evidence. A
        # packet-only OMP review is dispatched with `--no-session`, so there is
        # no session to import and no transcript to hold -- the harness output
        # is the evidence, and the conductor parses the response from it.
        #
        # The absence is asserted rather than tolerated: a session-less result
        # carrying evidence or a session id would mean the provider table and
        # the worker disagree about what ran, which is the one thing this
        # schema exists to catch.
        session_bearing = result.provider in _SESSION_EVIDENCE_PROVIDERS
        if status == "completed":
            if result.harness.exit_status != 0:
                raise ValueError("completed worker result reports a failed harness")
            if not model_call_started:
                raise ValueError("completed worker result denies that the model call started")
            if result.provider == "omp":
                if result.omp_evidence is None:
                    raise ValueError("grounded OMP result lacks native evidence")
                transcript = validate_omp_transcript(
                    result.omp_evidence,
                    policy="read-only-no-shell",
                    cwd=result.worker_cwd,
                    prompt_sha256=result.prompt_sha256 or None,
                )
                header = cast(dict[str, object], transcript["header"])
                if result.session_id != header["session_id"]:
                    raise ValueError("OMP result session differs from native evidence")
                if transcript != omp_transcript(result.harness.stdout):
                    raise ValueError("OMP result evidence differs from harness stream")
                if evidence is not None:
                    raise ValueError("OMP result cannot carry Codex/ctx evidence")
            elif session_bearing:
                if evidence is None:
                    raise ValueError("completed worker result lacks successful harness evidence")
                if not result.session_id or not result.ctx_version:
                    raise ValueError("completed worker result lacks its model or ctx identity")
            elif not result.harness.stdout.strip():
                # For a session-less provider the output IS the evidence, so an
                # empty one is not a completed review. Three concurrent
                # offloaded reviews returned exactly this -- exit 0, no output,
                # `completed` -- because they shared one artifact path, and
                # nothing in the schema objected.
                raise ValueError(
                    f"{result.provider} keeps no session, so a completed result "
                    "must carry harness output"
                )
            else:
                if evidence is not None:
                    raise ValueError(
                        f"{result.provider} keeps no session, so its result must carry no evidence"
                    )
                if result.session_id:
                    raise ValueError(
                        f"{result.provider} keeps no session, so its result must name none"
                    )
        if status == "preflight_failed" and model_call_started:
            raise ValueError("preflight failure claims that the model call started")
        if evidence is not None and not result.session_id:
            raise ValueError("worker evidence lacks its provider session identity")
        if status != "completed" and not result.failure:
            raise ValueError("failed worker result lacks a diagnostic")
        if result.omp_evidence is not None and result.provider != "omp":
            raise ValueError("non-OMP result carries OMP evidence")
        if result.provider == "omp-packet" and status == "completed":
            omp_transcript(
                result.harness.stdout,
                policy="packet-only-no-tools",
                cwd=result.worker_cwd,
                prompt_sha256=result.prompt_sha256 or None,
            )
        return result

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_call_id": self.model_call_id,
            "provider": self.provider,
            "status": self.status,
            "worker_version": self.worker_version,
            "worker_protocol_version": self.worker_protocol_version,
            "worker_source_sha256": self.worker_source_sha256,
            "prompt_sha256": self.prompt_sha256,
            "ctx_version": self.ctx_version,
            "worker_cwd": self.worker_cwd,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "model_call_started": self.model_call_started,
            "session_id": self.session_id,
            "harness": self.harness.to_dict(),
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
            "failure": self.failure,
            "harness_timeout_seconds": self.harness_timeout_seconds,
            "harness_started_at": self.harness_started_at,
        }
        if self.omp_evidence is not None:
            payload["omp_evidence"] = self.omp_evidence
        if self.worker_identity is not None:
            payload["worker_identity"] = self.worker_identity.to_dict()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def summary(self, *, artifact_path: str = str(WORKER_RESULT_PATH)) -> dict[str, object]:
        encoded = self.to_json().encode()
        summary: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_call_id": self.model_call_id,
            "provider": self.provider,
            "status": self.status,
            "worker_version": self.worker_version,
            "worker_protocol_version": self.worker_protocol_version,
            "worker_source_sha256": self.worker_source_sha256,
            "prompt_sha256": self.prompt_sha256,
            "ctx_version": self.ctx_version,
            "session_id": self.session_id,
            "model_call_started": self.model_call_started,
            "artifact_path": artifact_path,
            "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
            "failure": self.failure,
            "harness_timeout_seconds": self.harness_timeout_seconds,
            "harness_started_at": self.harness_started_at,
        }
        if self.worker_identity is not None:
            summary["worker_identity"] = self.worker_identity.to_dict()
        return summary


def _json_object(result: CommandResult, *, label: str) -> dict[str, object]:
    if result.exit_status != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise ValueError(f"{label} failed: {detail[-1] if detail else 'no output'}")
    try:
        decoded = cast(object, json.loads(result.stdout))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} did not return JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must return an object")
    return cast(dict[str, object], decoded)


def codex_session_id(output: str) -> str | None:
    """Read the provider-owned ID only from Codex's structured start event."""
    for line in output.splitlines():
        try:
            decoded = cast(object, json.loads(line))
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        event = cast(dict[str, object], decoded)
        if event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id
    return None


def command_failure_detail(result: CommandResult) -> str:
    """Prefer a provider's structured failure over incidental terminal output."""
    for line in reversed(result.stdout.splitlines()):
        try:
            decoded = cast(object, json.loads(line))
        except json.JSONDecodeError:
            continue
        if not isinstance(decoded, dict):
            continue
        event = cast(dict[str, object], decoded)
        if event.get("type") not in {"error", "turn.failed"}:
            continue
        message = event.get("message")
        error = event.get("error")
        if isinstance(message, str) and message:
            return message
        if isinstance(error, dict):
            nested_message = cast(dict[str, object], error).get("message")
            if isinstance(nested_message, str) and nested_message:
                return nested_message
    detail = (result.stderr or result.stdout).strip().splitlines()
    return detail[-1] if detail else "no output"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_result(path: Path, result: WorkerResult, *, cwd: Path) -> None:
    """Atomically publish a result and sync its file and directory entries."""
    root = cwd.resolve()
    destination = path.resolve() if path.is_absolute() else (root / path).resolve()
    if destination == root or root not in destination.parents:
        raise ValueError("worker evidence output must stay inside its working directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(result.to_json())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = destination.parent
        while True:
            _fsync_directory(directory)
            if directory == root:
                break
            directory = directory.parent
    finally:
        if temporary.exists():
            temporary.unlink()


def _decode_stream(raw: object) -> str:
    """Text from whatever a killed subprocess left on a captured stream.

    `subprocess.TimeoutExpired` carries bytes when the stream was captured in
    binary mode, `str` under `text=True`, and `None` when nothing was
    captured. All three reach here, and none of them may raise: this runs on a
    path whose whole purpose is to preserve an explanation.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _last_line(text: str) -> str:
    """The last non-blank line, which is where a CLI puts its terminal event.

    Both streams are read, in stdout-then-stderr order, because which one
    carries the cause is a property of the invocation rather than of the tool.
    Measured on codex the same evening, and the two readings disagree:

    - under ``--json``, which is what this project dispatches: stdout carried
      733 bytes ending in ``{"type":"error","message":"You've hit your usage
      limit ..."}`` while stderr carried 39 bytes of
      ``Reading additional input from stdin...`` -- an informational banner
      printed even with stdin closed.
    - without ``--json``, a peer measured the reverse: an empty stdout and the
      human-readable error on stderr.

    So a rule naming one stream is wrong half the time. "Capture stderr" would
    have recorded the 39-byte banner and dropped the deciding bytes for every
    dispatch this project makes; "capture stdout" fails the other invocation.
    Read both and take the last non-blank line, which is terminal in either.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def execute_worker(
    *,
    provider: str,
    model_call_id: str,
    command: list[str],
    output: Path,
    timeout: float | None,
    ctx_timeout: float,
    expect_protocol: int | None = None,
    expect_source_sha256: str | None = None,
    prompt_payload: str | None = None,
    expect_prompt_sha256: str | None = None,
    harness_model: str | None = None,
    max_cost_usd: float | None = None,
    invoke: CommandRunner = run_worker_command,
    invoke_prompt: PromptCommandRunner = run_command_with_prompt,
    which: Which = shutil.which,
    clock: Clock = time.time,
    cwd: Path | None = None,
) -> WorkerResult:
    """Run one harness and export its exact ctx session into ``output``."""
    started_at = clock()
    working_directory = (cwd or Path.cwd()).resolve()
    empty_harness = HarnessOutcome(None, "", "")
    worker_source_sha256 = installed_source_sha256()
    worker_identity = installed_worker_identity(source_sha256=worker_source_sha256)
    prompt = ""
    prompt_sha256 = ""
    harness_started_at: float | None = None

    def finish(
        status: str,
        *,
        harness: HarnessOutcome = empty_harness,
        ctx_version: str = "",
        session_id: str = "",
        evidence: WorkerEvidence | None = None,
        failure: str = "",
        model_call_started: bool = False,
        omp_evidence: dict[str, object] | None = None,
    ) -> WorkerResult:
        result = WorkerResult(
            model_call_id=model_call_id,
            provider=provider,
            status=status,
            worker_version=__version__,
            worker_protocol_version=WORKER_PROTOCOL_VERSION,
            worker_source_sha256=worker_source_sha256,
            prompt_sha256=prompt_sha256,
            ctx_version=ctx_version,
            worker_cwd=str(working_directory),
            started_at=started_at,
            completed_at=clock(),
            model_call_started=model_call_started,
            session_id=session_id,
            harness=harness,
            evidence=evidence,
            failure=failure,
            worker_identity=worker_identity,
            harness_timeout_seconds=timeout,
            harness_started_at=harness_started_at,
            omp_evidence=omp_evidence,
        )
        _write_result(output, result, cwd=working_directory)
        return result

    if expect_protocol is not None and expect_protocol != WORKER_PROTOCOL_VERSION:
        return finish(
            "preflight_failed",
            failure=protocol_mismatch_detail(expect_protocol, WORKER_PROTOCOL_VERSION),
        )
    if expect_source_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expect_source_sha256):
            return finish("preflight_failed", failure="expected worker source digest is malformed")
        if not worker_source_sha256:
            return finish(
                "preflight_failed",
                failure="worker package source identity is unavailable",
            )
        if worker_source_sha256 != expect_source_sha256:
            return finish(
                "preflight_failed",
                failure=(
                    "worker source mismatch: conductor expects "
                    f"{expect_source_sha256}, worker provides {worker_source_sha256}; "
                    "install the matching agent-execution package on the remote worker"
                ),
            )
    if prompt_payload is not None:
        if Path(prompt_payload).name != prompt_payload or prompt_payload in {".", ".."}:
            return finish("preflight_failed", failure="worker prompt payload name is unsafe")
        payload_root = os.environ.get("WEFT_PAYLOAD_DIR", "")
        if not payload_root:
            return finish("preflight_failed", failure="WEFT_PAYLOAD_DIR is not set")
        payload_path = Path(payload_root) / prompt_payload
        try:
            prompt_bytes = payload_path.read_bytes()
            prompt = prompt_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return finish(
                "preflight_failed",
                failure=f"could not read prompt payload {prompt_payload}: {error}",
            )
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        if expect_prompt_sha256 is not None and prompt_sha256 != expect_prompt_sha256:
            return finish(
                "preflight_failed",
                failure=(
                    "prompt payload digest mismatch: conductor expects "
                    f"{expect_prompt_sha256}, worker read {prompt_sha256}"
                ),
            )
    elif expect_prompt_sha256 is not None:
        return finish("preflight_failed", failure="expected prompt digest lacks a payload")
    if provider not in SUPPORTED_WORKER_PROVIDERS:
        return finish("preflight_failed", failure=f"unsupported worker provider: {provider}")
    if not command:
        return finish("preflight_failed", failure="worker harness command is empty")
    expected_executable = _PROVIDER_EXECUTABLES[provider]
    if Path(command[0]).name != expected_executable:
        return finish(
            "preflight_failed",
            failure=(
                f"worker provider {provider} runs {expected_executable}, "
                f"but the command is {command[0]}"
            ),
        )
    ctx = which("ctx")
    executable = which(command[0])
    if ctx is None and provider == "codex":
        return finish("preflight_failed", failure="ctx is not installed on the worker")
    if executable is None and provider == "codex":
        return finish("preflight_failed", failure=f"{provider} is not installed on the worker")
    omp_invocation = None
    if provider in {"omp", "omp-packet"}:
        try:
            logical_command = [*command, *(["--model", harness_model] if harness_model else [])]
            omp_invocation = validate_omp_command(logical_command, working_directory)
            expected_policy = "read-only-no-shell" if provider == "omp" else "packet-only-no-tools"
            if omp_invocation.policy != expected_policy:
                raise ValueError("OMP adapter identity disagrees with command tool policy")
            require_omp_sdk()
        except ValueError as error:
            return finish("preflight_failed", failure=str(error))
        executable = "omp"
    try:
        max_cost_usd = validate_max_cost_usd(max_cost_usd)
        if max_cost_usd is not None:
            # The actual selector is authoritative, including inline commands.
            model = omp_invocation.selector if omp_invocation is not None else (harness_model or "")
            route = model.partition("/")[0] if provider in {"omp", "omp-packet"} else provider
            profile_name = ""
            if "--profile" in command:
                profile_index = command.index("--profile") + 1
                if profile_index == len(command):
                    raise ValueError("--profile has no value")
                profile_name = command[profile_index]
            estimate = estimate_execution_cost(
                {
                    "adapter": provider,
                    "provider": route,
                    "model": model,
                    "model_argument": model,
                    "profile": profile_name,
                    "worst_case_cost": 0.0,
                },
                cwd=working_directory,
                refresh=True,
            )
            require_cost_cap(estimate, max_cost_usd)
    except ValueError as error:
        return finish("preflight_failed", failure=f"cost-cap-refused: {error}")
    ctx_version = ""
    if provider == "codex":
        try:
            version_result = invoke(
                [cast(str, ctx), "--version"], working_directory, min(ctx_timeout, 10.0)
            )
        except subprocess.TimeoutExpired:
            return finish("preflight_failed", failure="ctx version check timed out")
        except (FileNotFoundError, PermissionError) as error:
            return finish("preflight_failed", failure=f"ctx version check failed: {error}")
        if version_result.exit_status != 0:
            detail = (version_result.stderr or version_result.stdout).strip()
            return finish("preflight_failed", failure=f"ctx version check failed: {detail}")
        ctx_version = (version_result.stdout or version_result.stderr).strip()

    try:
        # `--model` is spliced here rather than carried in the submitted
        # command line. Weft scans a job's command for `--model`,
        # `--model-name` and `--base-model` and stages the value as a
        # HuggingFace repo (`internal/dataloc/pyscan.go`), so a GLM selector
        # became `download hf:zai/glm-5.3-flash`, which does not exist and
        # blocked the job before it ran. The selector is a provider/model pair
        # for the harness, not a dataset reference, and the only way to say so
        # is to keep it out of the text weft reads.
        harness_command = [cast(str, executable), *command[1:]]
        if harness_model:
            harness_command.extend(["--model", harness_model])
        harness_started_at = clock()
        command_result = (
            invoke_prompt(harness_command, working_directory, prompt, timeout)
            if prompt_payload is not None
            else invoke(harness_command, working_directory, timeout)
        )
    except subprocess.TimeoutExpired as error:
        # A timed-out harness has usually already said why it stopped making
        # progress, and that explanation is on the exception rather than lost:
        # `subprocess.run` attaches whatever it captured before the deadline.
        # Discarding it recorded `harness-timeout` for a codex run whose output
        # held "You've hit your usage limit", so the failure was unattributable,
        # matched no quota marker, and rested no billing account -- automatic
        # selection then kept choosing an account with no headroom.
        salvaged_stdout = _decode_stream(error.stdout)
        salvaged_stderr = _decode_stream(error.stderr)
        reported = _last_line(salvaged_stdout) or _last_line(salvaged_stderr)
        failure = "harness-timeout"
        if reported:
            failure = f"{failure}; provider reported: {reported}"
        return finish(
            "harness_failed",
            ctx_version=ctx_version,
            harness=HarnessOutcome(
                124,
                salvaged_stdout,
                salvaged_stderr or f"harness exceeded {timeout}s",
            ),
            failure=failure,
            model_call_started=True,
        )
    except (FileNotFoundError, PermissionError) as error:
        return finish(
            "preflight_failed",
            ctx_version=ctx_version,
            failure=f"could not start {provider}: {error}",
        )
    harness = HarnessOutcome(
        command_result.exit_status, command_result.stdout, command_result.stderr
    )

    def evidence_failure(detail: str) -> str:
        # The export is secondary evidence. A failed harness remains the cause
        # even when its session was never persisted (for example, oversized input).
        if command_result.exit_status:
            return (
                f"{provider} exited {command_result.exit_status}: "
                f"{command_failure_detail(command_result)}; evidence collection failed: {detail}"
            )
        return detail

    if omp_invocation is not None:
        if command_result.exit_status != 0:
            return finish(
                "harness_failed",
                harness=harness,
                model_call_started=True,
                failure=f"{provider} failed: {command_failure_detail(command_result)}",
            )
        try:
            transcript = omp_transcript(
                command_result.stdout,
                selector=omp_invocation.selector,
                policy=omp_invocation.policy,
                cwd=str(working_directory),
                prompt_sha256=prompt_sha256
                or hashlib.sha256(omp_invocation.prompt.encode()).hexdigest(),
            )
        except ValueError as error:
            return finish(
                "evidence_failed",
                harness=harness,
                model_call_started=True,
                failure=f"invalid OMP execution evidence: {error}",
            )
        return finish(
            "completed",
            harness=harness,
            model_call_started=True,
            session_id=str(cast(dict[str, object], transcript["header"])["session_id"])
            if provider == "omp"
            else "",
            omp_evidence=transcript if provider == "omp" else None,
        )
    if provider not in _SESSION_EVIDENCE_PROVIDERS:
        if command_result.exit_status != 0:
            return finish(
                "harness_failed",
                ctx_version=ctx_version,
                harness=harness,
                failure=(
                    f"{provider} exited {command_result.exit_status}: "
                    f"{command_failure_detail(command_result)}"
                ),
                model_call_started=True,
            )
        return finish(
            "completed",
            ctx_version=ctx_version,
            harness=harness,
            model_call_started=True,
        )

    assert ctx is not None

    session_id = codex_session_id(command_result.stdout)
    if session_id is None:
        if command_result.exit_status:
            return finish(
                "preflight_failed",
                ctx_version=ctx_version,
                harness=harness,
                failure=(
                    f"{provider} exited {command_result.exit_status}: "
                    f"{command_failure_detail(command_result)}"
                ),
                model_call_started=False,
            )
        return finish(
            "evidence_failed",
            ctx_version=ctx_version,
            harness=harness,
            failure="Codex output lacks a structured thread.started event",
            model_call_started=True,
        )

    try:
        imported = _validated_ctx_import_receipt(
            _json_object(
                invoke(
                    [
                        ctx,
                        "import",
                        "--provider",
                        provider,
                        "--format",
                        "json",
                        "--progress",
                        "none",
                    ],
                    working_directory,
                    ctx_timeout,
                ),
                label="ctx import",
            )
        )
        transcript = _json_object(
            invoke(
                [
                    ctx,
                    "show",
                    "session",
                    "--provider",
                    provider,
                    "--provider-session",
                    session_id,
                    "--mode",
                    "log",
                    "--format",
                    "json",
                ],
                working_directory,
                ctx_timeout,
            ),
            label="ctx session export",
        )
        # Validate the complete public ctx contract before making it durable.
        validate_ctx_transcript(
            transcript,
            provider=provider,
            session_id=session_id,
            remote_cwd=str(working_directory),
        )
        evidence = WorkerEvidence(import_receipt=imported, transcript=transcript)
    except subprocess.TimeoutExpired:
        status = "harness_failed" if command_result.exit_status else "evidence_failed"
        return finish(
            status,
            ctx_version=ctx_version,
            session_id=session_id,
            harness=harness,
            failure=evidence_failure(f"ctx evidence collection exceeded {ctx_timeout}s"),
            model_call_started=True,
        )
    except (OSError, ValueError) as error:
        status = "harness_failed" if command_result.exit_status else "evidence_failed"
        return finish(
            status,
            ctx_version=ctx_version,
            session_id=session_id,
            harness=harness,
            failure=evidence_failure(str(error)),
            model_call_started=True,
        )

    if command_result.exit_status != 0:
        return finish(
            "harness_failed",
            ctx_version=ctx_version,
            session_id=session_id,
            harness=harness,
            evidence=evidence,
            failure=(
                f"{provider} exited {command_result.exit_status}: "
                f"{command_failure_detail(command_result)}"
            ),
            model_call_started=True,
        )
    return finish(
        "completed",
        ctx_version=ctx_version,
        session_id=session_id,
        harness=harness,
        evidence=evidence,
        model_call_started=True,
    )
