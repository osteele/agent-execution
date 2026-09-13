"""Atomic Weft admission, durable result retrieval, and explicit consumption.

Only a validated not_accepted receipt permits local fallback. Missing observations
never establish that an accepted execution failed or that an unknown one is safe
to submit again. Product state and result interpretation belong to the consumer.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from agent_execution.command import CommandResult, CommandRunner, ProgressCallback, run_command
from agent_execution.costs import validate_max_cost_usd
from agent_execution.identity import source_sha256
from agent_execution.omp_execution import omp_transcript, validate_omp_command
from agent_execution.weft_protocol import WEFT_HOST_LIST_COMMAND, parse_weft_host_capabilities
from agent_execution.worker import (
    WORKER_PROJECT,
    WORKER_PROTOCOL_VERSION,
    WorkerResult,
    protocol_mismatch_detail,
)

WEFT_RECEIPT_VERSION = "weft.run.receipt.v1"
WEFT_JOB_LIST_VERSION = 1
WEFT_HOST_OBSERVATION_TIMEOUT = 10.0
WEFT_INVENTORY_TAG = "inventory"
WEFT_AUTO_HOST = "auto"
WEFT_TOOL_CAPABILITY = "tool:agent-execution"
WORKER_PROMPT_PAYLOAD = "execution-prompt"
WEFT_LIVE_JOB_STATUSES = frozenset({"draft", "queued", "pending_placement", "starting", "running"})
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
InterruptibleWait = Callable[[float], bool]
ReadinessObservation = Callable[[str], tuple[str, str]]
ARTIFACT_RETRY_DELAYS = (0.5, 1.0, 2.0)
ADMISSION_RETRY_DELAYS = (1.0, 2.0)
LOST_OBSERVATION_PROBE_SECONDS = 30.0
REMOTE_OBSERVATION_SECONDS = 900.0
DIAGNOSTIC_TAIL_LINES = 200


def harness_capability_name(provider: str) -> str:
    """Worker variants use the same installed physical harness."""
    return "omp" if provider == "omp-packet" else provider


def _tail(text: str) -> str:
    """The last lines of a stream, for a diagnostic that must stay bounded."""
    return "\n".join(text.strip().splitlines()[-DIAGNOSTIC_TAIL_LINES:])


def uninterrupted_wait(seconds: float) -> bool:
    time.sleep(seconds)
    return False


@dataclass(frozen=True)
class WeftRunReceipt:
    """The normalized subset of Weft's versioned submission receipt."""

    job_id: str
    placement_decision: str
    selected_host: str
    source_pin: str
    accepted_immediately: bool
    deduplicated: bool
    idempotency_key: str

    @classmethod
    def parse(cls, text: str, *, idempotency_key: str) -> WeftRunReceipt:
        try:
            value = cast(object, json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError("Weft submission did not return JSON") from error
        if not isinstance(value, dict):
            raise ValueError("Weft submission receipt must be an object")
        raw = cast(dict[str, object], value)

        def text_field(name: str) -> str:
            field = raw.get(name, "")
            if not isinstance(field, str):
                raise ValueError(f"Weft receipt {name} must be a string")
            return field

        if raw.get("api_version") != WEFT_RECEIPT_VERSION:
            raise ValueError(f"unsupported Weft submission receipt: {raw.get('api_version')!r}")
        decision = raw.get("placement_decision")
        if not isinstance(decision, str) or decision not in {
            "accepted_immediately",
            "deduplicated",
            "not_accepted",
        }:
            raise ValueError(f"invalid Weft placement decision: {decision!r}")
        accepted = raw.get("accepted_immediately")
        if not isinstance(accepted, bool):
            raise ValueError("Weft receipt accepted_immediately must be a boolean")
        deduplicated = raw.get("deduplicated", False)
        if not isinstance(deduplicated, bool):
            raise ValueError("Weft receipt deduplicated must be a boolean")
        received_key = text_field("idempotency_key")
        if received_key != idempotency_key:
            raise ValueError("Weft receipt did not echo the assignment idempotency key")
        receipt = cls(
            job_id=text_field("job_id"),
            placement_decision=str(decision),
            selected_host=text_field("selected_host"),
            source_pin=text_field("source_pin"),
            accepted_immediately=accepted,
            deduplicated=deduplicated,
            idempotency_key=received_key,
        )
        if decision == "not_accepted":
            if receipt.job_id or accepted or deduplicated:
                raise ValueError("not_accepted Weft receipt claims a durable job")
            return receipt
        if not receipt.job_id or not receipt.source_pin:
            raise ValueError("accepted Weft receipt lacks a job ID or source pin")
        if decision == "accepted_immediately" and not accepted:
            raise ValueError("accepted_immediately Weft receipt denies immediate acceptance")
        if decision == "deduplicated" and not deduplicated:
            raise ValueError("deduplicated Weft receipt lacks its deduplication flag")
        return receipt

    def to_dict(self) -> dict[str, object]:
        return {
            "api_version": WEFT_RECEIPT_VERSION,
            "job_id": self.job_id,
            "placement_decision": self.placement_decision,
            "selected_host": self.selected_host,
            "source_pin": self.source_pin,
            "accepted_immediately": self.accepted_immediately,
            "deduplicated": self.deduplicated,
            "idempotency_key": self.idempotency_key,
        }


class WeftExecutionAmbiguous(TimeoutError):
    """Weft may own the assignment, so executing a local fallback is unsafe."""


class WeftExecutionDetached(TimeoutError):
    """A durable Weft job outlived the local waiter and must be retrieved."""


class WeftAdmissionCancelled(RuntimeError):
    """The enclosing fan-out ended while remote admission was backing off."""


class WeftPlacementRefused(RuntimeError):
    """No declared Weft host can satisfy the dispatch requirements."""


@dataclass(frozen=True)
class WeftJobFailure:
    """A durable job is proven failed and has no worker-result artifact."""

    job_id: str
    detail: str


@dataclass(frozen=True)
class WeftRetrievalOutcome:
    """A worker artifact was absent and the job observation explains what is known."""

    status: Literal["pending", "unretrievable", "unknown"]
    detail: str
    job_status: str | None = None


def parse_worker_command(command: object) -> dict[str, str]:
    """Parse the single worker argv emitted by _remote_command."""
    if not isinstance(command, str):
        raise ValueError("accepted command is missing")
    # Reject shell composition instead of attributing another invocation's flags.
    if any(part in command for part in ("\n", "\r", "$", "`")):
        raise ValueError("accepted command contains shell composition")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    argv = list(lexer)
    if any(
        token.startswith("#") or (token and all(char in "();<>|&" for char in token))
        for token in argv
    ):
        raise ValueError("accepted command contains shell composition")
    if argv[:2] != ["agent-execution-worker", "execute"] or "--" not in argv:
        raise ValueError("accepted command is not a worker execution")
    worker_args = argv[2 : argv.index("--")]
    # Only flag/value pairs are emitted; values cannot impersonate worker flags.
    allowed_flags = {
        "--provider",
        "--model-call-id",
        "--harness-model",
        "--max-cost-usd",
        "--expect-protocol",
        "--expect-source-sha256",
        "--prompt-payload",
        "--expect-prompt-sha256",
        "--evidence-out",
        "--timeout",
        "--ctx-timeout",
    }
    if len(worker_args) % 2:
        raise ValueError("accepted command has ambiguous worker arguments")
    options: dict[str, str] = {}
    for index in range(0, len(worker_args), 2):
        flag, value = worker_args[index : index + 2]
        if flag not in allowed_flags or flag in options or not value or value.startswith("--"):
            raise ValueError("accepted command has ambiguous worker arguments")
        options[flag] = value
    return options


def submitted_job_from_listing(raw: str, *, model_call_id: str) -> str | None:
    """Return the one Weft job whose recorded dispatch names this model call."""
    try:
        parsed = cast(object, json.loads(raw or "[]"))
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        envelope = cast(dict[str, object], parsed)
        version = envelope.get("version")
        if (
            envelope.get("kind") != "job_list"
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version != WEFT_JOB_LIST_VERSION
        ):
            return None
        rows = envelope.get("jobs")
    else:
        # Older Weft releases returned the rows without an envelope.
        rows = parsed
    if not isinstance(rows, list):
        return None
    matches: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry = cast(dict[str, object], row)
        try:
            options = parse_worker_command(entry.get("command"))
        except ValueError:
            continue
        job_id = entry.get("job_id")
        if isinstance(job_id, str) and job_id and options.get("--model-call-id") == model_call_id:
            matches.add(job_id)
    return next(iter(matches)) if len(matches) == 1 else None


def probe_submitted_job(
    *,
    model_call_id: str,
    cwd: Path,
    timeout: float = LOST_OBSERVATION_PROBE_SECONDS,
    invoke: CommandRunner = run_command,
    executable: str = "weft",
) -> str | None:
    """Find one exact idempotent Weft submission through its versioned job list."""
    try:
        listing = invoke(
            [
                executable,
                "list",
                "jobs",
                "--filter",
                model_call_id,
                "--format",
                "json",
                "--all",
            ],
            cwd,
            timeout,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if listing.exit_status != 0:
        return None
    return submitted_job_from_listing(listing.stdout, model_call_id=model_call_id)


class WeftCommandRunner:
    """Run one harness through atomic Weft admission, else run it locally."""

    def __init__(
        self,
        *,
        host: str,
        agent: str,
        model_call_id: str,
        fallback: CommandRunner | None,
        invoke: CommandRunner = run_command,
        clock: Clock = time.monotonic,
        sleep: Sleeper = time.sleep,
        admission_wait: InterruptibleWait = uninterrupted_wait,
        executable: str = "weft",
        progress: ProgressCallback | None = None,
        expected_source_sha256: str | None = None,
        owner_description: str | None = None,
        submitter_session: str | None = None,
        readiness: ReadinessObservation | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        self.host = host
        self.agent = agent
        self.model_call_id = model_call_id
        self.fallback = fallback
        self.invoke = invoke
        self.clock = clock
        self.sleep = sleep
        self.admission_wait = admission_wait
        self.executable = executable
        self.progress = progress
        self.expected_source_sha256 = expected_source_sha256 or source_sha256()
        self.owner_description = owner_description or f"agent-execution model call {model_call_id}"
        self.readiness = readiness

        self.max_cost_usd = validate_max_cost_usd(max_cost_usd)
        self.prompt_sha256: str | None = None
        self.omp_selector: str | None = None
        self.last_execution: dict[str, object] | None = None
        self.submitter_session = submitter_session or f"agent-execution/v1/{model_call_id}"

    def _progress(self, state: str, **fields: object) -> None:
        if self.progress is not None:
            self.progress("transport_progress", {"state": state, **fields})

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise WeftExecutionAmbiguous(
                f"Weft dispatch of model call {self.model_call_id} exceeded its deadline"
            )
        return remaining

    def _call(self, argv: list[str], cwd: Path, deadline: float) -> CommandResult:
        try:
            return self.invoke(argv, cwd, self._remaining(deadline))
        except subprocess.TimeoutExpired as error:
            reference = str(self.last_execution.get("job_id", "")) if self.last_execution else ""
            if (
                self.last_execution is not None
                and self.last_execution.get("transport") == "weft"
                and reference
            ):
                raise WeftExecutionDetached(
                    f"Weft job {reference} outlived its local waiter"
                ) from error
            detail = f" for {reference}" if reference else " before a receipt was validated"
            raise WeftExecutionAmbiguous(f"Weft timed out{detail}") from error

    @staticmethod
    def _prompt_index(command: list[str]) -> int:
        """Where the brief sits in a harness command line.

        Two callers need this and must agree: one hashes the prompt to pin what
        the worker will be fed, and one replaces it with `-` so the prompt
        travels as a Weft payload instead of as an argument. They disagreed
        before -- the hash took `command[-1]` while the rewrite asserted `codex
        exec` -- so OMP dispatches were refused outright, and had the refusal
        been lifted alone the hash would have pinned a trailing flag as the
        brief.

        The two supported shapes, both verified to read a prompt from stdin
        under `-`:

            codex exec [flags] <prompt>    -- prompt last
            omp -p <prompt> [flags]        -- prompt follows `-p`

        A shape this does not recognize is refused rather than guessed at
        positionally.
        """
        if command[:2] == ["codex", "exec"] and len(command) >= 3:
            return len(command) - 1
        if "-p" in command:
            index = command.index("-p")
            if index + 1 < len(command):
                return index + 1
        raise ValueError(
            "Weft prompt payload transport does not know where the prompt sits in "
            f"{command[0]!r}: expected `codex exec ... <prompt>` or `-p <prompt>`"
        )

    @classmethod
    def _prompt_from_stdin(cls, command: list[str]) -> list[str]:
        """The same command with its brief replaced by a stdin marker."""
        rewritten = list(command)
        rewritten[cls._prompt_index(command)] = "-"
        return rewritten

    @staticmethod
    def _split_model_flag(command: list[str]) -> tuple[list[str], str | None]:
        """Lift `--model VALUE` out of a command, returning both.

        Weft scans a job's command for `--model`, `--model-name` and
        `--base-model` and stages the value as a HuggingFace repo
        (`internal/dataloc/pyscan.go`). A harness selector is a provider/model
        pair, not a dataset reference, so `--model zai/glm-5.3-flash` became
        `download hf:zai/glm-5.3-flash`, which does not exist -- the job was
        blocked before it ran, with a staging error naming a repo nobody had
        asked for.

        Declaring any explicit `--input` would suppress weft's inference
        wholesale (`FilterAutoDetectedInputs` returns nil once anything is
        declared), but that means declaring an input this review does not have
        in order to silence a guess about one. Lifting the flag says the true
        thing instead: the selector is the worker's business, and it reaches
        the harness from `--harness-model` at exec time.
        """
        stripped: list[str] = []
        model: str | None = None
        index = 0
        while index < len(command):
            token = command[index]
            if token == "--model" and index + 1 < len(command):
                model = command[index + 1]
                index += 2
                continue
            if token.startswith("--model="):
                model = token.split("=", 1)[1]
                index += 1
                continue
            stripped.append(token)
            index += 1
        return stripped, model

    def _remote_command(self, command: list[str], cwd: Path, timeout: float | None) -> str:
        resolved = str(cwd.resolve())
        if self.agent in {"omp", "omp-packet"}:
            validate_omp_command(command, cwd)
        # Weft establishes the mirrored project as the job's working directory.
        # A local absolute --cd/--dir would otherwise select the wrong user's
        # tree on the worker.
        argv, harness_model = self._split_model_flag(self._prompt_from_stdin(command))
        rewritten = ["." if part == resolved else part for part in argv]
        evidence_reserve = 90.0 if timeout is None else min(90.0, max(3.0, timeout * 0.1))
        harness_timeout = None if timeout is None else max(1.0, timeout - evidence_reserve)
        ctx_timeout = max(1.0, evidence_reserve / 3.0)
        return shlex.join(
            [
                "agent-execution-worker",
                "execute",
                "--provider",
                self.agent,
                "--model-call-id",
                self.model_call_id,
                *(["--harness-model", harness_model] if harness_model else []),
                *(
                    ["--max-cost-usd", str(self.max_cost_usd)]
                    if self.max_cost_usd is not None
                    else []
                ),
                "--expect-protocol",
                str(WORKER_PROTOCOL_VERSION),
                *(
                    ["--expect-source-sha256", self.expected_source_sha256]
                    if self.expected_source_sha256 is not None
                    else []
                ),
                "--prompt-payload",
                WORKER_PROMPT_PAYLOAD,
                "--expect-prompt-sha256",
                self.prompt_sha256 or "",
                "--evidence-out",
                self.worker_result_path,
                *(["--timeout", str(harness_timeout)] if harness_timeout is not None else []),
                "--ctx-timeout",
                str(ctx_timeout),
                "--",
                *rewritten,
            ]
        )

    def _placement_unknown(self, detail: str) -> dict[str, object]:
        capability = harness_capability_name(self.agent)
        return {
            "state": "unknown",
            "command": [self.executable, *WEFT_HOST_LIST_COMMAND[1:]],
            "required_capabilities": [f"agent:{capability}", WEFT_TOOL_CAPABILITY],
            "detail": detail,
        }

    def _placement_check(self, cwd: Path, deadline: float) -> dict[str, object]:
        command = [self.executable, *WEFT_HOST_LIST_COMMAND[1:]]
        required = (f"agent:{harness_capability_name(self.agent)}", WEFT_TOOL_CAPABILITY)
        try:
            inventory = self.invoke(command, cwd, self._remaining(deadline))
        except (OSError, subprocess.TimeoutExpired) as error:
            return self._placement_unknown(f"weft host list --json could not be read: {error}")
        if inventory.exit_status != 0:
            detail = (
                inventory.stderr.strip()
                or inventory.stdout.strip()
                or f"exit {inventory.exit_status}"
            )
            return self._placement_unknown(f"weft host list --json failed: {detail}")
        try:
            declarations = parse_weft_host_capabilities(
                inventory.stdout,
                command=tuple(command),
            )
        except ValueError as error:
            return self._placement_unknown(str(error))

        candidates = (
            declarations
            if self.host == WEFT_AUTO_HOST
            else tuple(item for item in declarations if item.host == self.host)
        )
        if any(
            declaration.values is not None
            and all(capability in declaration.values for capability in required)
            for declaration in candidates
        ):
            return {
                "state": "eligible",
                "command": command,
                "required_capabilities": list(required),
            }

        scope = "any inventory host" if self.host == WEFT_AUTO_HOST else f"host {self.host!r}"
        details = [
            (
                f"Weft cannot place this dispatch: {scope} does not declare every required "
                f"capability ({', '.join(required)})"
            )
        ]
        if not candidates:
            details.append(
                "the inventory contains no candidate host for this route; missing capabilities: "
                + ", ".join(required)
            )
        else:
            never_declared = [
                capability
                for capability in required
                if not any(
                    declaration.values is not None and capability in declaration.values
                    for declaration in candidates
                )
            ]
            if never_declared:
                details.append("no candidate host declares: " + ", ".join(never_declared))
            absent = [item.host for item in candidates if item.values is None]
            if absent:
                details.append(
                    "hosts with no configured capabilities value (key absent): " + ", ".join(absent)
                )
            empty = [item.host for item in candidates if item.values == ()]
            if empty:
                details.append("hosts declaring an empty capability list: " + ", ".join(empty))
            for declaration in candidates:
                if declaration.values is None:
                    continue
                missing = [
                    capability for capability in required if capability not in declaration.values
                ]
                if missing:
                    details.append(f"{declaration.host} is missing: {', '.join(missing)}")
        return {
            "state": "refused",
            "command": command,
            "required_capabilities": list(required),
            "detail": "; ".join(details),
        }

    @staticmethod
    def _attach_placement_diagnostic(
        execution: dict[str, object], placement: dict[str, object]
    ) -> None:
        if placement.get("state") != "eligible":
            existing = execution.get("diagnostics")
            diagnostics = dict(existing) if isinstance(existing, dict) else {}
            diagnostics["placement"] = placement
            execution["diagnostics"] = diagnostics

    def _execution(
        self,
        receipt: WeftRunReceipt,
        *,
        transport: str,
        admission: dict[str, object],
    ) -> dict[str, object]:
        return {
            "transport": transport,
            "host": receipt.selected_host or self.host,
            "job_id": receipt.job_id,
            "receipt": receipt.to_dict(),
            "admission": admission,
            "prompt_sha256": self.prompt_sha256,
            **({"omp_selector": self.omp_selector} if self.omp_selector else {}),
            "submitter_session": self.submitter_session,
            "expected_worker_source_sha256": self.expected_source_sha256,
        }

    @staticmethod
    def _processing(
        execution: dict[str, object],
        *,
        state: str,
        step: str,
        detail: str = "",
        **fields: object,
    ) -> None:
        processing: dict[str, object] = {"state": state, "step": step}
        if detail:
            processing["detail"] = detail
        # Named diagnostics ride alongside `detail`, which is one line and
        # cannot hold the bytes that explain a crash. Empty values are dropped
        # rather than stored: a recorded empty stream would be read as "the
        # harness printed nothing", which is a different fact from "we kept
        # nothing".
        for name, value in fields.items():
            if value not in (None, "", 0):
                processing[name] = value
        execution["processing"] = processing

    def _mark_processed(
        self,
        *,
        job_id: str,
        cwd: Path,
        execution: dict[str, object],
    ) -> None:
        """Mark a Weft result consumed after its consumer's ingestion is durable."""
        try:
            marked = self.invoke(
                [self.executable, "job", "mark-processed", job_id],
                cwd,
                30.0,
            )
        except subprocess.TimeoutExpired:
            self._processing(
                execution,
                state="mark_failed",
                step="mark_processed",
                detail="mark-processed timed out after 30s",
            )
            return
        except (FileNotFoundError, PermissionError) as error:
            self._processing(
                execution,
                state="mark_failed",
                step="mark_processed",
                detail=str(error),
            )
            return
        execution["processed"] = marked.exit_status == 0
        if marked.exit_status != 0:
            detail = (marked.stderr or marked.stdout).strip().splitlines()
            execution["processing_error"] = detail[-1] if detail else "no output"
            self._processing(
                execution,
                state="mark_failed",
                step="mark_processed",
                detail=str(execution["processing_error"]),
            )
        else:
            self._processing(execution, state="marked", step="mark_processed")

    def _artifact(self, job_id: str, cwd: Path, deadline: float) -> CommandResult:
        """Contain Weft's artifact auto-sync outside the consumer's source tree.

        ``artifact cat`` may materialize its relative artifact path under its
        working directory before printing it. Keep that side effect in private
        scratch storage; ``cwd`` still names the original local project for
        worker evidence path mapping, never this retrieval directory.
        """
        target = cwd.resolve()
        scratch_root = Path(tempfile.gettempdir()).resolve()
        if scratch_root.is_relative_to(target):
            scratch_root = Path.home() / ".cache" / "agent-execution" / "artifacts"
            if scratch_root.resolve().is_relative_to(target):
                raise OSError("no artifact scratch directory exists outside the target tree")
            scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(
            prefix="agent-execution-artifact-", dir=scratch_root
        ) as temporary:
            return self._call(
                [self.executable, "artifact", "cat", job_id, self.worker_result_path],
                Path(temporary),
                deadline,
            )

    def _worker_result(self, job_id: str, cwd: Path, deadline: float) -> CommandResult:
        artifact = CommandResult(1, "", "worker result was not requested")
        for delay in (0.0, *ARTIFACT_RETRY_DELAYS):
            if delay:
                try:
                    remaining = self._remaining(deadline)
                except WeftExecutionAmbiguous:
                    break
                self.sleep(min(delay, remaining))
            try:
                artifact = self._artifact(job_id, cwd, deadline)
            except (WeftExecutionAmbiguous, WeftExecutionDetached) as error:
                artifact = CommandResult(2, "", str(error))
                break
            except OSError as error:
                artifact = CommandResult(2, "", str(error))
            if artifact.exit_status == 0:
                return artifact
        return artifact

    def _worker_result_after_lost_observation(self, job_id: str, cwd: Path) -> CommandResult | None:
        """Fetch the worker result once the local waiter stopped observing the job.

        A lapsed waiter is a fact about the observation, not about the job: the review
        may well have finished while the status surface it was watching went stale.
        The artifact is the direct observation the waiter stood in for, so it is
        consulted before the assignment is classified as detached.

        Returns ``None`` when the artifact is not there, which is the genuine detached
        case and the only one that leaves work for a later retrieval.
        """
        deadline = self.clock() + LOST_OBSERVATION_PROBE_SECONDS
        try:
            artifact = self._artifact(job_id, cwd, deadline)
        except (WeftExecutionAmbiguous, WeftExecutionDetached, OSError):
            return None
        return artifact if artifact.exit_status == 0 else None

    @staticmethod
    def _retrieval_diagnostics(
        *, artifact: CommandResult, log: CommandResult, status: CommandResult
    ) -> dict[str, object]:
        def tail(text: str) -> str:
            return "\n".join(text.strip().splitlines()[-DIAGNOSTIC_TAIL_LINES:])

        diagnostics: dict[str, object] = {
            "artifact_stdout": tail(artifact.stdout),
            "artifact_stderr": tail(artifact.stderr),
            "worker_log_stdout": tail(log.stdout),
            "worker_log_stderr": tail(log.stderr),
            "status_stdout": tail(status.stdout),
            "status_stderr": tail(status.stderr),
        }
        for stream, text in (("stdout", log.stdout), ("stderr", log.stderr)):
            for line in reversed(text.splitlines()):
                if '"schema_version"' not in line or "agent-execution.worker-result/v1" not in line:
                    continue
                start = line.find("{")
                end = line.rfind("}")
                if start < 0 or end < start:
                    continue
                try:
                    worker = WorkerResult.parse(line[start : end + 1])
                except ValueError:
                    continue
                summary = worker.summary()
                summary["source"] = f"weft_log_{stream}"
                diagnostics["worker_result"] = summary
                return diagnostics
        return diagnostics

    def _command_result_from_worker(
        self,
        *,
        job_id: str,
        cwd: Path,
        execution: dict[str, object],
        artifact: CommandResult,
    ) -> CommandResult:
        """Validate one durable worker artifact and reconstruct its command result."""
        try:
            worker = WorkerResult.parse(artifact.stdout)
            if worker.model_call_id != self.model_call_id:
                raise ValueError("worker result does not match the requested model call")
            if worker.provider != self.agent:
                raise ValueError("worker result does not match the requested provider")
        except ValueError as error:
            self._processing(
                execution,
                state="not_marked",
                step="worker_result_validation",
                detail=str(error),
            )
            return CommandResult(
                1,
                "",
                f"invalid Weft worker result for {job_id}: {error}",
                execution=execution,
            )
        # Name the artifact this call actually wrote. `summary()` defaults to
        # the historical fixed path, so the record read
        # `outputs/agent-execution-worker-result.json` for every call while the
        # per-call file sat beside it -- a field describing where the evidence
        # is, pointing somewhere it is not.
        summary = worker.summary(artifact_path=self.worker_result_path)
        summary["artifact_sha256"] = hashlib.sha256(artifact.stdout.encode()).hexdigest()
        execution["worker_result"] = summary
        execution["duration_seconds"] = worker.completed_at - worker.started_at

        if worker.worker_protocol_version != WORKER_PROTOCOL_VERSION:
            detail = protocol_mismatch_detail(
                WORKER_PROTOCOL_VERSION,
                worker.worker_protocol_version,
            )
            self._processing(
                execution,
                state="not_marked",
                step="worker_protocol_validation",
                detail=detail,
            )
            return CommandResult(
                1, worker.harness.stdout, detail, execution=execution, worker_result=worker
            )

        expected_source = execution.get("expected_worker_source_sha256")
        if expected_source != worker.worker_source_sha256:
            detail = (
                "worker source mismatch: conductor expects "
                f"{expected_source}, remote worker reports {worker.worker_source_sha256}"
            )
            self._processing(
                execution,
                state="not_marked",
                step="worker_source_validation",
                detail=detail,
            )
            return CommandResult(
                1, worker.harness.stdout, detail, execution=execution, worker_result=worker
            )
        expected_prompt = execution.get("prompt_sha256")
        if isinstance(expected_prompt, str) and expected_prompt != worker.prompt_sha256:
            detail = (
                "worker prompt mismatch: conductor expects "
                f"{expected_prompt}, remote worker reports {worker.prompt_sha256}"
            )
            self._processing(
                execution,
                state="not_marked",
                step="worker_prompt_validation",
                detail=detail,
            )
            return CommandResult(
                1, worker.harness.stdout, detail, execution=execution, worker_result=worker
            )

        if self.agent in {"omp", "omp-packet"} and worker.status == "completed":
            try:
                expected_selector = execution.get("omp_selector")
                if not isinstance(expected_selector, str) or not expected_selector:
                    raise ValueError("OMP dispatch receipt lacks its exact inference selector")
                omp_transcript(
                    worker.harness.stdout,
                    selector=expected_selector,
                    policy="read-only-no-shell" if self.agent == "omp" else "packet-only-no-tools",
                    cwd=worker.worker_cwd,
                    prompt_sha256=worker.prompt_sha256 or None,
                )
            except ValueError as error:
                return CommandResult(
                    1, worker.harness.stdout, str(error), execution=execution, worker_result=worker
                )
        evidence_unavailable = (
            self.agent != "omp"
            and worker.status == "evidence_failed"
            and worker.harness.exit_status == 0
        )
        if evidence_unavailable:
            diagnostics = execution.get("diagnostics")
            recorded = dict(diagnostics) if isinstance(diagnostics, dict) else {}
            recorded["worker_evidence_failure"] = worker.failure
            execution["diagnostics"] = recorded
        elif worker.status != "completed":
            # Reported by llm-performance-models against wj7222: the record
            # carried one line of `detail` and the bytes that would explain the
            # crash lived only in a studio-side weft log, which expires. Tails
            # are bounded and kept on the same terms as the local path.
            self._processing(
                execution,
                state="not_marked",
                step="worker_execution",
                detail=worker.failure,
                worker_status=worker.status,
                harness_exit_status=worker.harness.exit_status,
                stdout_bytes=len(worker.harness.stdout),
                stdout_tail=worker.harness.stdout[-8000:],
                stderr_bytes=len(worker.harness.stderr),
                stderr_tail=worker.harness.stderr[-8000:],
            )
            return CommandResult(
                1,
                worker.harness.stdout,
                worker.failure or worker.harness.stderr,
                execution=execution,
                worker_result=worker,
            )

        self._processing(execution, state="pending", step="consumer_validation")

        def mark_consumed() -> None:
            self._mark_processed(job_id=job_id, cwd=cwd, execution=execution)

        return CommandResult(
            worker.harness.exit_status or 0,
            worker.harness.stdout,
            worker.harness.stderr,
            execution=execution,
            worker_result=worker,
            consumed=mark_consumed,
        )

    @property
    def worker_result_path(self) -> str:
        """Where this call's worker result lands, unique per model call.

        Every offloaded review on one host runs in the same synced working
        directory, so a fixed `outputs/agent-execution-worker-result.json` is a
        shared mutable path: three concurrent reviews wrote and read one file
        and all three returned an empty harness output under a `completed`
        worker status -- a false success, silent, and only reproducible under
        concurrency. Raw omp runs three at a time on this host without
        trouble, so the contention was ours, not the harness's.
        """
        return f"outputs/agent-execution-worker-result-{self.model_call_id}.json"

    def record_observed_host(self, job_id: str, cwd: Path, execution: dict[str, object]) -> None:
        """Record where Weft actually ran a job, beside where we asked it to.

        ``execution["host"]`` is admission-time placement -- the host this
        project requested or the receipt selected. It is not where the job ran:
        wj6989 was submitted for ``studio``, replanned onto rental ``wi7777``
        after an unplace, and died there with exit 127 while our record still
        said ``studio``. A consumer reading that field learned our intent and
        believed it had learned an observation.

        The observed host is available on ``weft job list --format json``,
        which is a versioned envelope (``kind: job_list``, ``version: 1``) --
        so this reads a declared surface rather than parsing another tool's
        human-readable output. ``weft job info`` and ``weft status`` reject
        ``--json`` outright, which is why this pages a list for one job.

        Absence is recorded as absence. A job outside the listing window, a
        surface whose version we do not recognize, or a row without a host all
        leave ``observed_host`` unset with a reason, because a field that
        silently falls back to the requested host would make the two
        indistinguishable again -- which is the whole defect.
        """
        observation: dict[str, object] = {"source": "weft.job_list/v1"}
        try:
            listed = self.invoke(
                [self.executable, "job", "list", "--format", "json", "--limit", "200"],
                cwd,
                WEFT_HOST_OBSERVATION_TIMEOUT,
            )
        except (subprocess.TimeoutExpired, OSError) as error:
            observation["unobserved"] = f"{type(error).__name__}"
            execution["host_observation"] = observation
            return
        if listed.exit_status != 0:
            observation["unobserved"] = f"exit {listed.exit_status}"
            execution["host_observation"] = observation
            return
        try:
            payload = json.loads(listed.stdout)
        except json.JSONDecodeError:
            observation["unobserved"] = "unparseable listing"
            execution["host_observation"] = observation
            return
        if not isinstance(payload, dict) or payload.get("kind") != "job_list":
            observation["unobserved"] = "unrecognized listing surface"
            execution["host_observation"] = observation
            return
        version = payload.get("version")
        if version != WEFT_JOB_LIST_VERSION:
            observation["unobserved"] = f"listing version {version!r}"
            execution["host_observation"] = observation
            return
        rows = payload.get("jobs")
        rows = rows if isinstance(rows, list) else []
        for row in rows:
            if not isinstance(row, dict) or str(row.get("job_id") or "") != job_id:
                continue
            host = str(row.get("host") or "").strip()
            if host:
                observation["observed_host"] = host
                requested = str(execution.get("host") or "").strip()
                if requested and host != requested:
                    observation["differs_from_requested"] = requested
            else:
                observation["unobserved"] = "row carries no host"
            execution["host_observation"] = observation
            return
        observation["unobserved"] = "job not in listing window"
        execution["host_observation"] = observation

    def _inspect_job(
        self, job_id: str, cwd: Path, deadline: float
    ) -> dict[str, object] | WeftRetrievalOutcome:
        """Read the public job record with the retrieval budget and exact identity."""
        try:
            inspected = self._call(
                [self.executable, "job", "inspect", job_id, "--json"], cwd, deadline
            )
        except (WeftExecutionAmbiguous, WeftExecutionDetached, OSError) as error:
            return WeftRetrievalOutcome("unknown", f"could not inspect Weft job {job_id}: {error}")
        if inspected.exit_status != 0:
            detail = inspected.stderr.strip() or inspected.stdout.strip()
            return WeftRetrievalOutcome(
                "unknown",
                f"could not inspect Weft job {job_id}" + (f": {detail}" if detail else ""),
            )
        try:
            raw = cast(object, json.loads(inspected.stdout))
        except json.JSONDecodeError:
            return WeftRetrievalOutcome("unknown", f"Weft job {job_id} inspect output was not JSON")
        if not isinstance(raw, dict):
            return WeftRetrievalOutcome(
                "unknown", f"Weft job {job_id} inspect output was not an object"
            )
        record = cast(dict[str, object], raw)
        if record.get("id") != job_id:
            return WeftRetrievalOutcome(
                "unknown", f"Weft inspect returned a record for another job, not {job_id}"
            )
        return record

    def _dispatched_source(self, record: dict[str, object]) -> str:
        """Recover a legacy pin from the accepted command, never the worker artifact."""
        options = parse_worker_command(record.get("command"))

        def one_value(flag: str) -> str:
            if flag not in options:
                raise ValueError(f"accepted command must carry exactly one {flag}")
            return options[flag]

        if one_value("--model-call-id") != self.model_call_id:
            raise ValueError("accepted command names another model call")
        expected = one_value("--expect-source-sha256")
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("accepted command carries an invalid source hash")
        return expected

    @staticmethod
    def _unknown_source(job_id: str, execution: dict[str, object]) -> WeftRetrievalOutcome | None:
        expected = execution.get("expected_worker_source_sha256")
        if isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected):
            return None
        return WeftRetrievalOutcome(
            "unknown", f"Weft job {job_id} has no valid dispatched worker source expectation"
        )

    def retrieve_from_file(
        self,
        *,
        job_id: str,
        cwd: Path,
        execution: dict[str, object],
        artifact_path: Path,
    ) -> CommandResult | WeftRetrievalOutcome:
        """Validate one worker artifact read from disk, through the same parser.

        The case on record is an artifact copied off a worker whose verdict a
        human then read out of a transcript by hand, because nothing could
        ingest the bytes. A second reported case was withdrawn on inspection:
        the job had been cancelled 58 seconds into an 810-second budget and
        almost certainly never wrote its evidence, and weft recorded its
        artifact as `unknown` rather than absent. That is an ordinary
        cancellation, not a storage fault -- nothing here is evidence that an
        artifact store loses artifacts, and this path is not a remedy for one.

        A second parser for the same bytes would be a second definition of
        what a verdict is, so this differs from `retrieve` only in where the
        bytes come from: identity, provider, and nonce checks are the ones
        `retrieve` performs.
        """
        try:
            stdout = artifact_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return WeftRetrievalOutcome(
                "unknown", f"Weft job {job_id} artifact {artifact_path} is unreadable: {error}"
            )
        unknown = self._unknown_source(job_id, execution)
        if unknown is not None:
            return unknown
        return self._command_result_from_worker(
            job_id=job_id,
            cwd=cwd,
            execution=execution,
            artifact=CommandResult(0, stdout, ""),
        )

    def retrieve(
        self,
        *,
        job_id: str,
        cwd: Path,
        execution: dict[str, object],
        timeout: float = LOST_OBSERVATION_PROBE_SECONDS,
    ) -> CommandResult | WeftJobFailure | WeftRetrievalOutcome:
        """Retrieve one accepted job without submitting or falling back locally."""
        self.last_execution = execution
        deadline = self.clock() + timeout
        artifact: CommandResult | None = None
        artifact_detail = ""
        try:
            artifact = self._worker_result(job_id, cwd, deadline)
        except (WeftExecutionAmbiguous, WeftExecutionDetached, OSError) as error:
            artifact_detail = str(error)
        if artifact is not None and artifact.exit_status == 0:
            # Source identity gates the worker artifact, not observations of
            # whether its job is still running or has already failed.
            if execution.get("expected_worker_source_sha256") in (None, ""):
                record = self._inspect_job(job_id, cwd, deadline)
                if isinstance(record, WeftRetrievalOutcome):
                    return record
                try:
                    expected = self._dispatched_source(record)
                except ValueError as error:
                    return WeftRetrievalOutcome(
                        "unknown", f"Weft job {job_id} dispatch expectation is unknown: {error}"
                    )
                execution["expected_worker_source_sha256"] = expected
            unknown = self._unknown_source(job_id, execution)
            if unknown is not None:
                return unknown
            return self._command_result_from_worker(
                job_id=job_id,
                cwd=cwd,
                execution=execution,
                artifact=artifact,
            )

        if artifact is not None:
            artifact_detail = artifact.stderr.strip() or artifact.stdout.strip()

        # Artifact absence alone says nothing about the job. Weft's normalized,
        # identity-checked job record distinguishes work that has not run from a
        # finished job whose result cannot be read; an unreadable record establishes
        # neither under decision 0050.
        record = self._inspect_job(job_id, cwd, deadline)
        if isinstance(record, WeftRetrievalOutcome):
            return record
        raw_status = record.get("status")
        if not isinstance(raw_status, str) or not raw_status:
            return WeftRetrievalOutcome(
                "unknown",
                f"Weft job {job_id} inspect output carried no readable status",
            )
        if raw_status == "queued" and record.get("start_time") is not None:
            # Contradictory: queued jobs have not started. Neither reading is
            # established, so neither is asserted.
            return WeftRetrievalOutcome(
                "unknown",
                f"Weft job {job_id} is queued but has a start time",
                job_status=raw_status,
            )
        if raw_status in WEFT_LIVE_JOB_STATUSES:
            return WeftRetrievalOutcome(
                "pending",
                f"Weft job {job_id} is {raw_status} and may still produce its result",
                job_status=raw_status,
            )
        if raw_status != "failed":
            return WeftRetrievalOutcome(
                "unretrievable",
                f"Weft job {job_id} is {raw_status} but its worker-result artifact "
                "could not be read" + (f": {artifact_detail}" if artifact_detail else ""),
                job_status=raw_status,
            )
        # The inspect record owns the cause. Artifact recovery and log reading
        # describe our observations, never why the accepted job failed.
        reason = record.get("failure_reason")
        reason = reason if isinstance(reason, str) and reason.strip() else None
        job_failure: dict[str, object] = {
            "job_id": job_id,
            "status": raw_status,
            "source": [self.executable, "job", "inspect", job_id, "--json"],
        }
        if reason is not None:
            job_failure["failure_reason"] = reason
        else:
            job_failure["unobserved"] = "inspect carried no readable failure_reason"
        if isinstance(record.get("host"), str):
            job_failure["host"] = record["host"]
        existing = execution.get("diagnostics")
        diagnostics = dict(existing) if isinstance(existing, dict) else {}
        diagnostics["job_failure"] = job_failure
        execution["diagnostics"] = diagnostics
        failure = WeftJobFailure(
            job_id=job_id,
            detail=f"Weft job {job_id} failed: {reason or 'cause not reported by inspect'}",
        )
        self._processing(
            execution,
            state="not_applicable",
            step="worker_result_retrieval",
            detail="worker-result artifact could not be read"
            + (f": {artifact_detail}" if artifact_detail else ""),
            job_id=job_id,
        )
        # Bounded, unparsed context only. A timeout or refusal here cannot undo
        # the identity-checked terminal observation already recorded above.
        log_command = [self.executable, "log", job_id, "--tail", "30"]
        log_diagnostic: dict[str, object] = {"command": log_command}
        diagnostics["job_failure_log"] = log_diagnostic
        try:
            logged = self._call(log_command, cwd, deadline)
        except (WeftExecutionAmbiguous, WeftExecutionDetached, OSError) as error:
            log_diagnostic["unobserved"] = str(error)
        else:
            for stream, text in (("stdout", logged.stdout), ("stderr", logged.stderr)):
                excerpt = "\n".join(text.strip().splitlines()[-30:])
                if excerpt:
                    log_diagnostic[stream] = excerpt
            if logged.exit_status != 0:
                log_diagnostic["unobserved"] = f"exit {logged.exit_status}"
        return failure

    def _submit(
        self,
        command: list[str],
        cwd: Path,
        timeout: float | None,
        deadline: float,
    ) -> CommandResult:
        route = [] if self.host == WEFT_AUTO_HOST else ["--host", self.host]
        # The brief, located by shape. `command[-1]` was the third place this
        # file assumed the prompt comes last: for an OMP command it is the
        # trailing `--system-prompt` value, so the payload shipped to the
        # worker would have been the system prompt and the review would have
        # run against it.
        prompt = command[self._prompt_index(command)]
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="agent-execution-prompt-"
        ) as payload:
            payload.write(prompt)
            payload.flush()
            return self._call(
                [
                    self.executable,
                    "run",
                    *route,
                    "--agent",
                    # Weft's agent vocabulary names the harness, not our
                    # adapter variant: `--agent omp-packet` exits 1 with no
                    # JSON, which surfaced here as "Weft submission did not
                    # return JSON" -- a parse complaint for a rejected flag.
                    # The capability filter below already uses this spelling,
                    # so the two now agree by construction.
                    harness_capability_name(self.agent),
                    "--require-capability",
                    WEFT_TOOL_CAPABILITY,
                    # `agent-execution-worker` is deployed per host, so a review
                    # can only run where it was installed. The capability
                    # requirement above expresses that for inventory hosts,
                    # which declare capabilities; a rental declares none, so
                    # it cannot satisfy or fail that filter. A job that
                    # replanned onto a fresh rental therefore ran a command
                    # the machine had never heard of and died with a bare
                    # exit 127. `inventory` is Weft's reserved tag for
                    # blocking rental placement, so the constraint survives
                    # any later replan, retry, or unplace rather than holding
                    # only for the placement we happened to get.
                    "--tag",
                    WEFT_INVENTORY_TAG,
                    # The installed worker owns its environment; the target
                    # snapshot is evidence, not a project to install.
                    "--setup",
                    "none",
                    "--project",
                    WORKER_PROJECT,
                    "--submitter-session",
                    self.submitter_session,
                    "--produces",
                    self.worker_result_path,
                    "--payload",
                    f"{WORKER_PROMPT_PAYLOAD}={payload.name}",
                    "--if-online",
                    "--idempotency-key",
                    self.model_call_id,
                    "--json",
                    "-m",
                    # Product-owned identity remains visible beside the exact
                    # model-call id carried in the worker command.
                    self.owner_description,
                    self._remote_command(command, cwd, timeout),
                ],
                cwd,
                deadline,
            )

    def _probe_submitted_job(self, cwd: Path, deadline: float) -> str | None:
        """The job this assignment created, when its submission receipt would not parse.

        The exact worker flag identifies the dispatch. Descriptions and substring
        matches are discovery hints, not proof that a job belongs to this call.

        Only presence answers. Exactly one match identifies the job; zero does not
        prove none was created, because a job may not be indexed the instant it is
        accepted, and more than one cannot be attributed to this assignment. Both
        leave the caller's original verdict standing, so this can only ever move an
        assignment from ambiguous to detached — never the reverse, and never to a
        state that would license re-execution.
        """
        budget = min(LOST_OBSERVATION_PROBE_SECONDS, max(1.0, deadline - self.clock()))
        return probe_submitted_job(
            model_call_id=self.model_call_id,
            cwd=cwd,
            timeout=budget,
            invoke=self.invoke,
            executable=self.executable,
        )

    def _probed_submission_execution(
        self,
        *,
        job_id: str,
        attempt: int,
        attempts: list[dict[str, object]],
        placement: dict[str, object],
        diagnostics: dict[str, object],
    ) -> dict[str, object]:
        """Persist the job identity established by an exact assignment probe."""
        attempts.append(
            {
                "attempt": attempt,
                "receipt": None,
                "acceptance_basis": "assignment_probe",
            }
        )
        admission: dict[str, object] = {
            "attempt_count": len(attempts),
            "attempts": attempts,
            "final_placement": {
                "transport": "weft",
                "attempt": attempt,
                "host": self.host,
                "job_id": job_id,
                "acceptance_basis": "assignment_probe",
            },
        }
        execution: dict[str, object] = {
            "transport": "weft",
            "host": self.host,
            "job_id": job_id,
            "admission": admission,
            "prompt_sha256": self.prompt_sha256,
            **({"omp_selector": self.omp_selector} if self.omp_selector else {}),
            "expected_worker_source_sha256": self.expected_source_sha256,
            "diagnostics": diagnostics,
        }
        self._attach_placement_diagnostic(execution, placement)
        self.last_execution = execution
        self._progress(
            "accepted",
            transport="weft",
            host=self.host,
            job_id=job_id,
            acceptance_basis="assignment_probe",
            admission=admission,
            submitter_session=self.submitter_session,
            expected_worker_source_sha256=execution["expected_worker_source_sha256"],
            survives_process_exit=True,
        )
        return execution

    def _await_accepted_job(
        self,
        *,
        job_id: str,
        execution: dict[str, object],
        cwd: Path,
        deadline: float,
    ) -> CommandResult:
        """Observe and retrieve a job whose durable acceptance is established.

        A valid receipt and an exact assignment probe establish the same job
        identity. Once either has done so, a broken waiter is an observation
        failure, not renewed ambiguity about whether the job exists.
        """
        self.last_execution = execution
        try:
            wait_seconds = max(1, int(self._remaining(deadline)))
            self._progress(
                "waiting",
                transport="weft",
                host=execution["host"],
                job_id=job_id,
                wait_seconds=wait_seconds,
            )
            status = self._call(
                [
                    self.executable,
                    "status",
                    job_id,
                    "--wait",
                    "--wait-timeout",
                    f"{wait_seconds}s",
                ],
                cwd,
                deadline,
            )
        except (WeftExecutionDetached, WeftExecutionAmbiguous, OSError) as error:
            status = CommandResult(2, "", str(error))

        recovered: CommandResult | None = None
        # The status command's exit code describes the watcher, not the worker.
        # In particular, exit 1 can be a broken observation just as exit 2 can
        # be a timeout. Only the worker artifact can establish a review result.
        if status.exit_status != 0:
            diagnostics = execution.get("diagnostics")
            execution["diagnostics"] = {
                **(diagnostics if isinstance(diagnostics, dict) else {}),
                "status_stdout": "\n".join(
                    status.stdout.strip().splitlines()[-DIAGNOSTIC_TAIL_LINES:]
                ),
                "status_stderr": "\n".join(
                    status.stderr.strip().splitlines()[-DIAGNOSTIC_TAIL_LINES:]
                ),
            }
            recovered = self._worker_result_after_lost_observation(job_id, cwd)
            if recovered is None:
                self._processing(execution, state="pending", step="detached_retrieval")
                raise WeftExecutionDetached(f"Weft job {job_id} outlived its local waiter")
            self._progress(
                "observation_recovered",
                transport="weft",
                host=execution["host"],
                job_id=job_id,
            )
            status = CommandResult(0, status.stdout, status.stderr)

        self._progress(
            "retrieving",
            transport="weft",
            host=execution["host"],
            job_id=job_id,
        )
        artifact = (
            recovered if recovered is not None else self._worker_result(job_id, cwd, deadline)
        )
        if artifact.exit_status != 0:
            try:
                log = self._call(
                    [self.executable, "log", job_id, "--full", "--no-sync"],
                    cwd,
                    deadline,
                )
            except (WeftExecutionAmbiguous, WeftExecutionDetached, OSError) as error:
                log = CommandResult(2, "", str(error))
            diagnostics = self._retrieval_diagnostics(artifact=artifact, log=log, status=status)
            existing_diagnostics = execution.get("diagnostics")
            if isinstance(existing_diagnostics, dict):
                diagnostics = {**existing_diagnostics, **diagnostics}
            execution["diagnostics"] = diagnostics
            if diagnostics.get("worker_result") is not None:
                execution["worker_result"] = diagnostics["worker_result"]
            self._processing(
                execution,
                state="pending",
                step="worker_result_retrieval",
                detail="worker-result artifact is not yet observable",
            )
            detail = "\n".join(
                f"{key}: {value}"
                for key, value in diagnostics.items()
                if isinstance(value, str) and value
            )
            raise WeftExecutionDetached(
                f"Weft job {job_id} result remains unobserved: "
                f"{detail or 'worker-result artifact is not available'}"
            )
        return self._command_result_from_worker(
            job_id=job_id,
            cwd=cwd,
            execution=execution,
            artifact=artifact,
        )

    def __call__(self, command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        # Keep construction/retrieval available for historical native artifacts,
        # but never submit or fall back locally for a new native Codex call.
        if self.agent == "codex":
            raise ValueError("native Codex execution is unsupported; use OpenAI models via OMP")
        deadline = self.clock() + (REMOTE_OBSERVATION_SECONDS if timeout is None else timeout)
        # The brief the worker will be handed, located by shape rather than by
        # position: hashing `command[-1]` pinned a trailing flag for any harness
        # that does not put its prompt last.
        prompt = command[self._prompt_index(command)]
        self.prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        if self.agent in {"omp", "omp-packet"}:
            self.omp_selector = validate_omp_command(command, cwd).selector
        self.last_execution = None
        placement = self._placement_check(cwd, deadline)
        if placement["state"] == "unknown":
            self.last_execution = {
                "transport": "weft",
                "host": self.host,
                "diagnostics": {"placement": placement},
            }
            self._progress(
                "placement_unknown",
                transport="weft",
                host=self.host,
                detail=placement["detail"],
            )
        elif placement["state"] == "refused":
            detail = str(placement["detail"])
            self.last_execution = {
                "transport": "weft",
                "host": self.host,
                "diagnostics": {"placement": placement},
                "processing": {
                    "state": "refused",
                    "step": "placement_check",
                    "detail": detail,
                },
            }
            self._progress("placement_refused", transport="weft", host=self.host, detail=detail)
            raise WeftPlacementRefused(detail)

        # A recorded protocol disagreement refuses here, before a job exists. The
        # same comparison already runs on the returned worker result, but by then
        # the job has been queued and executed remotely — and a retry routes
        # straight back into the deployment that just refused it, changing no
        # condition. `unknown` proceeds: a host nothing has observed is not a
        # host known to disagree, and the post-hoc check still catches it.
        readiness, readiness_detail = (
            self.readiness(self.host) if self.readiness is not None else ("unknown", "")
        )
        if readiness == "mismatch":
            self._progress(
                "deployment_refused", transport="weft", host=self.host, detail=readiness_detail
            )
            raise WeftPlacementRefused(readiness_detail)

        attempts: list[dict[str, object]] = []
        receipt: WeftRunReceipt | None = None
        submission: CommandResult | None = None
        for attempt in range(1, len(ADMISSION_RETRY_DELAYS) + 2):
            self._progress("submitting", transport="weft", host=self.host, attempt=attempt)
            try:
                submission = self._submit(command, cwd, timeout, deadline)
            except WeftExecutionAmbiguous as error:
                # The local `weft run` waiter can consume the entire harness
                # budget before returning a receipt. Its timeout says nothing
                # about whether Weft accepted the idempotency key, so spend one
                # fresh bounded observation on the exact assignment route.
                accepted = self._probe_submitted_job(
                    cwd, self.clock() + LOST_OBSERVATION_PROBE_SECONDS
                )
                if accepted is None:
                    self.last_execution = {
                        "transport": "weft",
                        "host": self.host,
                        "diagnostics": {"submission_error": str(error)},
                        "processing": {
                            "state": "unknown",
                            "step": "submission_receipt",
                            "detail": (
                                "no Weft job was yet attributable to this model call; "
                                "retain the idempotency key and probe for acceptance "
                                "before retrying"
                            ),
                        },
                    }
                    self._progress(
                        "submission_unknown",
                        transport="weft",
                        host=self.host,
                        model_call_id=self.model_call_id,
                    )
                    raise
                execution = self._probed_submission_execution(
                    job_id=accepted,
                    attempt=attempt,
                    attempts=attempts,
                    placement=placement,
                    diagnostics={"submission_error": str(error)},
                )
                self._processing(execution, state="pending", step="detached_retrieval")
                raise WeftExecutionDetached(
                    f"Weft job {accepted} outlived its submission waiter"
                ) from error
            try:
                receipt = WeftRunReceipt.parse(
                    submission.stdout.strip(), idempotency_key=self.model_call_id
                )
            except ValueError as error:
                # A malformed response does not prove that submission failed before
                # creating a job. Retrying could execute an uncertain assignment twice.
                # But whether a job exists is directly observable, and classifying the
                # assignment terminally without looking is the defect decision 0050
                # names: an observation failure is not a verdict on the thing observed.
                # Five completed reviews were discarded this way before anyone looked.
                accepted = self._probe_submitted_job(cwd, deadline)
                if accepted is not None:
                    execution = self._probed_submission_execution(
                        job_id=accepted,
                        attempt=attempt,
                        attempts=attempts,
                        placement=placement,
                        diagnostics={
                            "unreadable_receipt": str(error),
                            # The message alone says a receipt did not parse and
                            # not what arrived, so keep the returned bytes.
                            "receipt_stdout": _tail(submission.stdout),
                            "receipt_stderr": _tail(submission.stderr),
                        },
                    )
                    return self._await_accepted_job(
                        job_id=accepted,
                        execution=execution,
                        cwd=cwd,
                        deadline=deadline,
                    )
                self.last_execution = {
                    "transport": "weft",
                    "host": self.host,
                    "diagnostics": {
                        "unreadable_receipt": str(error),
                        "receipt_stdout": _tail(submission.stdout),
                        "receipt_stderr": _tail(submission.stderr),
                    },
                }
                raise WeftExecutionAmbiguous(str(error)) from error
            attempts.append(
                {
                    "attempt": attempt,
                    "receipt": receipt.to_dict(),
                    "submission_exit_status": submission.exit_status,
                }
            )
            if receipt.placement_decision != "not_accepted":
                break
            if attempt > len(ADMISSION_RETRY_DELAYS):
                admission: dict[str, object] = {
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "final_placement": {"transport": "local", "attempt": attempt},
                }
                execution = self._execution(
                    receipt,
                    transport="weft-local-fallback",
                    admission=admission,
                )
                self._attach_placement_diagnostic(execution, placement)
                self._processing(execution, state="not_applicable", step="local_fallback")
                self.last_execution = execution
                self._progress("local_fallback", transport="local", admission=admission)
                if self.fallback is None:
                    raise RuntimeError("Weft retrieval has no local fallback")
                local = self.fallback(
                    command, cwd, None if timeout is None else self._remaining(deadline)
                )
                return replace(local, execution=execution)
            # The assignment ID is Weft's idempotency key. Re-submitting this
            # same assignment after a proven not_accepted receipt can only
            # retry admission; it cannot create a second execution. No other
            # receipt or submission failure reaches this branch.
            delay = ADMISSION_RETRY_DELAYS[attempt - 1]
            self._progress(
                "admission_retry",
                transport="weft",
                host=self.host,
                attempt=attempt,
                next_attempt=attempt + 1,
                backoff_seconds=delay,
            )
            if self.admission_wait(delay):
                raise WeftAdmissionCancelled(
                    f"Weft admission for {self.model_call_id} was cancelled during backoff"
                )

        if receipt is None or submission is None:
            raise RuntimeError("Weft admission ended without a receipt")

        admission = cast(
            dict[str, object],
            {
                "attempt_count": len(attempts),
                "attempts": attempts,
                "final_placement": {
                    "transport": "weft",
                    "attempt": len(attempts),
                    "host": receipt.selected_host or self.host,
                    "job_id": receipt.job_id,
                },
            },
        )
        execution = self._execution(receipt, transport="weft", admission=admission)
        self._attach_placement_diagnostic(execution, placement)
        if submission.exit_status != 0:
            existing = execution.get("diagnostics")
            execution["diagnostics"] = {
                **(existing if isinstance(existing, dict) else {}),
                "submission_exit_status": submission.exit_status,
                "submission_stdout": _tail(submission.stdout),
                "submission_stderr": _tail(submission.stderr),
            }
        self._processing(execution, state="pending", step="terminal_status")
        self.last_execution = execution
        self._progress(
            "accepted",
            transport="weft",
            host=execution["host"],
            job_id=receipt.job_id,
            placement_decision=receipt.placement_decision,
            receipt=receipt.to_dict(),
            admission=admission,
            submitter_session=self.submitter_session,
            expected_worker_source_sha256=execution["expected_worker_source_sha256"],
            survives_process_exit=True,
        )
        return self._await_accepted_job(
            job_id=receipt.job_id,
            execution=execution,
            cwd=cwd,
            deadline=deadline,
        )
