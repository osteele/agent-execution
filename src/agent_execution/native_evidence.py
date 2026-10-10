"""Evidence-only recovery for retained successful native Claude executions.

This module never launches a harness. It addresses only the executor explicitly
retained by the caller and reads the provider-owned session through ctx's public
CLI, returning capture evidence separately from the immutable worker result.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass

from agent_execution.claude_execution import claude_envelope
from agent_execution.worker import WORKER_PROTOCOL_VERSION, WorkerResult
from agent_execution.worker_transcript import ValidatedTranscript, validate_ctx_transcript


@dataclass(frozen=True)
class RecoveredNativeEvidence:
    """Later evidence, kept distinct from the original worker result."""

    provider: str
    session_id: str
    worker_protocol_version: int
    worker_source_sha256: str
    transcript: dict[str, object]
    validated: ValidatedTranscript


def recover_native_evidence(
    original: WorkerResult,
    *,
    executor: str,
    account: str,
    expected_protocol_version: int,
    expected_source_sha256: str,
    timeout: float = 30.0,
) -> RecoveredNativeEvidence:
    """Capture ctx evidence for an already-successful native Claude call.

    `executor` and `account` must come from the consumer's exact registration
    for the observed producer, not requested placement or the worker result.
    The operation runs only ctx import/show over SSH; it cannot rerun or amend
    the original harness result. The original cwd is transcript identity, not
    a directory that must still exist on the producer.
    """
    if original.provider != "claude" or original.status != "evidence_failed":
        raise ValueError("recovery requires a retained native Claude evidence_failed result")
    if not original.model_call_started or original.harness.exit_status != 0:
        raise ValueError("recovery requires a successful original harness execution")
    if expected_protocol_version != WORKER_PROTOCOL_VERSION:
        raise ValueError("recovery worker protocol differs from this SDK")
    if original.worker_protocol_version != expected_protocol_version:
        raise ValueError("original result protocol differs from retained dispatch identity")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256)
        or original.worker_source_sha256 != expected_source_sha256
    ):
        raise ValueError("original result source differs from retained dispatch identity")
    if original.worker_identity is not None and (
        original.worker_identity.protocol_version != expected_protocol_version
        or original.worker_identity.source_sha256 != expected_source_sha256
    ):
        raise ValueError("original worker identity differs from retained dispatch identity")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", account):
        raise ValueError("invalid registered executor account")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", executor):
        raise ValueError("invalid registered executor")
    if type(timeout) not in {int, float} or timeout <= 0:
        raise ValueError("recovery timeout must be positive")
    try:
        envelope = claude_envelope(original.harness.stdout, session_id=original.session_id)
        native_session = str(envelope["session_id"])
    except ValueError as error:
        raise ValueError(f"original native success identity is invalid: {error}") from error

    target = f"{account}@{executor}"
    prefix = (
        'export PATH="$HOME/.local/bin:$HOME/.bun/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"; '
    )

    def ctx(argv: list[str]) -> dict[str, object]:
        remote = prefix + shlex.join(argv)
        try:
            completed = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            # A failed or interrupted observation remains unknown to the caller.
            raise ValueError(f"ctx recovery observation failed on {target}: {error}") from error
        if completed.returncode:
            raise ValueError(f"ctx recovery command failed on {target}: {completed.stderr.strip()}")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("ctx recovery command did not return JSON") from error
        if not isinstance(value, dict):
            raise ValueError("ctx recovery command returned a non-object")
        return value

    ctx(["ctx", "import", "--provider", "claude", "--format", "json", "--progress", "none"])
    transcript = ctx(
        [
            "ctx",
            "show",
            "session",
            "--provider",
            "claude",
            "--provider-session",
            native_session,
            "--mode",
            "log",
            "--format",
            "json",
        ]
    )
    validated = validate_ctx_transcript(
        transcript, provider="claude", session_id=native_session, remote_cwd=original.worker_cwd
    )
    return RecoveredNativeEvidence(
        provider="claude",
        session_id=native_session,
        worker_protocol_version=expected_protocol_version,
        worker_source_sha256=expected_source_sha256,
        transcript=transcript,
        validated=validated,
    )
