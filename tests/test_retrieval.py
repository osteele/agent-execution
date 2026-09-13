"""Durable worker-result retrieval and explicit consumption acknowledgment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_execution.command import CommandResult
from agent_execution.weft import (
    WeftCommandRunner,
    WeftJobFailure,
    WeftRetrievalOutcome,
)
from agent_execution.worker import WORKER_PROTOCOL_VERSION, HarnessOutcome, WorkerResult
from tests.support.omp import omp_output
from tests.test_weft import dispatch_command, worker_artifact

KEY = "model-call-7"
SOURCE = "a" * 64
JOB = "wj42"


def inspect_record(status: str = "completed", **changes: object) -> str:
    return json.dumps(
        {
            "id": JOB,
            "status": status,
            "command": dispatch_command(),
            "failure_reason": "worker failed",
            "host": "studio",
            **changes,
        }
    )


def native_artifact(*, prompt: str = "review") -> str:
    return WorkerResult(
        model_call_id=KEY,
        provider="codex",
        status="completed",
        worker_version="0.1.0",
        worker_protocol_version=WORKER_PROTOCOL_VERSION,
        worker_source_sha256=SOURCE,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        ctx_version="ctx-1",
        worker_cwd="/remote/project",
        started_at=10,
        completed_at=12,
        model_call_started=True,
        session_id="thread-42",
        harness=HarnessOutcome(0, "historical answer", ""),
    ).to_json()


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = 0.0
        self.calls: list[tuple[list[str], Path, float | None]] = []
        self.artifact_calls: list[CommandResult | OSError | subprocess.TimeoutExpired] = [
            CommandResult(0, worker_artifact(), "")
        ]
        self.inspect = CommandResult(0, inspect_record(), "")
        self.log = CommandResult(0, "worker log tail", "")
        self.mark: list[CommandResult | OSError | subprocess.TimeoutExpired] = [
            CommandResult(0, "", "")
        ]
        self.execution: dict[str, object] = {
            "transport": "weft",
            "host": "studio",
            "expected_worker_source_sha256": SOURCE,
            "prompt_sha256": hashlib.sha256(b"review").hexdigest(),
            "omp_selector": "anthropic/claude-opus-5",
        }
        self.runner = WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=None,
            invoke=self.invoke,
            clock=lambda: self.now,
            sleep=self.sleep,
        )

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def invoke(self, command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        self.calls.append((command, cwd, timeout))
        if command[1] == "artifact":
            self.assertFalse(cwd.resolve().is_relative_to(self.root.resolve()))
            outcome = self.artifact_calls.pop(0)
        elif command[1] == "job" and command[2] == "inspect":
            outcome = self.inspect
        elif command[1] == "log":
            outcome = self.log
        elif command[1] == "job" and command[2] == "mark-processed":
            outcome = self.mark.pop(0)
        else:
            raise AssertionError(command)
        if isinstance(outcome, (OSError, subprocess.TimeoutExpired)):
            raise outcome
        return outcome

    def processing(self) -> dict[str, object]:
        processing = self.execution["processing"]
        assert isinstance(processing, dict)
        return processing

    def diagnostics(self) -> dict[str, object]:
        diagnostics = self.execution["diagnostics"]
        assert isinstance(diagnostics, dict)
        return diagnostics

    def retrieve(
        self, *, timeout: float = 30.0
    ) -> CommandResult | WeftJobFailure | WeftRetrievalOutcome:
        return self.runner.retrieve(
            job_id=JOB, cwd=self.root, execution=self.execution, timeout=timeout
        )

    def test_valid_artifact_requires_explicit_acknowledgment(self) -> None:
        result = self.retrieve()
        self.assertIsInstance(result, CommandResult)
        assert isinstance(result, CommandResult)
        self.assertEqual(result.stdout, omp_output(cwd="/remote/project", final="remote answer"))
        self.assertEqual(self.processing()["state"], "pending")
        self.assertNotIn("processed", self.execution)
        result.mark_consumed()
        self.assertEqual(self.execution["processed"], True)
        self.assertEqual(self.processing()["state"], "marked")

    def test_acknowledgment_failure_is_visible_and_retryable(self) -> None:
        self.mark = [subprocess.TimeoutExpired(["weft", "job", "mark-processed"], 30)]
        result = self.retrieve()
        assert isinstance(result, CommandResult)
        result.mark_consumed()
        self.assertNotIn("processed", self.execution)
        self.assertEqual(self.processing()["state"], "mark_failed")
        self.mark = [CommandResult(1, "", "state rejected")]
        result.mark_consumed()
        self.assertEqual(self.execution["processed"], False)
        self.mark = [CommandResult(0, "", "")]
        result.mark_consumed()
        self.assertEqual(self.execution["processed"], True)
        self.assertEqual(self.processing()["state"], "marked")

    def test_retrieval_does_not_submit_or_fallback(self) -> None:
        result = self.retrieve()
        self.assertIsInstance(result, CommandResult)
        self.assertEqual([command[1] for command, _, _ in self.calls], ["artifact"])
        self.assertIsNone(self.runner.fallback)

    def test_mismatched_artifact_identity_is_not_consumed(self) -> None:
        wrong = WorkerResult.parse(worker_artifact())
        self.artifact_calls = [
            CommandResult(0, json.dumps({**wrong.to_dict(), "model_call_id": "other"}), "")
        ]
        result = self.retrieve()
        assert isinstance(result, CommandResult)
        self.assertEqual(result.exit_status, 1)
        self.assertEqual(self.processing()["step"], "worker_result_validation")
        result.mark_consumed()
        self.assertEqual(self.mark, [CommandResult(0, "", "")])

    def test_validation_boundaries_refuse_consumption(self) -> None:

        cases = [
            ("provider", "omp-packet", "worker_result_validation", {}),
            ("worker_protocol_version", 99, "worker_protocol_validation", {}),
            ("worker_source_sha256", "b" * 64, "worker_source_validation", {}),
            (
                "prompt_sha256",
                "0" * 64,
                "worker_result_validation",
                {"prompt_sha256": "0" * 64},
            ),
        ]
        for field, value, step, execution_changes in cases:
            with self.subTest(field=field):
                payload = json.loads(worker_artifact())
                payload[field] = value
                self.artifact_calls = [CommandResult(0, json.dumps(payload), "")]
                self.execution.update(execution_changes)
                result = self.retrieve()
                assert isinstance(result, CommandResult)
                self.assertEqual(result.exit_status, 1)
                self.assertEqual(self.processing()["state"], "not_marked")
                self.assertEqual(self.processing()["step"], step)
                result.mark_consumed()
                self.assertEqual(self.mark, [CommandResult(0, "", "")])

    def test_noncompleted_worker_result_keeps_bounded_diagnostics_unacknowledged(self) -> None:
        payload = json.loads(worker_artifact())
        payload["status"] = "harness_failed"
        payload["failure"] = "worker crashed"
        payload["harness"]["exit_status"] = 2
        payload["harness"]["stdout"] = "x" * 9000
        payload["harness"]["stderr"] = "e" * 9000
        self.artifact_calls = [CommandResult(0, json.dumps(payload), "")]
        result = self.retrieve()
        assert isinstance(result, CommandResult)
        self.assertEqual(result.exit_status, 1)
        stdout_tail = self.processing()["stdout_tail"]
        assert isinstance(stdout_tail, str)
        self.assertEqual(len(stdout_tail), 8000)
        result.mark_consumed()
        self.assertEqual(self.mark, [CommandResult(0, "", "")])

    def test_missing_source_is_recovered_from_accepted_command(self) -> None:
        del self.execution["expected_worker_source_sha256"]
        result = self.retrieve()
        self.assertIsInstance(result, CommandResult)
        self.assertEqual(self.execution["expected_worker_source_sha256"], SOURCE)
        self.assertEqual(
            [command[1:3] for command, _, _ in self.calls],
            [["artifact", "cat"], ["job", "inspect"]],
        )

    def test_untrusted_legacy_pin_is_rejected(self) -> None:
        del self.execution["expected_worker_source_sha256"]
        self.inspect = CommandResult(0, inspect_record(command=dispatch_command("other")), "")
        outcome = self.retrieve()
        self.assertIsInstance(outcome, WeftRetrievalOutcome)
        assert isinstance(outcome, WeftRetrievalOutcome)
        self.assertEqual(outcome.status, "unknown")

    def test_file_ingestion_uses_the_same_contract(self) -> None:
        artifact = self.root / "artifact.json"
        artifact.write_text(worker_artifact())
        self.runner = WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=None,
            invoke=self.invoke,
        )
        result = self.runner.retrieve_from_file(
            job_id=JOB,
            cwd=self.root,
            execution=self.execution,
            artifact_path=artifact,
        )
        self.assertIsInstance(result, CommandResult)
        self.assertEqual(self.calls, [])

    def test_native_file_ingestion_still_refuses_new_execution(self) -> None:
        execution: dict[str, object] = {"expected_worker_source_sha256": SOURCE}
        artifact = self.root / "artifact.json"
        artifact.write_text(native_artifact())
        self.runner = WeftCommandRunner(
            host="studio",
            agent="codex",
            model_call_id=KEY,
            fallback=None,
            invoke=self.invoke,
        )
        result = self.runner.retrieve_from_file(
            job_id=JOB,
            cwd=self.root,
            execution=execution,
            artifact_path=artifact,
        )
        self.assertIsInstance(result, CommandResult)
        with self.assertRaisesRegex(ValueError, "native Codex execution is unsupported"):
            self.runner(["codex", "exec", "review"], self.root, 30.0)

    def test_job_statuses_classify_absent_artifact_without_consumption(self) -> None:
        expected = {
            "running": "pending",
            "queued": "pending",
            "queued-start-time": "unknown",
            "completed": "unretrievable",
            "cancelled": "unretrievable",
            "malformed": "unretrievable",
        }
        for name, status in expected.items():
            with self.subTest(name=name):
                record = (
                    inspect_record("queued", start_time=10)
                    if name == "queued-start-time"
                    else inspect_record(name)
                )
                self.inspect = CommandResult(0, record, "")
                self.artifact_calls = [CommandResult(1, "", "not available")] * 4
                outcome = self.retrieve()
                assert isinstance(outcome, WeftRetrievalOutcome)
                self.assertEqual(outcome.status, status)

    def test_proven_failed_job_uses_inspect_cause_and_bounded_log(self) -> None:
        self.inspect = CommandResult(0, inspect_record("failed"), "")
        self.artifact_calls = [CommandResult(1, "", "not available")] * 4
        self.log = CommandResult(1, "", "e" * 1000)
        outcome = self.retrieve()
        self.assertIsInstance(outcome, WeftJobFailure)
        diagnostics = self.diagnostics()
        job_failure = diagnostics["job_failure"]
        job_log = diagnostics["job_failure_log"]
        assert isinstance(job_failure, dict)
        assert isinstance(job_log, dict)
        self.assertEqual(job_failure["failure_reason"], "worker failed")
        self.assertEqual(job_log["unobserved"], "exit 1")
        self.assertLessEqual(len(str(job_log.get("stderr", ""))), 1000)

    def test_artifact_retry_respects_deadline(self) -> None:
        self.artifact_calls = [
            CommandResult(1, "", "later"),
            CommandResult(0, worker_artifact(), ""),
        ]
        self.assertIsInstance(self.retrieve(timeout=30.0), CommandResult)
        artifact_timeouts = [
            timeout for command, _, timeout in self.calls if command[1] == "artifact"
        ]
        self.assertTrue(all(timeout is not None and timeout > 0 for timeout in artifact_timeouts))
        self.assertLessEqual(len(artifact_timeouts), 2)

    def test_transient_artifact_error_is_retryable(self) -> None:
        self.artifact_calls = [
            OSError("artifact raced publication"),
            CommandResult(0, worker_artifact(), ""),
        ]
        self.assertIsInstance(self.retrieve(timeout=30.0), CommandResult)
        artifact_timeouts = [
            timeout for command, _, timeout in self.calls if command[1] == "artifact"
        ]
        self.assertEqual(len(artifact_timeouts), 2)
        self.assertTrue(all(timeout is not None and timeout > 0 for timeout in artifact_timeouts))

    def test_inspect_identity_and_errors_are_not_verdicts(self) -> None:
        self.artifact_calls = [CommandResult(1, "", "not available")] * 4
        self.inspect = CommandResult(0, inspect_record(id="other"), "")
        outcome = self.retrieve()
        self.assertIsInstance(outcome, WeftRetrievalOutcome)
        assert isinstance(outcome, WeftRetrievalOutcome)
        self.assertEqual(outcome.status, "unknown")
        self.inspect = subprocess.TimeoutExpired(["weft", "job", "inspect"], 30)
        self.artifact_calls = [CommandResult(1, "", "not available")] * 4
        outcome = self.retrieve()
        self.assertIsInstance(outcome, WeftRetrievalOutcome)
        assert isinstance(outcome, WeftRetrievalOutcome)
        self.assertEqual(outcome.status, "unknown")
