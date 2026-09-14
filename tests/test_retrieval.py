"""Durable worker-result retrieval and explicit consumption acknowledgment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_execution.command import CommandResult
from agent_execution.identity import worker_evidence_path
from agent_execution.weft import (
    SUPPORTED_WORKER_PROTOCOL_VERSIONS,
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
        worker_protocol_version=1,
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
    def test_supported_worker_protocol_versions_are_explicit(self) -> None:
        self.assertEqual(SUPPORTED_WORKER_PROTOCOL_VERSIONS, frozenset({1, 2}))

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
            "expected_worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "worker_result_path": worker_evidence_path(KEY),
            "omp_policy": "read-only-no-shell",
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
        self.assertEqual(result.exit_status, 0, result.stderr)
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
            [["job", "inspect"], ["artifact", "cat"]],
        )

    def test_untrusted_legacy_pin_is_rejected(self) -> None:
        del self.execution["expected_worker_source_sha256"]
        self.inspect = CommandResult(0, inspect_record(command=dispatch_command("other")), "")
        outcome = self.retrieve()
        self.assertIsInstance(outcome, WeftRetrievalOutcome)
        assert isinstance(outcome, WeftRetrievalOutcome)
        self.assertEqual(outcome.status, "unknown")

    def test_historical_read_only_paths_follow_the_accepted_protocol(self) -> None:
        for path in (
            "outputs/agent-execution-worker-result.json",
            f"outputs/agent-execution-worker-result-{KEY}.json",
        ):
            with self.subTest(path=path):
                self.calls.clear()
                self.execution = {}
                command = dispatch_command().replace("--expect-protocol 2", "--expect-protocol 1")
                command = command.replace(" -- ", f" --evidence-out {path} -- ", 1)
                self.inspect = CommandResult(0, inspect_record(command=command), "")
                self.artifact_calls = [CommandResult(0, worker_artifact(protocol=1), "")]
                result = self.retrieve()
                assert isinstance(result, CommandResult)
                self.assertEqual(result.exit_status, 0, result.stderr)
                self.assertEqual(self.execution["expected_worker_protocol_version"], 1)
                self.assertEqual(self.calls[-1][0][-1], path)
                summary = self.execution["worker_result"]
                assert isinstance(summary, dict)
                self.assertEqual(summary["artifact_path"], path)

    def test_retired_selector_recovers_without_authorizing_new_execution(self) -> None:
        selector = "anthropic/claude-opus-4-6"
        path = f"outputs/agent-execution-worker-result-{KEY}.json"
        self.execution.pop("expected_worker_protocol_version")
        self.execution.pop("worker_result_path")
        self.execution["omp_selector"] = selector
        command = dispatch_command().replace("--expect-protocol 2", "--expect-protocol 1")
        command = command.replace("anthropic/claude-opus-5", selector)
        command = command.replace(" -- ", f" --evidence-out {path} -- ", 1)
        self.inspect = CommandResult(0, inspect_record(command=command), "")
        self.artifact_calls = [CommandResult(0, worker_artifact(protocol=1, selector=selector), "")]
        result = self.retrieve()
        self.assertIsInstance(result, CommandResult)
        assert isinstance(result, CommandResult)
        self.assertEqual(result.exit_status, 0, result.stderr)
        self.assertEqual(self.calls[-1][0][-1], path)
        from agent_execution.omp_execution import validate_omp_command
        from tests.support.omp import omp_command

        with self.assertRaises(ValueError):
            validate_omp_command(omp_command(".", selector=selector), Path("."))

    def test_result_cannot_downgrade_the_retained_protocol(self) -> None:
        self.artifact_calls = [CommandResult(0, worker_artifact(protocol=1), "")]
        result = self.retrieve()
        assert isinstance(result, CommandResult)
        self.assertEqual(result.exit_status, 1)
        self.assertEqual(self.processing()["step"], "worker_protocol_validation")
        self.assertEqual(self.execution["expected_worker_protocol_version"], 2)

    def test_result_claims_cannot_supply_missing_dispatch_protocol(self) -> None:
        del self.execution["expected_worker_protocol_version"]
        self.inspect = CommandResult(
            0, inspect_record(command=dispatch_command().replace("--expect-protocol 2 ", "")), ""
        )
        outcome = self.retrieve()
        assert isinstance(outcome, WeftRetrievalOutcome)
        self.assertEqual(outcome.status, "unknown")
        self.assertEqual([argv[1:3] for argv, _, _ in self.calls], [["job", "inspect"]])

    def test_unprotected_legacy_paths_and_legacy_writer_results_are_unknown(self) -> None:
        for path, policy in (
            ("outputs/custom.json", "read-only-no-shell"),
            (f"outputs/agent-execution-worker-result-{KEY}.json", "workspace-write-no-shell"),
        ):
            with self.subTest(path=path, policy=policy):
                self.execution.update(
                    expected_worker_protocol_version=1, worker_result_path=path, omp_policy=policy
                )
                outcome = self.retrieve()
                assert isinstance(outcome, WeftRetrievalOutcome)
                self.assertEqual(outcome.status, "unknown")
                self.assertEqual(self.calls, [])

    def test_protocol_two_rejects_accepted_caller_selected_path(self) -> None:
        self.execution = {}
        command = dispatch_command().replace(" -- ", " --evidence-out outputs/custom.json -- ", 1)
        self.inspect = CommandResult(0, inspect_record(command=command), "")
        outcome = self.retrieve()
        assert isinstance(outcome, WeftRetrievalOutcome)
        self.assertEqual(outcome.status, "unknown")
        self.assertIn("does not accept --evidence-out", outcome.detail)

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

    def test_writer_file_ingestion_requires_explicit_dispatch_authority(self) -> None:
        artifact = self.root / "writer-artifact.json"
        artifact.write_text(
            worker_artifact(writer=True, selector="openai-codex/gpt-6-astra"),
            encoding="utf-8",
        )
        self.execution["omp_selector"] = "openai-codex/gpt-6-astra"
        # Historical receipts do not grant writes, even for a writer-capable
        # provider returning a valid writer transcript.
        refused = self.runner.retrieve_from_file(
            job_id=JOB,
            cwd=self.root,
            execution=self.execution,
            artifact_path=artifact,
        )
        assert isinstance(refused, CommandResult)
        self.assertEqual(refused.exit_status, 1)
        refused.mark_consumed()
        self.assertEqual(self.calls, [])

        self.execution["omp_policy"] = "workspace-write-no-shell"
        accepted = self.runner.retrieve_from_file(
            job_id=JOB,
            cwd=self.root,
            execution=self.execution,
            artifact_path=artifact,
        )
        assert isinstance(accepted, CommandResult)
        self.assertEqual(accepted.exit_status, 0, accepted.stderr)
        self.assertIn("remote answer", accepted.stdout)
        accepted.mark_consumed()
        self.assertTrue(self.execution["processed"])

    def test_recorded_policy_mismatches_refuse_consumption(self) -> None:
        self.execution["omp_selector"] = "openai-codex/gpt-6-astra"
        for writer, policy in (
            (True, "read-only-no-shell"),
            (False, "workspace-write-no-shell"),
            (False, None),
        ):
            with self.subTest(writer=writer, policy=policy):
                self.execution["omp_policy"] = policy
                self.artifact_calls = [
                    CommandResult(
                        0,
                        worker_artifact(writer=writer, selector="openai-codex/gpt-6-astra"),
                        "",
                    )
                ]
                result = self.retrieve()
                assert isinstance(result, CommandResult)
                self.assertEqual(result.exit_status, 1)
                self.assertEqual(self.processing()["state"], "not_marked")
                result.mark_consumed()
                self.assertEqual(self.mark, [CommandResult(0, "", "")])

    def test_native_file_ingestion_still_refuses_new_execution(self) -> None:
        execution: dict[str, object] = {
            "expected_worker_source_sha256": SOURCE,
            "expected_worker_protocol_version": 1,
            "worker_result_path": f"outputs/agent-execution-worker-result-{KEY}.json",
        }
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

    def test_status_inspection_has_an_independent_deadline(self) -> None:
        self.artifact_calls = [CommandResult(1, "", "not available")] * 4
        invoke = self.runner.invoke

        def delayed_artifact(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            result = invoke(command, cwd, timeout)
            if command[1] == "artifact":
                self.now += 30.0
            return result

        self.runner.invoke = delayed_artifact
        outcome = self.retrieve(timeout=30.0)
        self.assertIsInstance(outcome, WeftRetrievalOutcome)
        assert isinstance(outcome, WeftRetrievalOutcome)
        self.assertEqual(outcome.status, "unretrievable")
        inspect_timeouts = [
            timeout for command, _, timeout in self.calls if command[1:3] == ["job", "inspect"]
        ]
        self.assertEqual(len(inspect_timeouts), 1)
        self.assertIsNotNone(inspect_timeouts[0])
        assert inspect_timeouts[0] is not None
        self.assertGreater(inspect_timeouts[0], 0)

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
