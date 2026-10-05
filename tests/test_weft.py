"""Weft admission contracts using synthetic responses at the external CLI boundary."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_execution.command import CommandResult
from agent_execution.identity import source_worker_executable, worker_evidence_path
from agent_execution.omp_execution import omp_transcript
from agent_execution.weft import (
    ARTIFACT_RETRIEVAL_SECONDS,
    LOST_OBSERVATION_PROBE_SECONDS,
    SubmissionLookup,
    WeftAdmissionCancelled,
    WeftCommandRunner,
    WeftExecutionAmbiguous,
    WeftExecutionDetached,
    WeftPlacementRefused,
    WeftRunReceipt,
    lookup_submitted_job,
    parse_worker_command,
    submitted_job_from_listing,
)
from agent_execution.worker import WORKER_PROTOCOL_VERSION, HarnessOutcome, WorkerResult
from tests.support.omp import omp_command, omp_output

KEY = "model-call-7"
SOURCE = "a" * 64


def receipt(decision: str = "accepted_immediately", **changes: object) -> str:
    value = {
        "api_version": "weft.run.receipt.v1",
        "job_id": "" if decision == "not_accepted" else "wj42",
        "placement_decision": decision,
        "selected_host": "studio",
        "source_pin": "" if decision == "not_accepted" else "pin-42",
        "accepted_immediately": decision == "accepted_immediately",
        "deduplicated": decision == "deduplicated",
        "idempotency_key": KEY,
    }
    return json.dumps({**value, **changes})


def listing(*rows: dict[str, object], version: object = 1) -> str:
    return json.dumps({"kind": "job_list", "version": version, "jobs": rows})


def dispatch_command(key: str = KEY) -> str:
    return shlex.join(
        [
            "agent-execution-worker",
            "execute",
            "--provider",
            "omp",
            "--model-call-id",
            key,
            "--expect-source-sha256",
            SOURCE,
            "--expect-protocol",
            str(WORKER_PROTOCOL_VERSION),
            "--expect-prompt-sha256",
            hashlib.sha256(b"review").hexdigest(),
            "--",
            *omp_command(".", prompt="-"),
        ]
    )


def worker_artifact(
    prompt: str = "review",
    *,
    packet: bool = False,
    writer: bool = False,
    selector: str = "anthropic/claude-opus-5-5",
    protocol: int = WORKER_PROTOCOL_VERSION,
) -> str:
    stdout = omp_output(
        cwd="/remote/project",
        selector=selector,
        prompt=prompt,
        final="remote answer",
        policy="packet-only-no-tools"
        if packet
        else "workspace-write-no-shell"
        if writer
        else "read-only-no-shell",
    )
    return WorkerResult(
        model_call_id=KEY,
        provider="omp-packet" if packet else "omp",
        status="completed",
        worker_version="0.1.0",
        worker_protocol_version=protocol,
        worker_source_sha256=SOURCE,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        ctx_version="",
        worker_cwd="/remote/project",
        started_at=10,
        completed_at=12,
        model_call_started=True,
        session_id="" if packet else "session-1",
        harness=HarnessOutcome(0, stdout, ""),
        omp_evidence=None if packet else omp_transcript(stdout),
    ).to_json()


class SubmissionLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.executable = self.root / "weft"

    def lookup(self, body: str, *, exit_status: int = 0) -> SubmissionLookup:
        self.executable.write_text(
            "#!/bin/sh\n"
            f"test \"$1 $2 $3 $4\" = 'job lookup --idempotency-key {KEY}' || exit 7\n"
            f"printf '%s' {shlex.quote(body)}\n"
            f"exit {exit_status}\n",
            encoding="utf-8",
        )
        self.executable.chmod(0o755)
        # These tests read the stub's output, not its timing. macOS scans a
        # freshly written executable on first exec, measured at ~2.5s on studio,
        # so a 1s budget timed out every case there.
        return lookup_submitted_job(KEY, timeout=30, cwd=self.root, executable=str(self.executable))

    def test_found_returns_receipt_job_and_host(self) -> None:
        result = self.lookup(
            json.dumps(
                {
                    "api_version": "weft.job.lookup.v1",
                    "idempotency_key": KEY,
                    "found": True,
                    "receipt": json.loads(receipt("deduplicated")),
                }
            )
        )
        self.assertEqual(result.status, "found")
        self.assertEqual(result.job_id, "wj42")
        self.assertEqual(result.host, "studio")

    def test_only_validated_negative_answer_is_absent(self) -> None:
        result = self.lookup(
            json.dumps(
                {"api_version": "weft.job.lookup.v1", "idempotency_key": KEY, "found": False}
            )
        )
        self.assertEqual(result.status, "absent")
        self.assertIsNone(result.job_id)

    def test_failed_lookup_is_unknown(self) -> None:
        result = self.lookup("", exit_status=4)
        self.assertEqual(result.status, "unknown")
        self.assertIn("exited 4", result.reason or "")

    def test_timed_out_lookup_is_unknown(self) -> None:
        self.executable.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
        self.executable.chmod(0o755)
        result = lookup_submitted_job(
            KEY, timeout=0.01, cwd=self.root, executable=str(self.executable)
        )
        self.assertEqual(result.status, "unknown")
        self.assertIn("TimeoutExpired", result.reason or "")

    def test_malformed_json_is_unknown(self) -> None:
        result = self.lookup("{invalid")
        self.assertEqual(result.status, "unknown")
        self.assertIn("invalid JSON", result.reason or "")

    def test_unknown_version_cannot_prove_absence(self) -> None:
        result = self.lookup(
            json.dumps(
                {"api_version": "weft.job.lookup.v2", "idempotency_key": KEY, "found": False}
            )
        )
        self.assertEqual(result.status, "unknown")
        self.assertIn("api_version", result.reason or "")

    def test_missing_executable_is_unknown(self) -> None:
        result = lookup_submitted_job(
            KEY, timeout=1, cwd=self.root, executable=str(self.root / "missing-weft")
        )
        self.assertEqual(result.status, "unknown")
        self.assertIn("FileNotFoundError", result.reason or "")

    def test_mismatched_key_is_unknown(self) -> None:
        result = self.lookup(
            json.dumps(
                {"api_version": "weft.job.lookup.v1", "idempotency_key": "other", "found": False}
            )
        )
        self.assertEqual(result.status, "unknown")
        self.assertIn("idempotency key", result.reason or "")


class AdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = 0.0
        self.submit_elapsed = 0.0
        self.submissions: list[CommandResult | subprocess.TimeoutExpired] = []
        self.calls: list[tuple[list[str], Path, float | None]] = []
        self.payloads: list[str] = []
        self.local_calls: list[list[str]] = []
        self.cancel = False
        self.hosts = CommandResult(
            0,
            json.dumps(
                {
                    "kind": "host_list",
                    "version": 1,
                    "hosts": [
                        {
                            "name": "studio",
                            "capabilities": ["agent:omp", "tool:agent-execution"],
                        }
                    ],
                }
            ),
            "",
        )
        self.jobs = listing()
        self.status: CommandResult | subprocess.TimeoutExpired = CommandResult(0, "completed", "")
        self.artifact = CommandResult(0, worker_artifact(), "")
        self.runner = WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=self.fallback,
            invoke=self.invoke,
            clock=lambda: self.now,
            sleep=self.sleep,
            admission_wait=self.wait,
            expected_source_sha256=SOURCE,
        )

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def wait(self, seconds: float) -> bool:
        self.sleep(seconds)
        return self.cancel

    def fallback(self, command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        self.local_calls.append(command)
        return CommandResult(0, "local answer", "")

    def invoke(self, command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        self.calls.append((command, cwd, timeout))
        if command[1] == "host":
            return self.hosts
        if command[1] == "run":
            payload = command[command.index("--payload") + 1].split("=", 1)[1]
            self.payloads.append(Path(payload).read_text(encoding="utf-8"))
            self.now += self.submit_elapsed
            outcome = self.submissions.pop(0)
            if isinstance(outcome, subprocess.TimeoutExpired):
                raise outcome
            return outcome
        if command[1:3] == ["list", "jobs"]:
            return CommandResult(0, self.jobs, "")
        if command[1:3] == ["job", "list"]:
            return CommandResult(0, self.jobs, "")
        if command[1] == "status":
            if isinstance(self.status, subprocess.TimeoutExpired):
                raise self.status
            return self.status
        if command[1] == "artifact":
            self.assertFalse(cwd.resolve().is_relative_to(self.root.resolve()))
            return self.artifact
        if command[1] == "log":
            return CommandResult(0, "", "")
        if command[1:3] == ["job", "mark-processed"]:
            return CommandResult(0, "", "")
        raise AssertionError(command)

    def run_dispatch(
        self,
        *,
        timeout: float | None = 30.0,
        prompt: str = "review",
        packet: bool = False,
    ) -> CommandResult:
        return self.runner(
            omp_command(str(self.root), prompt=prompt, packet=packet),
            self.root,
            timeout,
        )

    def assert_remote(self, result: CommandResult) -> None:
        self.assertEqual(result.exit_status, 0, result.stderr)
        self.assertIn("remote answer", result.stdout)
        self.assertEqual(self.local_calls, [])
        self.assertIsNotNone(result.execution)
        assert result.execution is not None
        self.assertEqual(result.execution["job_id"], "wj42")

    def test_accepted_receipt_survives_nonzero_submit_exit(self) -> None:
        self.submissions = [CommandResult(-15, receipt(), "reader interrupted")]
        self.assert_remote(self.run_dispatch())

    def test_protocol_two_publishes_and_retrieves_the_protected_call_path(self) -> None:
        self.submissions = [CommandResult(0, receipt(), "")]
        result = self.run_dispatch()
        self.assert_remote(result)
        submitted = next(argv for argv, _, _ in self.calls if argv[1] == "run")
        self.assertIn("--if-online", submitted)
        self.assertEqual(
            submitted[submitted.index("--produces") + 1],
            ".agent-execution/results/model-call-7.json",
        )
        remote = shlex.split(submitted[-1])
        self.assertNotIn("--evidence-out", remote)
        self.assertEqual(remote[remote.index("--expect-protocol") + 1], "2")
        artifact = next(argv for argv, _, _ in self.calls if argv[1] == "artifact")
        self.assertEqual(artifact[-1], worker_evidence_path(KEY))
        assert result.execution is not None
        self.assertEqual(result.execution["expected_worker_protocol_version"], 2)
        summary = result.execution["worker_result"]
        assert isinstance(summary, dict)
        self.assertEqual(summary["artifact_path"], artifact[-1])

    def test_the_job_is_filed_under_the_consumers_project(self) -> None:
        self.submissions = [CommandResult(0, receipt(), "")]
        self.run_dispatch()
        submitted = next(argv for argv, _, _ in self.calls if argv[1] == "run")
        self.assertEqual(submitted[submitted.index("--project") + 1], "agent-execution")

        self.calls.clear()
        self.runner = WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=self.fallback,
            invoke=self.invoke,
            clock=lambda: self.now,
            sleep=self.sleep,
            admission_wait=self.wait,
            expected_source_sha256=SOURCE,
            project="agent-review",
        )
        self.submissions = [CommandResult(0, receipt(), "")]
        self.run_dispatch()
        submitted = next(argv for argv, _, _ in self.calls if argv[1] == "run")
        self.assertEqual(submitted[submitted.index("--project") + 1], "agent-review")

    def test_the_job_reserves_one_core_rather_than_weft_s_default(self) -> None:
        """Undeclared CPU books 7 of studio's 12 cores, admitting one job at a time."""
        self.submissions = [CommandResult(0, receipt(), "")]
        self.run_dispatch()
        submitted = next(argv for argv, _, _ in self.calls if argv[1] == "run")
        self.assertEqual(submitted[submitted.index("--cpu-reserve") + 1], "1")
        # A Weft option, not part of the worker command Weft executes.
        self.assertLess(submitted.index("--cpu-reserve"), submitted.index("-m"))

    def test_queue_enabled_submission_does_not_require_an_online_host(self) -> None:
        self.runner = WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=self.fallback,
            invoke=self.invoke,
            clock=lambda: self.now,
            sleep=self.sleep,
            admission_wait=self.wait,
            expected_source_sha256=SOURCE,
            allow_queue=True,
        )
        self.submissions = [CommandResult(0, receipt("queued"), "")]

        self.assert_remote(self.run_dispatch())

        submitted = next(argv for argv, _, _ in self.calls if argv[1] == "run")
        self.assertNotIn("--if-online", submitted)

    def test_tracked_result_survives_waiter_exit_and_later_completion(self) -> None:
        self.submissions = [CommandResult(0, receipt(), "")]
        self.status = subprocess.TimeoutExpired(["weft", "status"], 30)
        completed = False
        outputs: set[str] = set()

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            if command[1] == "run":
                # Weft DiscoverJobOutputsSince uses Outputs plus convention
                # directories; Produces is a separate manifest/dependency list.
                outputs.update(
                    command[index + 1]
                    for index, arg in enumerate(command[:-1])
                    if arg == "--output"
                )
            if command[1:3] == ["artifact", "cat"]:
                self.artifact = (
                    CommandResult(0, worker_artifact(), "")
                    if completed and command[-1] in outputs
                    else CommandResult(1, "", "no tracked output")
                )
            if command[1:3] == ["job", "inspect"]:
                self.calls.append((command, cwd, timeout))
                return CommandResult(
                    0,
                    json.dumps({"id": "wj42", "status": "completed"}),
                    "",
                )
            return self.invoke(command, cwd, timeout)

        self.runner.invoke = invoke
        with self.assertRaises(WeftExecutionDetached):
            self.run_dispatch()
        assert self.runner.last_execution is not None
        execution = json.loads(json.dumps(self.runner.last_execution))

        # The worker finishes after the original waiter is gone. Only declared
        # tracked outputs are available at this synthetic external CLI boundary.
        completed = True
        consumer = WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=None,
            invoke=invoke,
            clock=lambda: self.now,
            sleep=self.sleep,
        )
        self.calls.clear()
        result = consumer.retrieve(job_id="wj42", cwd=self.root, execution=execution)
        self.assertIsInstance(result, CommandResult, result)
        assert isinstance(result, CommandResult)
        self.assert_remote(result)
        self.assertTrue(all(argv[1:3] == ["artifact", "cat"] for argv, _, _ in self.calls))
        self.assertEqual(self.payloads, ["review"])

    def test_writer_dispatch_receipt_survives_retrieval_and_recovery(self) -> None:
        self.runner = WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=self.fallback,
            invoke=self.invoke,
            clock=lambda: self.now,
            sleep=self.sleep,
            expected_source_sha256=SOURCE,
            worker_executable=source_worker_executable(SOURCE),
        )
        for route in ("receipt", "probe", "submission_timeout", "lost_watcher"):
            with self.subTest(route=route):
                self.calls.clear()
                self.submissions = [
                    subprocess.TimeoutExpired(["weft", "run"], 30)
                    if route == "submission_timeout"
                    else CommandResult(0, "unreadable" if route == "probe" else receipt(), "")
                ]
                self.status = CommandResult(
                    -15 if route == "lost_watcher" else 0, "", "watcher interrupted"
                )
                self.artifact = CommandResult(
                    0,
                    worker_artifact(writer=True, selector="openai-codex/gpt-6-astra"),
                    "",
                )

                def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
                    if command[1:3] == ["list", "jobs"]:
                        submitted = next(argv for argv, _, _ in self.calls if argv[1] == "run")
                        self.jobs = listing({"job_id": "wj42", "command": submitted[-1]})
                    return self.invoke(command, cwd, timeout)

                self.runner.invoke = invoke
                command = omp_command(
                    str(self.root), writer=True, selector="openai-codex/gpt-6-astra"
                )
                if route == "submission_timeout":
                    with self.assertRaises(WeftExecutionDetached):
                        self.runner(command, self.root, 30.0)
                else:
                    self.assert_remote(self.runner(command, self.root, 30.0))
                assert self.runner.last_execution is not None
                execution = json.loads(json.dumps(self.runner.last_execution))
                # A new consumer must use only the durable dispatch receipt,
                # not the original runner's in-memory invocation policy.
                consumer = WeftCommandRunner(
                    host="studio",
                    agent="omp",
                    model_call_id=KEY,
                    fallback=None,
                    invoke=self.invoke,
                    clock=lambda: self.now,
                    sleep=self.sleep,
                )
                result = consumer.retrieve(job_id="wj42", cwd=self.root, execution=execution)
                assert isinstance(result, CommandResult)
                self.assert_remote(result)
                result.mark_consumed()
                self.assertTrue(execution["processed"])

    def test_receipt_at_deadline_recovers_with_fresh_artifact_budget(self) -> None:
        self.submit_elapsed = 30.0
        self.submissions = [CommandResult(0, receipt(), "")]
        self.assert_remote(self.run_dispatch())
        self.assertFalse(any(command[1] == "status" for command, _, _ in self.calls))
        artifact_budgets = [budget for command, _, budget in self.calls if command[1] == "artifact"]
        # The property is that the fetch gets a FRESH budget rather than the
        # submission's exhausted deadline; the size is the retrieval constant,
        # which is deliberately larger than a status probe's (wb139).
        self.assertEqual(artifact_budgets, [ARTIFACT_RETRIEVAL_SECONDS])

    def test_receipt_at_deadline_without_artifact_detaches(self) -> None:
        self.submit_elapsed = 30.0
        self.submissions = [CommandResult(0, receipt(), "")]
        self.artifact = CommandResult(1, "", "not available")
        with self.assertRaises(WeftExecutionDetached):
            self.run_dispatch()
        assert self.runner.last_execution is not None
        self.assertEqual(self.runner.last_execution["job_id"], "wj42")
        self.assertEqual(self.local_calls, [])

    def test_malformed_receipt_types_still_probe_and_recover(self) -> None:
        self.submissions = [CommandResult(0, receipt(placement_decision=[]), "")]
        self.jobs = listing({"job_id": "wj42", "command": dispatch_command()})
        self.assert_remote(self.run_dispatch())

    def test_submission_timeout_preserves_exact_job_and_never_retries(self) -> None:
        self.submissions = [subprocess.TimeoutExpired(["weft", "run"], 30)]
        self.jobs = listing({"job_id": "wj42", "command": dispatch_command()})
        with self.assertRaises(WeftExecutionDetached):
            self.run_dispatch()
        assert self.runner.last_execution is not None
        self.assertEqual(self.runner.last_execution["job_id"], "wj42")
        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(self.local_calls, [])

    def test_description_mention_cannot_establish_dispatch_ownership(self) -> None:
        self.submissions = [CommandResult(1, "not json", "")]
        self.jobs = listing(
            {
                "job_id": "wj-other",
                "description": f"investigate {KEY}",
                "command": "echo report",
            }
        )
        with self.assertRaises(WeftExecutionAmbiguous):
            self.run_dispatch()
        self.assertFalse(any(command[1] == "status" for command, _, _ in self.calls))
        self.assertEqual(self.local_calls, [])

    def test_invalid_rejection_never_authorizes_retry_or_fallback(self) -> None:
        rejection = {"code": "host_offline", "detail": "studio is offline"}
        invalid = [
            receipt("not_accepted", rejection=None),
            receipt("not_accepted", rejection={"code": "host_offline", "detail": "\ud800"}),
            receipt("not_accepted", rejection=rejection, job_id="wj42"),
            receipt(rejection=rejection),
            receipt("deduplicated", rejection=rejection),
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                self.payloads.clear()
                self.calls.clear()
                self.local_calls.clear()
                self.submissions = [CommandResult(1, raw, "unmodified refusal evidence")] * 3
                with self.assertRaises(WeftExecutionAmbiguous):
                    self.run_dispatch()
                self.assertEqual(len(self.payloads), 1)
                self.assertEqual(self.local_calls, [])
                assert self.runner.last_execution is not None
                diagnostics = self.runner.last_execution["diagnostics"]
                assert isinstance(diagnostics, dict)
                self.assertEqual(diagnostics["receipt_stdout"], raw)
                self.assertEqual(diagnostics["receipt_stderr"], "unmodified refusal evidence")

    def test_rejection_then_acceptance_retains_diagnostics_without_local_execution(self) -> None:
        rejection = {"code": "admission_race_lost", "detail": "another caller won admission"}
        self.submissions = [
            CommandResult(1, receipt("not_accepted", rejection=rejection), "raw admission error"),
            CommandResult(0, receipt("deduplicated"), ""),
        ]
        result = self.run_dispatch()
        self.assert_remote(result)
        self.assertEqual(len(self.payloads), 2)
        assert result.execution is not None
        execution = json.loads(json.dumps(result.execution))
        attempts = execution["admission"]["attempts"]
        self.assertEqual(attempts[0]["receipt"]["rejection"], rejection)
        self.assertEqual(attempts[0]["submission_stderr"], "raw admission error")
        self.assertNotIn("rejection", execution["receipt"])
        self.assertEqual(execution["admission"]["final_placement"]["transport"], "weft")

    def test_rejection_retry_is_cancelled_before_next_submission(self) -> None:
        self.submissions = [CommandResult(1, receipt("not_accepted"), "offline")]
        self.cancel = True
        with self.assertRaises(WeftAdmissionCancelled):
            self.run_dispatch()
        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(self.local_calls, [])

    def test_rejection_diagnostics_survive_retries_and_local_fallback_timeout(self) -> None:
        long_reason = "\n".join(f"remote reason {index}" for index in range(250))
        first_rejection = {
            "code": "host_constraints_unsatisfied",
            "detail": "studio lacks the required capability",
        }
        final_rejection = {"code": "future_capacity_reason", "detail": "界" * 341 + "!"}
        self.submissions = [
            CommandResult(
                1,
                receipt("not_accepted", rejection=first_rejection),
                "host constraint mismatch",
            ),
            CommandResult(1, receipt("not_accepted"), "offline probe failed"),
            CommandResult(1, receipt("not_accepted", rejection=final_rejection), long_reason),
        ]

        def timed_out(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            self.local_calls.append(command)
            raise TimeoutError("local admission expired")

        self.runner.fallback = timed_out

        with self.assertRaisesRegex(TimeoutError, "local admission expired"):
            self.run_dispatch()

        assert self.runner.last_execution is not None
        admission = self.runner.last_execution["admission"]
        assert isinstance(admission, dict)
        attempts = admission["attempts"]
        assert isinstance(attempts, list)
        self.assertEqual(attempts[0]["receipt"]["rejection"], first_rejection)
        self.assertNotIn("rejection", attempts[1]["receipt"])
        self.assertEqual(attempts[2]["receipt"]["rejection"], final_rejection)
        self.assertEqual(
            [attempt["submission_stderr"] for attempt in attempts[:2]],
            ["host constraint mismatch", "offline probe failed"],
        )
        retained = attempts[2]["submission_stderr"].splitlines()
        self.assertEqual(len(retained), 200)
        self.assertEqual(retained[0], "remote reason 50")
        self.assertEqual(retained[-1], "remote reason 249")
        self.assertEqual(len(self.local_calls), 1)

    def test_admission_sequences_obey_no_double_execution(self) -> None:
        # Exhaust the bounded event model. R retries; A owns remotely; U is unknown.
        # The oracle is the first non-rejection, independently of production state.
        counts = {"local": 0, "remote": 0, "unknown": 0}
        for events in itertools.product("RAU", repeat=3):
            with self.subTest(events=events):
                self.calls.clear()
                self.local_calls.clear()
                self.payloads.clear()
                self.now = 0
                self.submissions = [
                    CommandResult(
                        0,
                        {
                            "R": receipt("not_accepted"),
                            "A": receipt(),
                            "U": "not json",
                        }[event],
                        "",
                    )
                    for event in events
                ]
                first = next((i for i, event in enumerate(events) if event != "R"), 3)
                expected = (
                    "local" if first == 3 else "remote" if events[first] == "A" else "unknown"
                )
                counts[expected] += 1
                if expected == "unknown":
                    with self.assertRaises(WeftExecutionAmbiguous):
                        self.run_dispatch()
                else:
                    result = self.run_dispatch()
                    if expected == "remote":
                        self.assert_remote(result)
                    else:
                        self.assertEqual(result.stdout, "local answer")
                self.assertEqual(len(self.local_calls), int(expected == "local"))
                self.assertEqual(len(self.payloads), min(first + 1, 3))
                keys = {
                    command[command.index("--idempotency-key") + 1]
                    for command, _, _ in self.calls
                    if command[1] == "run"
                }
                self.assertEqual(keys, {KEY})
        self.assertEqual(counts, {"local": 1, "remote": 13, "unknown": 13})

    def test_deduplication_and_unbounded_harness_keep_remote_ownership(self) -> None:
        self.submissions = [CommandResult(0, receipt("deduplicated"), "")]
        self.assert_remote(self.run_dispatch(timeout=None))
        self.assertTrue(
            all(timeout is not None and 0 < timeout <= 900 for _, _, timeout in self.calls)
        )

    def test_known_placement_and_deployment_refusals_do_not_dispatch(self) -> None:
        self.hosts = CommandResult(
            0, json.dumps({"kind": "host_list", "version": 1, "hosts": []}), ""
        )
        with self.assertRaises(WeftPlacementRefused):
            self.run_dispatch()
        self.assertEqual(self.local_calls, [])
        self.assertEqual(self.payloads, [])
        self.hosts = CommandResult(1, "", "inventory unavailable")
        self.runner.readiness = lambda host: ("mismatch", "protocol differs")
        with self.assertRaises(WeftPlacementRefused):
            self.run_dispatch()
        self.assertEqual(self.payloads, [])

    def test_unknown_placement_proceeds_to_atomic_admission(self) -> None:
        self.hosts = CommandResult(1, "", "inventory unavailable")
        self.submissions = [CommandResult(0, receipt(), "")]
        self.assert_remote(self.run_dispatch())

    def test_observed_host_requires_exact_integer_listing_version(self) -> None:
        row: dict[str, object] = {"job_id": "wj42", "host": "wi7777"}
        self.jobs = listing(row)
        execution: dict[str, object] = {"host": "studio"}
        self.runner.record_observed_host("wj42", self.root, execution)
        observation = execution["host_observation"]
        self.assertIsInstance(observation, dict)
        assert isinstance(observation, dict)
        self.assertEqual(observation["observed_host"], "wi7777")
        self.assertEqual(observation["differs_from_requested"], "studio")

        for version in (True, 1.0):
            with self.subTest(version=version):
                self.jobs = listing(row, version=version)
                execution = {"host": "studio"}
                self.runner.record_observed_host("wj42", self.root, execution)
                observation = execution["host_observation"]
                self.assertIsInstance(observation, dict)
                assert isinstance(observation, dict)
                self.assertEqual(observation["unobserved"], f"listing version {version!r}")
                self.assertNotIn("observed_host", observation)

    def test_observed_host_requires_string_job_and_host_fields(self) -> None:
        cases: tuple[tuple[dict[str, object], str, str], ...] = (
            ({"job_id": 42, "host": "studio"}, "42", "job not in listing window"),
            ({"job_id": "wj42", "host": 42}, "wj42", "row carries no host"),
            ({"job_id": "wj42", "host": ["studio"]}, "wj42", "row carries no host"),
        )
        for row, job_id, reason in cases:
            with self.subTest(row=row):
                self.jobs = listing(row)
                execution: dict[str, object] = {"host": "studio"}
                self.runner.record_observed_host(job_id, self.root, execution)
                observation = execution["host_observation"]
                self.assertIsInstance(observation, dict)
                assert isinstance(observation, dict)
                self.assertEqual(observation["unobserved"], reason)
                self.assertNotIn("observed_host", observation)

    def test_broken_watcher_recovers_or_detaches_without_fallback(self) -> None:
        self.submissions = [CommandResult(0, receipt(), "")]
        self.status = CommandResult(-15, "", "reader interrupted")
        self.assert_remote(self.run_dispatch())
        self.submissions = [CommandResult(0, receipt(), "")]
        self.status = subprocess.TimeoutExpired(["weft", "status"], 30)
        self.artifact = CommandResult(1, "", "not available")
        with self.assertRaises(WeftExecutionDetached):
            self.run_dispatch()
        self.assertEqual(self.local_calls, [])

    def test_packet_prompt_is_payload_only_and_model_flag_is_lifted(self) -> None:
        prompt = "private 'brief'\n模型 --model another/model $(false)"
        self.runner.agent = "omp-packet"
        self.submissions = [CommandResult(0, receipt(), "")]
        self.artifact = CommandResult(0, worker_artifact(prompt, packet=True), "")
        self.assert_remote(self.run_dispatch(prompt=prompt, packet=True))
        self.assertEqual(self.payloads, [prompt])
        submitted = next(command for command, _, _ in self.calls if command[1] == "run")
        self.assertNotIn(prompt, " ".join(submitted))
        worker = shlex.split(submitted[-1])
        self.assertEqual(worker[0], "agent-execution-worker")
        self.assertNotIn("--model", worker)
        self.assertIn("--harness-model", worker)
        self.assertEqual(
            worker[worker.index("--expect-prompt-sha256") + 1],
            hashlib.sha256(prompt.encode()).hexdigest(),
        )


class ReceiptTests(unittest.TestCase):
    def test_supported_decisions_round_trip(self) -> None:
        for decision in ("accepted_immediately", "queued", "deduplicated", "not_accepted"):
            parsed = WeftRunReceipt.parse(receipt(decision), idempotency_key=KEY)
            self.assertEqual(
                WeftRunReceipt.parse(json.dumps(parsed.to_dict()), idempotency_key=KEY),
                parsed,
            )
            self.assertIsNone(parsed.rejection)
            self.assertNotIn("rejection", parsed.to_dict())

    def test_rejection_accepts_future_codes_and_utf8_boundary(self) -> None:
        for detail in ("", "界" * 341 + "!"):
            with self.subTest(detail=detail):
                value = json.loads(receipt("not_accepted"))
                for name in ("job_id", "source_pin", "deduplicated"):
                    value.pop(name)
                value["rejection"] = {"code": "future_capacity_reason", "detail": detail}
                parsed = WeftRunReceipt.parse(json.dumps(value), idempotency_key=KEY)
                assert parsed.rejection is not None
                self.assertEqual(parsed.rejection.code, "future_capacity_reason")
                self.assertEqual(parsed.rejection.detail, detail)
                self.assertEqual(
                    WeftRunReceipt.parse(json.dumps(parsed.to_dict()), idempotency_key=KEY),
                    parsed,
                )

    def test_malformed_rejections_raise_value_error(self) -> None:
        invalid = [
            None,
            [],
            {"detail": "offline"},
            {"code": "host_offline"},
            {"code": "", "detail": "offline"},
            {"code": 1, "detail": "offline"},
            {"code": "host_offline", "detail": 1},
            {"code": "host_offline", "detail": "\ud800"},
            {"code": "host_offline", "detail": "界" * 342},
        ]
        for rejection in invalid:
            with self.subTest(rejection=rejection), self.assertRaises(ValueError):
                WeftRunReceipt.parse(
                    receipt("not_accepted", rejection=rejection), idempotency_key=KEY
                )

    def test_malformed_receipts_raise_value_error(self) -> None:
        invalid = [
            receipt(placement_decision=[]),
            receipt(placement_decision={}),
            receipt(idempotency_key="another-call"),
            receipt(accepted_immediately=1),
            receipt(job_id=""),
            receipt(source_pin=""),
            receipt("not_accepted", job_id="wj42"),
            receipt("not_accepted", deduplicated=True),
            receipt("deduplicated", deduplicated=False),
            receipt("queued", accepted_immediately=True),
            receipt(deduplicated="false"),
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                WeftRunReceipt.parse(raw, idempotency_key=KEY)

    def test_generated_invalid_decisions_are_total(self) -> None:
        seeds = (
            [int(os.environ["WEFT_FUZZ_SEED"])] if "WEFT_FUZZ_SEED" in os.environ else range(100)
        )
        shapes: set[str] = set()
        for seed in seeds:
            rng = random.Random(seed)
            for decision in (
                None,
                rng.randint(-100, 100),
                rng.random(),
                bool(seed % 2),
                [rng.randint(0, 10)],
                {"value": rng.randrange(10)},
                f"unknown-{seed}",
            ):
                shapes.add(type(decision).__name__)
                with (
                    self.subTest(seed=seed, decision=decision),
                    self.assertRaises(ValueError),
                ):
                    WeftRunReceipt.parse(receipt(placement_decision=decision), idempotency_key=KEY)
        self.assertEqual(shapes, {"NoneType", "int", "float", "bool", "list", "dict", "str"})


class JobAttributionTests(unittest.TestCase):
    def test_runner_refuses_an_unattributable_worker_before_submission(self) -> None:
        for executable in ("unrelated-worker", source_worker_executable("b" * 64)):
            with self.subTest(executable=executable), self.assertRaises(ValueError):
                WeftCommandRunner(
                    host="studio",
                    agent="omp",
                    model_call_id=KEY,
                    fallback=None,
                    expected_source_sha256=SOURCE,
                    worker_executable=executable,
                )

    def test_retained_command_requires_a_full_source_digest(self) -> None:
        for source in ("", "a" * 63, "a" * 65, "A" * 64, "../worker"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                source_worker_executable(source)

    def test_retained_command_is_attributable_only_to_its_exact_source(self) -> None:
        command = dispatch_command().replace(
            "agent-execution-worker", source_worker_executable(SOURCE), 1
        )
        options = parse_worker_command(command)
        self.assertEqual(options["--expect-source-sha256"], SOURCE)
        self.assertEqual(
            submitted_job_from_listing(
                listing({"job_id": "wj42", "command": command}), model_call_id=KEY
            ),
            "wj42",
        )
        for executable in (
            "agent-execution-worker-",
            "agent-execution-worker-" + "g" * 64,
            source_worker_executable("b" * 64),
            "/other/" + source_worker_executable(SOURCE),
            source_worker_executable(SOURCE) + "-extra",
        ):
            with self.subTest(executable=executable), self.assertRaises(ValueError):
                parse_worker_command(
                    command.replace(source_worker_executable(SOURCE), executable, 1)
                )

    def test_retained_command_still_rejects_shell_composition(self) -> None:
        command = dispatch_command().replace(
            "agent-execution-worker", source_worker_executable(SOURCE), 1
        )
        for suffix in ("; echo other", " && false", "\ntrue"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                parse_worker_command(command + suffix)

    def test_exact_unique_worker_identity_is_required(self) -> None:
        correct: dict[str, object] = {"job_id": "wj42", "command": dispatch_command()}
        self.assertEqual(submitted_job_from_listing(listing(correct), model_call_id=KEY), "wj42")
        self.assertEqual(
            submitted_job_from_listing(listing(correct, correct), model_call_id=KEY),
            "wj42",
        )
        invalid_commands = [
            dispatch_command(KEY + "-retry"),
            dispatch_command(KEY + "_other"),
            dispatch_command(KEY + "0"),
            "echo " + KEY,
            dispatch_command("other") + " --model-call-id " + KEY,
            dispatch_command().replace("--provider omp", "--provider omp --model-call-id other"),
            dispatch_command() + "; echo side-effect",
            "'unterminated",
        ]
        for command in invalid_commands:
            with self.subTest(command=command):
                self.assertIsNone(
                    submitted_job_from_listing(
                        listing(
                            {
                                "job_id": "wj-other",
                                "command": command,
                                "description": f"mentions {KEY}",
                            }
                        ),
                        model_call_id=KEY,
                    )
                )
        self.assertIsNone(
            submitted_job_from_listing(
                listing(
                    correct,
                    {
                        "job_id": "wj43",
                        "command": dispatch_command(),
                    },
                ),
                model_call_id=KEY,
            )
        )
        self.assertIsNone(
            submitted_job_from_listing(listing(correct, version=True), model_call_id=KEY)
        )


class UnreadableReceiptProbeTests(AdmissionTests):
    """An unreadable receipt whose probe found nothing records that it found nothing.

    Consumers must not retry an unobserved outcome — re-dispatching one risks
    running a review that already ran. But this path probes Weft for a job
    attributable to the call and only raises when the probe comes back empty,
    which is a proven absence rather than an unknown. Leaving that out of the
    record made the two indistinguishable, and agent-review then held a
    request `running` for four hours over a cycle that had completed
    (cycle ff82c4337350).
    """

    def test_a_probed_absence_is_recorded_as_such(self) -> None:
        self.submissions = [CommandResult(0, "not json", "")]

        with self.assertRaises(WeftExecutionAmbiguous):
            self.run_dispatch()

        assert self.runner.last_execution is not None
        execution = json.loads(json.dumps(self.runner.last_execution))
        self.assertEqual(execution["processing"]["state"], "unknown")
        self.assertEqual(execution["processing"]["step"], "submission_receipt")
        self.assertIn("no Weft job was attributable", execution["processing"]["detail"])
        self.assertNotIn("job_id", execution)
        # The returned bytes stay, because the message alone says a receipt did
        # not parse and not what arrived.
        self.assertIn("unreadable_receipt", execution["diagnostics"])


class RetrievalBudgetTests(unittest.TestCase):
    """Reading a result is not the same operation as asking after a job.

    Weft's wb139 diagnosis: retrieval spent 10-15s listing R2 prefixes and
    15-25s downloading a 6-13 MB payload, exceeding the 30s probe budget this
    module reused. The TimeoutExpired became
    WeftExecutionDetached("outlived its local waiter"), and a completed review
    was recorded unretrievable because the budget for reading it was sized for
    asking after it.
    """

    def test_a_result_download_gets_more_than_a_status_lookup(self) -> None:
        self.assertGreater(ARTIFACT_RETRIEVAL_SECONDS, LOST_OBSERVATION_PROBE_SECONDS)

    def test_the_retrieval_budget_covers_the_measured_worst_case(self) -> None:
        """15s listing plus 25s download was the observed failure, with headroom."""
        self.assertGreaterEqual(ARTIFACT_RETRIEVAL_SECONDS, 40.0)
