"""Exact-host Weft placement from the paired worker's own fresh status.

The manual inventory gate required a hand-maintained ``agent:<harness>`` tag
beside ``tool:agent-execution`` before any dispatch, even where the paired
worker's own validated execution-status observation admits the exact host and
subject. These regressions drive the real placement boundary: a real status
envelope produced by Agent Execution's own probe of a fake native install, a
real Weft inventory listing, and the runner's placement filter and full
dispatch path. A stale, foreign, mismatched, unbound, or unbacked record must
refuse; only the selected exact worker's current facts -- for the exact
subject that will run -- admit, and the authority is revalidated before every
submission attempt, not once before the loop.

The status producer probes a fake native install; the Weft process boundary
returns retained fixtures. Placement and dispatch use the real typed worker
record and runner.
"""

from __future__ import annotations

import copy
import getpass
import hashlib
import json
import os
import shlex
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from agent_execution.command import CommandResult
from agent_execution.identity import source_sha256
from agent_execution.weft import WeftCommandRunner, WeftPlacementRefused, WeftWorkerStatus
from agent_execution.worker import (
    WORKER_PROTOCOL_VERSION,
    HarnessOutcome,
    WorkerResult,
)
from tests.execution_status_fixtures import admitted_status
from tests.support.native_claude import (
    MODEL,
    FakeNativeInstall,
    packet_command,
    packet_envelope,
    probe_worker_surface,
)

KEY = "placement-regression"
#: The real installed paired build: the status envelope carries it, the
#: dispatch pins it, and the placement gate compares them.
SOURCE = source_sha256()
PROMPT = "Answer only from this packet."
TTL = 600.0


DESTINATION = f"{getpass.getuser()}@studio"


def _iso(moment: float) -> str:
    return datetime.fromtimestamp(moment, timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: object) -> float:
    assert isinstance(value, str)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def aged(envelope: dict, seconds: float) -> dict:
    """Age the receipt and shift the executor's timestamps consistently.

    The producer's own ``validate_status`` must still accept the envelope.
    Cached freshness advances in the requester's monotonic clock, while
    every observed fact and credential basis retain their relative ages.
    """
    result = copy.deepcopy(envelope)
    generated = _parse(result["generated_at"]) - seconds
    result["generated_at"] = _iso(generated)
    result["received_monotonic"] -= seconds
    for row in result["rows"]:
        for fact in row["facts"].values():
            if fact["observed_at"] is None:
                continue
            observed = _parse(fact["observed_at"]) - seconds
            fact["observed_at"] = _iso(observed)
            fact["expires_at"] = _iso(observed + TTL)
            fact["age_seconds"] = max(0.0, generated - observed)
            fact["stale"] = observed + TTL <= generated
        basis = row["credential_basis"]
        if isinstance(basis, dict) and basis.get("observed_at") is not None:
            observed = basis["observed_at"] - seconds
            basis["observed_at"] = observed
            basis["expires_at"] = _iso(observed + TTL)
            basis["age_seconds"] = max(0.0, generated - observed)
            basis["stale"] = observed + TTL <= generated
    return result


def claude_worker_artifact() -> str:
    """A completed claude-packet worker result for the exact dispatch above."""
    return WorkerResult(
        model_call_id=KEY,
        provider="claude-packet",
        status="completed",
        worker_version="0.1.0",
        worker_protocol_version=WORKER_PROTOCOL_VERSION,
        ctx_version="",
        worker_cwd="/remote/project",
        started_at=10,
        completed_at=12,
        model_call_started=True,
        session_id="",
        harness=HarnessOutcome(0, packet_envelope(), ""),
        worker_source_sha256=SOURCE,
        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
    ).to_json()


class WorkerStatusPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.install = FakeNativeInstall(self.root)
        environment = mock.patch.dict(
            os.environ,
            {
                **self.install.environment(),
                "AGENT_PROVIDER_STATUS_DIR": str(self.root / "observations"),
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.envelope = probe_worker_surface()
        self.envelope["received_monotonic"] = time.monotonic()
        self.now = 0.0
        self.calls: list[list[str]] = []
        self.submissions: list[CommandResult] = []
        self.hosts = CommandResult(
            0,
            json.dumps(
                {
                    "kind": "host_list",
                    "version": 1,
                    "hosts": [
                        {
                            "name": "studio",
                            "ssh_target": f"{getpass.getuser()}@studio",
                            "ssh_identity_file": "/Users/test/.ssh/agent_studio_ed25519",
                            "capabilities": ["tool:agent-execution"],
                        }
                    ],
                }
            ),
            "",
        )
        self.artifact = CommandResult(0, claude_worker_artifact(), "")

    def record(self, **changes: object) -> WeftWorkerStatus:
        values: dict[str, object] = {
            "host": "studio",
            "account": getpass.getuser(),
            "envelope": self.envelope,
            "expected_execution_sha256": source_sha256(),
            "destination": DESTINATION,
        }
        values.update(changes)
        return WeftWorkerStatus(**values)  # type: ignore[arg-type]

    def runner(
        self,
        *,
        host: str = "studio",
        record: WeftWorkerStatus | None,
        agent: str = "claude-packet",
    ) -> WeftCommandRunner:
        runner = WeftCommandRunner(
            host=host,
            agent=agent,
            model_call_id=KEY,
            fallback=None,
            invoke=self.invoke,
            clock=lambda: self.now,
            sleep=self.sleep,
            admission_wait=lambda seconds: self.sleep(seconds) or False,
            expected_source_sha256=SOURCE,
        )
        runner.worker_status = record
        return runner

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def invoke(self, command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        self.calls.append(command)
        if command[1] == "host":
            return self.hosts
        if command[1] == "run":
            if self.submissions:
                return self.submissions.pop(0)
            return CommandResult(
                0,
                json.dumps(
                    {
                        "api_version": "weft.run.receipt.v1",
                        "job_id": "wj42",
                        "placement_decision": "accepted_immediately",
                        "selected_host": "studio",
                        "source_pin": "pin-42",
                        "accepted_immediately": True,
                        "deduplicated": False,
                        "idempotency_key": KEY,
                    }
                ),
                "",
            )
        if command[1:3] == ["list", "jobs"] or command[1:3] == ["job", "list"]:
            return CommandResult(0, json.dumps({"kind": "job_list", "version": 1, "jobs": []}), "")
        if command[1] == "status":
            return CommandResult(0, "completed", "")
        if command[1] == "artifact":
            return self.artifact
        if command[1] == "log":
            return CommandResult(0, "", "")
        if command[1:3] == ["job", "mark-processed"]:
            return CommandResult(0, "", "")
        raise AssertionError(command)

    def placement(self, runner: WeftCommandRunner) -> dict[str, object]:
        """Run the real placement check the way `__call__` does."""
        runner.status_selector = "anthropic/" + MODEL
        return runner._placement_check(self.root, time.monotonic() + 30)

    def test_exact_host_admits_on_worker_facts_without_the_manual_tag(self) -> None:
        placement = self.placement(self.runner(record=self.record()))
        self.assertEqual(placement["state"], "eligible", placement.get("detail"))
        self.assertEqual(placement["capability_basis"], "worker-execution-status")

    def test_dispatch_enters_the_weft_runner_without_the_manual_harness_tag(self) -> None:
        runner = self.runner(record=self.record())
        result = runner(packet_command(PROMPT), self.root, 120)
        self.assertEqual(runner.status_selector, "anthropic/" + MODEL)
        self.assertEqual(runner.claude_contract["claude_model"], MODEL)
        self.assertEqual(result.exit_status, 0, result.stderr)
        self.assertIn("native answer", result.stdout)
        submit = next(command for command in self.calls if command[1] == "run")
        self.assertIn("studio", submit)
        self.assertIn("--host", submit)
        self.assertIn("--require-capability", submit)
        capability_index = submit.index("--require-capability")
        self.assertEqual(submit[capability_index + 1], "tool:agent-execution")
        self.assertNotIn("--agent", submit)
        self.assertNotIn("--require-capability=agent:claude", submit)
        self.assertNotIn("agent:claude", submit)
        remote = shlex.split(submit[-1])
        self.assertEqual(remote[:2], ["agent-execution-worker", "execute"])
        self.assertEqual(remote[remote.index("--provider") + 1], "claude-packet")
        self.assertEqual(remote[remote.index("--harness-model") + 1], MODEL)
        self.assertEqual(
            remote[remote.index("--expect-protocol") + 1], str(WORKER_PROTOCOL_VERSION)
        )
        self.assertEqual(remote[remote.index("--expect-source-sha256") + 1], SOURCE)
        self.assertIn("--", remote)
        harness = remote[remote.index("--") + 1 :]
        self.assertEqual(harness[0], "claude")
        self.assertNotIn("--model", harness)
        self.assertNotIn(MODEL, harness)
        self.assertIn("--safe-mode", harness)
        self.assertIn("--tools", harness)
        self.assertEqual(harness[harness.index("--tools") + 1], "")

    def test_valid_status_with_manual_tag_also_omits_weft_agent_filter(self) -> None:
        self.hosts = CommandResult(
            0,
            json.dumps(
                {
                    "kind": "host_list",
                    "version": 1,
                    "hosts": [
                        {
                            "name": "studio",
                            "ssh_target": f"{getpass.getuser()}@studio",
                            "ssh_identity_file": "/Users/test/.ssh/agent_studio_ed25519",
                            "capabilities": ["agent:claude", "tool:agent-execution"],
                        }
                    ],
                }
            ),
            "",
        )
        result = self.runner(record=self.record())(packet_command(PROMPT), self.root, 120)
        self.assertEqual(result.exit_status, 0, result.stderr)
        submit = next(command for command in self.calls if command[1] == "run")
        self.assertIn("--host", submit)
        self.assertNotIn("--agent", submit)
        self.assertIn("--require-capability", submit)
        self.assertIn("tool:agent-execution", submit)

    def _inventory(self, hosts: list[dict[str, object]]) -> None:
        self.hosts = CommandResult(
            0,
            json.dumps({"kind": "host_list", "version": 1, "hosts": hosts}),
            "",
        )

    def test_named_studio_exact_endpoint_binding_admits(self) -> None:
        self._inventory(
            [
                {
                    "name": "studio",
                    "ssh_target": DESTINATION,
                    "ssh_identity_file": "/Users/osteele/.ssh/agent_studio_ed25519",
                    "capabilities": ["tool:agent-execution"],
                }
            ]
        )
        placement = self.placement(self.runner(record=self.record()))
        self.assertEqual(placement["state"], "eligible", placement.get("detail"))
        self.assertEqual(placement["capability_basis"], "worker-execution-status")

    def test_exact_endpoint_does_not_require_an_explicit_identity_file(self) -> None:
        self._inventory(
            [
                {
                    "name": "studio",
                    "ssh_target": DESTINATION,
                    "capabilities": ["tool:agent-execution"],
                }
            ]
        )
        placement = self.placement(self.runner(record=self.record()))
        self.assertEqual(placement["state"], "eligible", placement.get("detail"))
        self.assertEqual(placement["capability_basis"], "worker-execution-status")

    def test_missing_target_refuses_only_worker_status_authority(self) -> None:
        self._inventory(
            [
                {
                    "name": "studio",
                    "ssh_identity_file": "/Users/test/.ssh/agent_studio_ed25519",
                    "capabilities": ["tool:agent-execution"],
                }
            ]
        )
        placement = self.placement(self.runner(record=self.record()))
        self.assertEqual(placement["state"], "refused")
        self.assertIn("no concrete ssh_target", str(placement["detail"]))

    def test_duplicate_target_binding_refuses_worker_status_authority(self) -> None:
        entry: dict[str, object] = {
            "name": "studio",
            "ssh_target": DESTINATION,
            "ssh_identity_file": "/Users/test/.ssh/agent_studio_ed25519",
            "capabilities": ["tool:agent-execution"],
        }
        self._inventory([entry, dict(entry)])
        placement = self.placement(self.runner(record=self.record()))
        self.assertEqual(placement["state"], "refused")
        self.assertIn("exactly one", str(placement["detail"]))

    def test_mismatched_target_account_or_host_refuses_worker_status_authority(self) -> None:
        self._inventory(
            [
                {
                    "name": "studio",
                    "ssh_target": "other-account@studio",
                    "ssh_identity_file": "/Users/test/.ssh/agent_studio_ed25519",
                    "capabilities": ["tool:agent-execution"],
                }
            ]
        )
        placement = self.placement(self.runner(record=self.record()))
        self.assertEqual(placement["state"], "refused")
        self.assertIn("does not match Review's", str(placement["detail"]))

    def test_malformed_endpoint_keeps_static_tag_route_intact(self) -> None:
        self._inventory(
            [
                {
                    "name": "studio",
                    "ssh_target": ["not", "a", "target"],
                    "ssh_identity_file": "/Users/test/.ssh/agent_studio_ed25519",
                    "capabilities": ["agent:claude", "tool:agent-execution"],
                }
            ]
        )
        placement = self.placement(self.runner(record=self.record()))
        self.assertEqual(placement["state"], "eligible", placement.get("detail"))
        self.assertNotIn("capability_basis", placement)

    def test_stale_worker_facts_refuse(self) -> None:
        record = self.record(envelope=aged(self.envelope, TTL + 100))
        runner = self.runner(record=record)
        with self.assertRaises(WeftPlacementRefused):
            runner(packet_command(PROMPT), self.root, 120)
        self.assertFalse(any(command[1] == "run" for command in self.calls))
        self.assertIn("does not currently admit capability", str(runner.last_execution))

    def test_cached_status_freshness_ignores_both_wall_clocks(self) -> None:
        received = self.envelope["received_monotonic"]
        for executor_shift in (-100_000.0, 100_000.0):
            for requester_now in (1_000.0, 100_000_000_000.0):
                with self.subTest(executor_shift=executor_shift, requester_now=requester_now):
                    envelope = aged(self.envelope, executor_shift)
                    envelope["received_monotonic"] = received
                    runner = self.runner(record=self.record(envelope=envelope))
                    with (
                        mock.patch("agent_execution.weft.time.time", return_value=requester_now),
                        mock.patch(
                            "agent_execution.weft.time.monotonic", return_value=received + TTL - 1
                        ),
                    ):
                        result = runner(packet_command(PROMPT), self.root, 120)
                    self.assertEqual(result.exit_status, 0)

    def test_cached_status_expires_at_monotonic_ttl(self) -> None:
        received = self.envelope["received_monotonic"]
        runner = self.runner(record=self.record())
        with (
            mock.patch("agent_execution.weft.time.monotonic", return_value=received + TTL),
            self.assertRaises(WeftPlacementRefused),
        ):
            runner(packet_command(PROMPT), self.root, 120)
        self.assertFalse(any(command[1] == "run" for command in self.calls))

    def test_missing_or_foreign_monotonic_receipt_cannot_admit(self) -> None:
        for receipt in (None, float("nan"), float("inf"), -float("inf"), True, "100", 10**12):
            with self.subTest(receipt=receipt):
                envelope = copy.deepcopy(self.envelope)
                if receipt is None:
                    envelope.pop("received_monotonic")
                else:
                    envelope["received_monotonic"] = receipt
                runner = self.runner(record=self.record(envelope=envelope))
                with self.assertRaises(WeftPlacementRefused):
                    runner(packet_command(PROMPT), self.root, 120)
        self.assertFalse(any(command[1] == "run" for command in self.calls))

    def test_workspace_write_dispatch_refuses_a_read_only_envelope(self) -> None:
        """A fresh read-only worker envelope cannot authorize a write dispatch.

        The subject the placement gate matches is the one the validated
        invocation will actually run -- an `offload-task` OMP dispatch under
        `workspace-write-no-shell`, derived the way the worker's own preflight
        derives it -- never a per-provider default. The reviewed patch matched
        a hardcoded `worker`/`read-only-no-shell` subject instead, so this
        fresh read-only envelope admitted a workspace-write dispatch whose own
        subject no row described.
        """
        selector = "anthropic/claude-opus-5-5"
        read_only = admitted_status(
            harness="omp", surface="worker", selectors=[selector], transport="weft"
        )
        self.assertEqual(read_only["rows"][0]["subject"]["surface"], "worker")
        self.assertEqual(read_only["rows"][0]["subject"]["tool_policy"], "read-only-no-shell")
        runner = self.runner(record=self.record(envelope=read_only), agent="omp")
        runner.status_selector = selector
        runner.omp_policy = "workspace-write-no-shell"
        placement = runner._placement_check(self.root, time.monotonic() + 30)
        self.assertEqual(placement["state"], "refused", placement.get("detail"))
        self.assertIn("no single row for this dispatch subject", str(placement["detail"]))
        self.assertIn(
            "offload-task/workspace-write-no-shell",
            str(placement["detail"]),
        )

    def test_offload_task_envelope_admits_the_matching_write_dispatch(self) -> None:
        """Only fresh status for the matching write subject admits a write dispatch."""
        selector = "anthropic/claude-opus-5-5"
        writer = admitted_status(
            harness="omp", surface="offload-task", selectors=[selector], transport="weft"
        )
        writer["received_monotonic"] = time.monotonic()
        runner = self.runner(record=self.record(envelope=writer), agent="omp")
        runner.status_selector = selector
        runner.omp_policy = "workspace-write-no-shell"
        placement = runner._placement_check(self.root, time.monotonic() + 30)
        self.assertEqual(placement["state"], "eligible", placement.get("detail"))
        self.assertEqual(placement["capability_basis"], "worker-execution-status")

    def test_read_only_dispatch_still_admits_its_own_read_only_row(self) -> None:
        selector = "anthropic/claude-opus-5-5"
        read_only = admitted_status(
            harness="omp", surface="worker", selectors=[selector], transport="weft"
        )
        read_only["received_monotonic"] = time.monotonic()
        runner = self.runner(record=self.record(envelope=read_only), agent="omp")
        runner.status_selector = selector
        runner.omp_policy = "read-only-no-shell"
        placement = runner._placement_check(self.root, time.monotonic() + 30)
        self.assertEqual(placement["state"], "eligible", placement.get("detail"))
        self.assertEqual(placement["capability_basis"], "worker-execution-status")

    def test_stale_status_after_proven_not_accepted_refuses_the_retry(self) -> None:
        """The authority is revalidated before every attempt, backoff included.

        The first attempt runs on the fresh record, without the manual tag.
        The proven not_accepted receipt then backs off -- and the record's
        facts expire before the retry. The retry must refuse on the
        revalidated status; it may not reuse the `worker-execution-status`
        basis computed before the loop. The reviewed patch computed the
        authority once and resubmitted on the stale basis; the pre-Lane-A gate
        refused before any submission existed at all.
        """
        self.submissions = [
            CommandResult(
                1,
                json.dumps(
                    {
                        "api_version": "weft.run.receipt.v1",
                        "job_id": "",
                        "placement_decision": "not_accepted",
                        "selected_host": "",
                        "source_pin": "",
                        "accepted_immediately": False,
                        "deduplicated": False,
                        "idempotency_key": KEY,
                        "rejection": {
                            "code": "admission_race_lost",
                            "detail": "another caller won admission",
                        },
                    }
                ),
                "host constraint mismatch",
            )
        ]
        runner = self.runner(record=self.record())
        retried = {"done": False}

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            result = self.invoke(command, cwd, timeout)
            if command[1] == "run" and not retried["done"]:
                # The facts the first attempt was admitted on expire during
                # the backoff window.
                retried["done"] = True
                runner.worker_status = self.record(envelope=aged(self.envelope, TTL + 100))
            return result

        runner.invoke = invoke
        with self.assertRaises(WeftPlacementRefused) as refused:
            runner(packet_command(PROMPT), self.root, 120)
        runs = [command for command in self.calls if command[1] == "run"]
        self.assertEqual(len(runs), 1)
        # The one real attempt used the authority route: no manual filter.
        self.assertNotIn("--agent", runs[0])
        self.assertIn("does not currently admit capability", str(refused.exception))

    def test_static_tags_regained_after_backoff_keep_the_agent_filter(self) -> None:
        """A retry whose fresh status no longer passes keeps the static filter.

        During the backoff the manual tag returns and the record's facts
        expire. The second submission must run under Weft's `--agent`
        shorthand from the static-tag route -- never under the stale
        worker-status basis, and never unfiltered.
        """
        self.submissions = [
            CommandResult(
                1,
                json.dumps(
                    {
                        "api_version": "weft.run.receipt.v1",
                        "job_id": "",
                        "placement_decision": "not_accepted",
                        "selected_host": "",
                        "source_pin": "",
                        "accepted_immediately": False,
                        "deduplicated": False,
                        "idempotency_key": KEY,
                        "rejection": {
                            "code": "admission_race_lost",
                            "detail": "another caller won admission",
                        },
                    }
                ),
                "host constraint mismatch",
            ),
            CommandResult(
                0,
                json.dumps(
                    {
                        "api_version": "weft.run.receipt.v1",
                        "job_id": "wj43",
                        "placement_decision": "accepted_immediately",
                        "selected_host": "studio",
                        "source_pin": "pin-43",
                        "accepted_immediately": True,
                        "deduplicated": False,
                        "idempotency_key": KEY,
                    }
                ),
                "",
            ),
        ]
        runner = self.runner(record=self.record())
        retried = {"done": False}

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            result = self.invoke(command, cwd, timeout)
            if command[1] == "run" and not retried["done"]:
                retried["done"] = True
                runner.worker_status = self.record(envelope=aged(self.envelope, TTL + 100))
                self.hosts = CommandResult(
                    0,
                    json.dumps(
                        {
                            "kind": "host_list",
                            "version": 1,
                            "hosts": [
                                {
                                    "name": "studio",
                                    "capabilities": ["agent:claude", "tool:agent-execution"],
                                }
                            ],
                        }
                    ),
                    "",
                )
            return result

        runner.invoke = invoke
        result = runner(packet_command(PROMPT), self.root, 120)
        self.assertEqual(result.exit_status, 0, result.stderr)
        runs = [command for command in self.calls if command[1] == "run"]
        self.assertEqual(len(runs), 2)
        self.assertNotIn("--agent", runs[0])
        agent_index = runs[1].index("--agent")
        self.assertEqual(runs[1][agent_index + 1], "claude")

    def test_unbound_record_never_relaxes_the_agent_filter(self) -> None:
        """A record with no observation origin is authority for nothing.

        String equality of the caller's alias alone is not proof: the record
        must carry the registered SSH destination its envelope was observed
        through. Without it the authority exception fails closed -- the
        ordinary static-tag behavior stands (refusal here, because the manual
        tag is missing) and the operational binding prerequisite is named.
        """
        runner = self.runner(record=self.record(destination=None))
        with self.assertRaises(WeftPlacementRefused):
            runner(packet_command(PROMPT), self.root, 120)
        self.assertFalse(any(command[1] == "run" for command in self.calls))

        # A destination whose SSH account disagrees with the record's account
        # is equally unbound.
        self.calls = []
        mismatched = self.record(destination="someone-else@studio.hub")
        with self.assertRaises(WeftPlacementRefused) as refused:
            self.runner(record=mismatched)(packet_command(PROMPT), self.root, 120)
        self.assertIn("does not match the registered destination account", str(refused.exception))
        self.assertFalse(any(command[1] == "run" for command in self.calls))

        # With the manual tag present, the static route admits normally and
        # the unbound record relaxes nothing.
        self.calls = []
        self.hosts = CommandResult(
            0,
            json.dumps(
                {
                    "kind": "host_list",
                    "version": 1,
                    "hosts": [
                        {"name": "studio", "capabilities": ["agent:claude", "tool:agent-execution"]}
                    ],
                }
            ),
            "",
        )
        result = runner(packet_command(PROMPT), self.root, 120)
        self.assertEqual(result.exit_status, 0, result.stderr)
        submit = next(command for command in self.calls if command[1] == "run")
        self.assertIn("--agent", submit)
        self.assertEqual(submit[submit.index("--agent") + 1], "claude")

    def test_wrong_host_account_source_or_subject_refuse_before_any_submission(self) -> None:
        cases = {
            "host": self.record(host="other"),
            "account": self.record(account="not-the-os-user"),
            "source": self.record(expected_execution_sha256="b" * 64),
        }
        forged = copy.deepcopy(self.envelope)
        forged["rows"][0]["subject"]["execution_sha256"] = "c" * 64
        cases["forged envelope"] = self.record(envelope=forged)
        selector_mismatch = copy.deepcopy(self.envelope)
        selector_mismatch["rows"][0]["subject"]["selector"] = "anthropic/claude-sonnet-4-5"
        cases["subject"] = self.record(envelope=selector_mismatch)
        for label, record in cases.items():
            with self.subTest(case=label):
                runner = self.runner(record=record)
                with self.assertRaises(WeftPlacementRefused):
                    runner(packet_command(PROMPT), self.root, 120)
                self.assertFalse(any(command[1] == "run" for command in self.calls))
                self.calls = []

    def test_missing_tool_declaration_and_auto_host_refuse(self) -> None:
        self.hosts = CommandResult(
            0,
            json.dumps(
                {
                    "kind": "host_list",
                    "version": 1,
                    "hosts": [{"name": "studio", "capabilities": ["agent:claude"]}],
                }
            ),
            "",
        )
        placement = self.placement(self.runner(record=self.record()))
        self.assertEqual(placement["state"], "refused")
        self.assertIn("tool:agent-execution", str(placement["detail"]))
        auto_runner = self.runner(host="auto", record=self.record(host="auto"))
        with self.assertRaises(WeftPlacementRefused):
            auto_runner(packet_command(PROMPT), self.root, 120)
        self.assertFalse(any(command[1] == "run" for command in self.calls))
        self.assertNotIn("worker status", str(auto_runner.last_execution))

    def test_fresh_quota_refusal_refuses(self) -> None:
        quota = copy.deepcopy(self.envelope)
        now = time.time()
        quota["rows"][0]["facts"]["quota"] = {
            "state": "unavailable",
            "observed_at": _iso(now - 10),
            "expires_at": _iso(now + TTL),
            "age_seconds": 10.0,
            "stale": False,
            "source": {"tool": "agent-execution", "method": "provider-refusal"},
            "detail": "Provider reported quota refusal",
            "condition": "quota",
        }
        placement = self.placement(self.runner(record=self.record(envelope=quota)))
        self.assertEqual(placement["state"], "refused")
        self.assertIn("quota unavailable", str(placement["detail"]))

    def test_no_status_keeps_static_agent_requirement_and_missing_tag_refuses(self) -> None:
        tagged = {
            "kind": "host_list",
            "version": 1,
            "hosts": [{"name": "studio", "capabilities": ["agent:claude", "tool:agent-execution"]}],
        }
        self.hosts = CommandResult(0, json.dumps(tagged), "")
        result = self.runner(record=None)(packet_command(PROMPT), self.root, 120)
        self.assertEqual(result.exit_status, 0, result.stderr)
        submit = next(command for command in self.calls if command[1] == "run")
        self.assertIn("--agent", submit)
        agent_index = submit.index("--agent")
        self.assertEqual(submit[agent_index + 1], "claude")
        self.assertIn("--require-capability", submit)
        self.assertIn("tool:agent-execution", submit)
        self.assertIn("--host", submit)
        self.assertIn("studio", submit)

        self.calls = []
        self.hosts = CommandResult(
            0,
            json.dumps(
                {
                    "kind": "host_list",
                    "version": 1,
                    "hosts": [{"name": "studio", "capabilities": ["tool:agent-execution"]}],
                }
            ),
            "",
        )
        runner = self.runner(record=None)
        with self.assertRaises(WeftPlacementRefused):
            runner(packet_command(PROMPT), self.root, 120)
        self.assertFalse(any(command[1] == "run" for command in self.calls))

    def test_tags_alone_still_admit_without_a_record(self) -> None:
        self.hosts = CommandResult(
            0,
            json.dumps(
                {
                    "kind": "host_list",
                    "version": 1,
                    "hosts": [
                        {
                            "name": "studio",
                            "capabilities": ["agent:claude", "tool:agent-execution"],
                        }
                    ],
                }
            ),
            "",
        )
        placement = self.placement(self.runner(record=None))
        self.assertEqual(placement["state"], "eligible")
        self.assertNotIn("capability_basis", placement)


if __name__ == "__main__":
    unittest.main()
