"""Weft job attributes: emitted submission arguments, construction freeze, validation.

Tests run the real WeftCommandRunner dispatch path against a synthetic external
`weft` CLI, so every assertion is about the protocol words the transport
actually receives, never about internal helpers.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from agent_execution.command import CommandResult
from agent_execution.omp_execution import omp_transcript
from agent_execution.weft import WeftCommandRunner
from agent_execution.worker import WORKER_PROTOCOL_VERSION, HarnessOutcome, WorkerResult
from tests.support.omp import omp_command, omp_output

KEY = "model-call-attr"
SOURCE = "b" * 64


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


def worker_artifact(prompt: str = "review") -> str:
    stdout = omp_output(
        cwd="/remote/project",
        prompt=prompt,
        final="remote answer",
        policy="read-only-no-shell",
    )
    return WorkerResult(
        model_call_id=KEY,
        provider="omp",
        status="completed",
        worker_version="0.1.0",
        worker_protocol_version=WORKER_PROTOCOL_VERSION,
        worker_source_sha256=SOURCE,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        ctx_version="",
        worker_cwd="/remote/project",
        started_at=10,
        completed_at=12,
        model_call_started=True,
        session_id="session-1",
        harness=HarnessOutcome(0, stdout, ""),
        omp_evidence=omp_transcript(stdout),
    ).to_json()


def attribute_pairs(argv: list[str]) -> list[tuple[str, str]]:
    """Every `--attr key=value` pair on one submission, in emitted order."""
    pairs: list[tuple[str, str]] = []
    index = 0
    while True:
        try:
            index = argv.index("--attr", index)
        except ValueError:
            return pairs
        key, value = argv[index + 1].split("=", 1)
        pairs.append((key, value))
        index += 2


def without_payload_path(argv: list[str]) -> list[str]:
    """The submission argv with its per-attempt payload file name normalized."""
    normalized = list(argv)
    normalized[argv.index("--payload") + 1] = "<payload>"
    return normalized


class Harness:
    """Synthetic external `weft` CLI recording every invocation at the boundary."""

    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.now = 0.0
        self.calls: list[tuple[list[str], Path, float | None]] = []
        self.submissions: list[CommandResult] = []
        self.status = CommandResult(0, "completed", "")
        self.artifact = CommandResult(0, worker_artifact(), "")
        self.local_calls: list[list[str]] = []
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

    def close(self) -> None:
        self.temporary.cleanup()

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def wait(self, seconds: float) -> bool:
        self.sleep(seconds)
        return False

    def fallback(self, command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        self.local_calls.append(command)
        return CommandResult(0, "local answer", "")

    def invoke(self, command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        self.calls.append((command, cwd, timeout))
        if command[1] == "host":
            return self.hosts
        if command[1] == "run":
            return self.submissions.pop(0)
        if command[1] == "status":
            return self.status
        if command[1:3] == ["artifact", "cat"]:
            return self.artifact
        if command[1] == "log":
            return CommandResult(0, "", "")
        if command[1:3] == ["job", "mark-processed"]:
            return CommandResult(0, "", "")
        raise AssertionError(command)

    def submissions_sent(self) -> list[list[str]]:
        return [argv for argv, _, _ in self.calls if argv[1] == "run"]


class WeftAttributeDispatchTests(unittest.TestCase):
    """Attributes ride real admissions through the synthetic external CLI."""

    def setUp(self) -> None:
        self.harness = Harness()
        self.addCleanup(self.harness.close)

    def runner(self, attributes: Mapping[str, str] | None = None) -> WeftCommandRunner:
        return WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=self.harness.fallback,
            invoke=self.harness.invoke,
            clock=lambda: self.harness.now,
            sleep=self.harness.sleep,
            admission_wait=self.harness.wait,
            expected_source_sha256=SOURCE,
            attributes=attributes,
        )

    def dispatch(self, runner: WeftCommandRunner) -> CommandResult:
        return runner(omp_command(str(self.harness.root)), self.harness.root, 30.0)

    def assert_remote(self, result: CommandResult) -> None:
        self.assertEqual(result.exit_status, 0, result.stderr)
        self.assertIn("remote answer", result.stdout)
        self.assertEqual(self.harness.local_calls, [])
        self.assertIsNotNone(result.execution)

    def test_no_attributes_leaves_submission_identical_to_the_baseline(self) -> None:
        baselines: list[list[str]] = []
        for attributes in (None, {}):
            self.harness.submissions = [CommandResult(0, receipt(), "")]
            self.assert_remote(self.dispatch(self.runner(attributes)))
            submitted = self.harness.submissions_sent()[0]
            self.assertNotIn("--attr", submitted)
            baselines.append(without_payload_path(submitted))
            self.harness.calls.clear()
        self.assertEqual(baselines[0], baselines[1])

    def test_attributes_are_emitted_as_attr_pairs_in_ascending_key_order(self) -> None:
        self.harness.submissions = [CommandResult(0, receipt(), "")]
        self.assert_remote(
            self.dispatch(self.runner({"zone": "eu-1", "alpha-team": "review", "b.c": "d"}))
        )
        submitted = self.harness.submissions_sent()[0]
        self.assertEqual(
            attribute_pairs(submitted),
            [("alpha-team", "review"), ("b.c", "d"), ("zone", "eu-1")],
        )
        # One flag per pair, each flag immediately followed by its key=value word.
        self.assertEqual(submitted.count("--attr"), 3)
        # The pairs are submission options before the final remote command; the
        # worker command Weft executes carries none of them.
        self.assertLess(submitted.index("--attr"), len(submitted) - 1)
        remote = shlex.split(submitted[-1])
        self.assertEqual(remote[:2], ["agent-execution-worker", "execute"])
        self.assertNotIn("--attr", remote)

    def test_maximum_legal_mapping_is_emitted_verbatim(self) -> None:
        long_key = "a" + "b" * 63  # exactly the 64-character key limit
        wide_value = "é" * 128  # exactly 256 UTF-8 bytes
        entries = [(f"key-{index:02d}", f"value-{index}") for index in range(31)]
        entries.append((long_key, wide_value))
        attributes = dict(entries)
        runner = self.runner(attributes)
        self.harness.submissions = [CommandResult(0, receipt(), "")]
        self.assert_remote(self.dispatch(runner))
        submitted = self.harness.submissions_sent()[0]
        self.assertEqual(attribute_pairs(submitted), sorted(entries))
        self.assertEqual(submitted.count("--attr"), 32)

    def test_legal_values_are_sent_verbatim_without_normalization(self) -> None:
        # Space, '=', path punctuation, non-ASCII text, and a non-control
        # Unicode format character (U+200B) are all legal and must survive
        # byte-for-byte; only control characters are forbidden.
        value = "review kind=spec path=/tmp/x café " + "\u200b"
        runner = self.runner({"note": value, "build.id-2_x": "ok"})
        self.harness.submissions = [CommandResult(0, receipt(), "")]
        self.assert_remote(self.dispatch(runner))
        submitted = self.harness.submissions_sent()[0]
        self.assertEqual(
            attribute_pairs(submitted),
            [("build.id-2_x", "ok"), ("note", value)],
        )

    def test_identical_attributes_on_admission_resend(self) -> None:
        attributes = {"review-kind": "spec", "attempt-bucket": "b2"}
        runner = self.runner(attributes)
        self.harness.submissions = [
            CommandResult(
                1,
                receipt(
                    "not_accepted",
                    rejection={"code": "host_offline", "detail": "studio is offline"},
                ),
                "host constraint mismatch",
            ),
            CommandResult(0, receipt("deduplicated"), ""),
        ]
        self.assert_remote(self.dispatch(runner))
        submissions = self.harness.submissions_sent()
        self.assertEqual(len(submissions), 2)
        expected = [("attempt-bucket", "b2"), ("review-kind", "spec")]
        self.assertEqual(attribute_pairs(submissions[0]), expected)
        self.assertEqual(attribute_pairs(submissions[1]), expected)
        # The same idempotency key, and identical submission words apart from
        # the per-attempt payload file name.
        self.assertEqual(submissions[0][submissions[0].index("--idempotency-key") + 1], KEY)
        self.assertEqual(submissions[1][submissions[1].index("--idempotency-key") + 1], KEY)
        self.assertEqual(without_payload_path(submissions[0]), without_payload_path(submissions[1]))

    def test_mapping_mutation_after_construction_cannot_change_what_is_emitted(self) -> None:
        attributes: dict[str, str] = {"team": "review"}
        runner = self.runner(attributes)
        attributes["team"] = "mutated-after-construction"
        attributes["extra"] = "late"
        self.harness.submissions = [CommandResult(0, receipt(), "")]
        self.assert_remote(self.dispatch(runner))
        self.assertEqual(attribute_pairs(self.harness.submissions_sent()[0]), [("team", "review")])

    def test_mapping_mutation_during_backoff_cannot_change_the_resend(self) -> None:
        attributes: dict[str, str] = {"team": "review"}

        def mutate_during_backoff(seconds: float) -> bool:
            self.harness.now += seconds
            attributes["team"] = "mutated-mid-flight"
            attributes["extra"] = "late"
            return False

        runner = WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=self.harness.fallback,
            invoke=self.harness.invoke,
            clock=lambda: self.harness.now,
            sleep=self.harness.sleep,
            admission_wait=mutate_during_backoff,
            expected_source_sha256=SOURCE,
            attributes=attributes,
        )
        self.harness.submissions = [
            CommandResult(
                1,
                receipt("not_accepted", rejection={"code": "drained", "detail": "retry later"}),
                "",
            ),
            CommandResult(0, receipt(), ""),
        ]
        self.assert_remote(self.dispatch(runner))
        submissions = self.harness.submissions_sent()
        self.assertEqual(len(submissions), 2)
        self.assertEqual(attribute_pairs(submissions[0]), [("team", "review")])
        self.assertEqual(attribute_pairs(submissions[1]), [("team", "review")])
        self.assertEqual(without_payload_path(submissions[0]), without_payload_path(submissions[1]))


class WeftAttributeValidationTests(unittest.TestCase):
    """Invalid mappings are refused at construction, before any invocation."""

    def setUp(self) -> None:
        self.harness = Harness()
        self.addCleanup(self.harness.close)

    def construct(self, attributes: object) -> WeftCommandRunner:
        return WeftCommandRunner(
            host="studio",
            agent="omp",
            model_call_id=KEY,
            fallback=None,
            invoke=self.harness.invoke,
            clock=lambda: self.harness.now,
            sleep=self.harness.sleep,
            expected_source_sha256=SOURCE,
            attributes=cast("Mapping[str, str] | None", attributes),
        )

    def assert_rejected(self, attributes: object) -> None:
        with self.assertRaises(ValueError):
            self.construct(attributes)
        # Refusal happens before any transport word is spoken.
        self.assertEqual(self.harness.calls, [])

    def test_more_than_32_entries_are_refused(self) -> None:
        illegal = {f"key-{index:02d}": str(index) for index in range(33)}
        self.assert_rejected(illegal)

    def test_keys_longer_than_64_characters_are_refused(self) -> None:
        self.assert_rejected({"a" + "b" * 64: "x"})

    def test_key_grammar_boundaries(self) -> None:
        for key in (
            "Team",  # uppercase
            "1team",  # must start with a letter
            "-team",
            "_team",
            ".team",
            "team!",
            "team x",
            "",  # empty
            "team\n",  # trailing newline: `$`-anchored matching would admit it
            "team\nx",
        ):
            with self.subTest(key=key):
                self.assert_rejected({key: "x"})
        self.assert_rejected({1: "x"})  # non-string key, still whole-mapping checked

    def test_value_byte_length_boundaries(self) -> None:
        illegal = [
            "",  # zero bytes
            "a" * 257,
            "é" * 129,  # 258 bytes
            "界" * 86,  # 258 bytes
        ]
        for value in illegal:
            with self.subTest(bytes=len(value.encode("utf-8"))):
                self.assert_rejected({"size": value})
        self.assertEqual(self.harness.calls, [])

    def test_control_characters_are_refused(self) -> None:
        for control in ("\x00", "\x1b", "\x7f", "\x85", "\x9f", "\t", "\n", "\r"):
            with self.subTest(control=repr(control)):
                self.assert_rejected({"note": f"a{control}b"})

    def test_lone_surrogate_value_is_refused_as_invalid_unicode(self) -> None:
        self.assert_rejected({"note": "review \ud800"})

    def test_invalid_mapping_types_are_refused(self) -> None:
        self.assert_rejected(["--attr", "a=b"])
        self.assert_rejected({"note": 1})

    def test_validation_covers_the_complete_mapping(self) -> None:
        # The invalid entry sits after a valid one in insertion order, so a
        # check of any emitted prefix alone would miss it.
        self.assert_rejected({"alpha": "ok", "BAD": "x"})
