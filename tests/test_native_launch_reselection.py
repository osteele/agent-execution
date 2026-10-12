"""One worker launch must run the executable its own preflight observed.

`native_claude_launch` resolves PATH. Execution's status probes resolve it
during the worker's fresh preflight, and the launcher resolved it again before
the launch -- so a binary swapped, upgraded, or re-pointed between observation
and launch was executed even though the facts the worker just admitted
described a different file. These regressions drive the true production
boundary -- a real ``execute_worker`` preflight (real status probe, real
wrapper protocol resolution, real environment scrubbing) with the launch
captured at the process boundary.

The re-pointing in the first test happens where production drift happens:
between the preflight's observation (help probe, binding) and the worker
consumes that binding at launch. On the unmodified pre-Lane-A baseline the
launch re-resolves PATH there, selects the binary the wrapper was re-pointed
to, and runs with an unprefixed PATH; after the fix the launch reuses the
observed executable and environment. The other two tests pin the refusal
sides: a launch whose one-shot binding is missing or expired at consumption
refuses before any harness start, and a binary that mutates between the help
probe and the authentication probe is never admitted as observed.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from agent_execution.credentials import BASIS_SUBSCRIPTION, CredentialBasis
from agent_execution.worker import execute_worker
from tests.support.native_claude import (
    MACHO_MAGIC,
    FakeNativeInstall,
    packet_command,
    packet_envelope,
)

SECRET = "sk-test-anthropic-secret"


class NativeLaunchReselectionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.install = FakeNativeInstall(self.root)
        environment = mock.patch.dict(
            os.environ,
            {
                **self.install.environment(),
                "ANTHROPIC_API_KEY": SECRET,
                "AGENT_PROVIDER_STATUS_DIR": str(self.root / "observations"),
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def subscription(self) -> mock.Mock:
        basis = CredentialBasis("claude", BASIS_SUBSCRIPTION, "none", time.time(), "fixture")
        return mock.Mock(state="available", answered=True, basis=basis)

    def test_worker_launch_reuses_the_executable_its_preflight_observed(self) -> None:
        """The preflight observed binary A; the launch must run A, not B.

        The wrapper resolves physical binary A throughout the preflight. The
        authentication probe -- the last observation of the preflight, after
        the help probe recorded the launch binding -- re-points the wrapper at
        binary B, exactly where a production upgrade lands between the
        worker's observation and its launch. Any second PATH resolution at
        launch time selects B; only the binding keeps the launch on A.
        """
        launched: list[tuple[list[str], dict[str, str]]] = []

        def launch(argv, cwd, prompt, timeout, *, environment=None, **kwargs):
            launched.append((list(argv), dict(environment or {})))
            return subprocess.CompletedProcess(argv, 0, packet_envelope(), "")

        def authenticate(**kwargs):
            self.install.point_to(self.install.native_b)
            return self.subscription()

        with (
            mock.patch(
                "agent_execution.execution_status.run_in_process_group",
                return_value=subprocess.CompletedProcess(
                    ["claude", "--help"], 0, "--safe-mode --tools --restricted", ""
                ),
            ),
            mock.patch("agent_execution.credentials.observe_claude_auth", side_effect=authenticate),
            mock.patch("agent_execution.claude_execution.run_in_process_group", side_effect=launch),
        ):
            result = execute_worker(
                provider="claude-packet",
                model_call_id="reselection-regression",
                command=packet_command(),
                timeout=30,
                ctx_timeout=10,
                cwd=self.root,
                execution_transport="weft",
            )
        self.assertEqual(result.status, "completed", result.failure)
        self.assertTrue(result.model_call_started)
        self.assertEqual(len(launched), 1)
        argv, environment = launched[0]
        # The launched executable is the exact file the fresh preflight
        # observed -- the wrapper's later answer must not reselect it.
        self.assertEqual(Path(argv[0]), self.install.native_a.resolve())
        self.assertNotEqual(Path(argv[0]), self.install.native_b.resolve())
        # The exported key the launch must never see stays scrubbed.
        self.assertNotIn("ANTHROPIC_API_KEY", environment)
        self.assertNotIn(SECRET, json.dumps(environment))

    def test_worker_refuses_when_its_binding_is_missing_or_expired(self) -> None:
        """No current one-shot binding, no launch: the worker refuses first.

        A binding that is absent or expired at consumption means this
        attempt's own validated observation no longer stands behind a launch.
        The worker must refuse before any harness start instead of resolving
        PATH a second time -- the old behavior launched whatever the wrapper
        currently answered, a binary the admitted facts never described.
        """
        launched: list[list[str]] = []

        def launch(argv, cwd, prompt, timeout, *, environment=None, **kwargs):
            launched.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, packet_envelope(), "")

        with (
            mock.patch(
                "agent_execution.execution_status.run_in_process_group",
                return_value=subprocess.CompletedProcess(
                    ["claude", "--help"], 0, "--safe-mode --tools --restricted", ""
                ),
            ),
            mock.patch(
                "agent_execution.credentials.observe_claude_auth",
                return_value=self.subscription(),
            ),
            mock.patch(
                "agent_execution.worker.take_native_launch_binding",
                return_value=None,
            ),
            mock.patch("agent_execution.claude_execution.run_in_process_group", side_effect=launch),
        ):
            result = execute_worker(
                provider="claude-packet",
                model_call_id="missing-binding-regression",
                command=packet_command(),
                timeout=30,
                ctx_timeout=10,
                cwd=self.root,
                execution_transport="weft",
            )
        self.assertEqual(result.status, "preflight_failed", result.failure)
        self.assertFalse(result.model_call_started)
        self.assertIn("no current one-shot binding", result.failure)
        self.assertEqual(launched, [])

    def test_identity_mutation_between_help_and_auth_refuses_the_observation(self) -> None:
        """A binary that changes mid-preflight is never admitted as observed.

        The physical file is mutated after the help probe captured its
        identity but before the authentication probe answers. The originally
        observed identity is the one the preflight holds the binary to, so the
        capability is refused, nothing binds, and no harness starts.
        """
        launched: list[list[str]] = []

        def launch(argv, cwd, prompt, timeout, *, environment=None, **kwargs):
            launched.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, packet_envelope(), "")

        def authenticate(**kwargs):
            self.install.native_a.write_bytes(MACHO_MAGIC + b"tampered")
            return self.subscription()

        with (
            mock.patch(
                "agent_execution.execution_status.run_in_process_group",
                return_value=subprocess.CompletedProcess(
                    ["claude", "--help"], 0, "--safe-mode --tools --restricted", ""
                ),
            ),
            mock.patch("agent_execution.credentials.observe_claude_auth", side_effect=authenticate),
            mock.patch("agent_execution.claude_execution.run_in_process_group", side_effect=launch),
        ):
            result = execute_worker(
                provider="claude-packet",
                model_call_id="identity-mutation-regression",
                command=packet_command(),
                timeout=30,
                ctx_timeout=10,
                cwd=self.root,
                execution_transport="weft",
            )
        # The worker's own fact gate refuses -- not a launch-time surprise
        # after the facts already admitted the dispatch.
        self.assertEqual(result.status, "preflight_failed", result.failure)
        self.assertFalse(result.model_call_started)
        self.assertIn("execution capability", result.failure)
        self.assertIn("changed after its capability help probe", result.failure)
        self.assertEqual(launched, [])
        from agent_execution.claude_execution import take_native_launch_binding

        self.assertIsNone(take_native_launch_binding())


if __name__ == "__main__":
    unittest.main()
