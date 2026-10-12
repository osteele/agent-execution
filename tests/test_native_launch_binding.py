"""Launch-binding API contracts: single use, staleness, and refusal semantics.

The behavioral regression for the worker's launch lives in
``test_native_launch_reselection``. This module pins the binding API itself:
it is consumed by exactly one launch, a stale or consumed binding is never
honored, a replaced or removed observed binary refuses rather than launching,
and only an available worker-surface observation records a binding. The
binding API is imported lazily inside each test, so on the unmodified
baseline each of these tests fails at that import while
``test_native_launch_reselection`` carries the behavioral baseline failure
for the boundary itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from agent_execution.credentials import BASIS_SUBSCRIPTION, CredentialBasis
from tests.support.native_claude import (
    FakeNativeInstall,
    packet_command,
    packet_envelope,
    probe_worker_surface,
)


class NativeLaunchBindingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.install = FakeNativeInstall(self.root)
        environment = mock.patch.dict(
            os.environ,
            {
                **self.install.environment(),
                "ANTHROPIC_API_KEY": "sk-test-anthropic-secret",
                "AGENT_PROVIDER_STATUS_DIR": str(self.root / "observations"),
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        from agent_execution.claude_execution import (
            discard_native_launch_binding,
            take_native_launch_binding,
        )

        discard_native_launch_binding()
        self.take = take_native_launch_binding

    def observe(self) -> None:
        basis = CredentialBasis("claude", BASIS_SUBSCRIPTION, "none", time.time(), "fixture")
        with (
            mock.patch(
                "agent_execution.execution_status.run_in_process_group",
                return_value=subprocess.CompletedProcess(
                    ["claude", "--help"], 0, "--safe-mode --tools --restricted", ""
                ),
            ),
            mock.patch(
                "agent_execution.credentials.observe_claude_auth",
                return_value=mock.Mock(state="available", answered=True, basis=basis),
            ),
        ):
            probe_worker_surface()

    def test_binding_is_single_use_and_a_stale_one_is_discarded(self) -> None:
        from agent_execution import claude_execution

        self.assertIsNone(self.take())
        self.observe()
        binding = self.take()
        assert binding is not None
        self.assertEqual(Path(binding.executable), self.install.native_a.resolve())
        self.assertIsNone(self.take())
        stale = claude_execution.NativeClaudeLaunchBinding(
            executable=str(self.install.native_a),
            environment={"PATH": str(self.root)},
            identity="stale",
            observed_at=time.time() - 600,
        )
        claude_execution._LAUNCH_BINDING = stale
        self.assertIsNone(self.take())
        self.assertIsNone(claude_execution._LAUNCH_BINDING)

    def test_replaced_or_removed_observed_binary_refuses_the_launch(self) -> None:
        from agent_execution.claude_execution import run_claude_command

        self.observe()
        binding = self.take()
        assert binding is not None
        replaced = mock.patch("agent_execution.claude_execution.run_in_process_group")
        self.install.native_a.write_bytes(self.install.native_a.read_bytes() + b"patched")
        with replaced as process_group:
            with self.assertRaises(FileNotFoundError):
                run_claude_command(packet_command(), self.root, 30, launch=binding)
            process_group.assert_not_called()
        self.install.native_a.unlink()
        with (
            mock.patch("agent_execution.claude_execution.run_in_process_group"),
            self.assertRaises(FileNotFoundError),
        ):
            run_claude_command(packet_command(), self.root, 30, launch=binding)

    def test_a_binding_launch_reuses_the_observed_binary_without_reresolving(self) -> None:
        from agent_execution.claude_execution import run_claude_command

        self.observe()
        binding = self.take()
        assert binding is not None
        self.install.point_to(self.install.native_b)
        captured: dict[str, object] = {}

        def launch(argv, cwd, prompt, timeout, *, environment=None, **kwargs):
            captured["argv"] = list(argv)
            captured["environment"] = dict(environment or {})
            return subprocess.CompletedProcess(argv, 0, packet_envelope(), "")

        with mock.patch(
            "agent_execution.claude_execution.run_in_process_group", side_effect=launch
        ):
            run_claude_command(packet_command(), self.root, 30, launch=binding)
        self.assertEqual(
            Path(cast(list[str], captured["argv"])[0]), self.install.native_a.resolve()
        )

    def test_failed_observations_never_bind(self) -> None:
        with mock.patch(
            "agent_execution.claude_execution.native_claude_launch",
            side_effect=FileNotFoundError("unavailable"),
        ):
            document = probe_worker_surface()
        self.assertEqual(document["rows"][0]["facts"]["capability"]["state"], "unavailable")
        self.assertIsNone(self.take())
        # The failed help probe is driven through `probe_worker_surface`'s own
        # parameter: an outer patch of the process boundary would be shadowed
        # by the helper's inner process-boundary patch and never observed.
        document = probe_worker_surface(help_output="")
        self.assertEqual(document["rows"][0]["facts"]["capability"]["state"], "unavailable")
        self.assertIsNone(self.take())

    def test_status_output_stays_free_of_secrets_and_the_schema_is_unchanged(self) -> None:
        document = probe_worker_surface()
        text = json.dumps(document, sort_keys=True)
        self.assertNotIn("sk-test-anthropic-secret", text)
        self.assertEqual(document["schema_version"], "agent-execution.execution-status/v1")
        self.assertEqual(document["rows"][0]["facts"]["capability"]["state"], "available")
        binding = self.take()
        assert binding is not None
        self.assertNotIn("ANTHROPIC_API_KEY", binding.environment)


if __name__ == "__main__":
    unittest.main()
