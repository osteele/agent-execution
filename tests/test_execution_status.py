"""Independent fact lifetimes, exact identity boundaries, and native launch auth."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from agent_execution import credentials, execution_status, provider_status

SELECTOR = "anthropic/claude-opus-5-5"


class ExecutionStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch.dict(
            os.environ,
            {
                "AGENT_PROVIDER_STATUS_DIR": str(self.root / "observations"),
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def query(self, **overrides):
        return dict(
            harness="omp",
            surface="offload-task",
            selectors=[SELECTOR],
            transport="weft",
            **overrides,
        )

    def fresh(self, *, model=True, credential=True, **overrides):
        def run(command, cwd, prompt, timeout, **kwargs):
            selectors = json.loads(command[command.index("--probe-models") - 1])
            output = {
                "schema_version": "agent-execution.omp-writer-readiness/v1",
                "writers": [
                    {
                        "selector": selector,
                        "model_available": model,
                        "credential_available": credential,
                        "available": model and credential,
                        "detail": "Pinned SDK observation",
                    }
                    for selector in selectors
                ],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(output), "")

        basis = credentials.CredentialBasis(
            "omp", credentials.BASIS_SUBSCRIPTION, "oauth", time.time(), "config"
        )
        with (
            mock.patch(
                "agent_execution.execution_status.require_omp_sdk", return_value=(self.root, "bun")
            ),
            mock.patch(
                "agent_execution.omp_execution.require_omp_sdk", return_value=(self.root, "bun")
            ),
            mock.patch("agent_execution.execution_status.run_in_process_group", side_effect=run),
            mock.patch("agent_execution.credentials.observe_credential_basis", return_value=basis),
        ):
            arguments = self.query()
            arguments.update(overrides)
            return execution_status.probe_status(**arguments)

    def test_catalog_credentials_quota_and_generation_are_independent(self):
        document = self.fresh(credential=False)
        facts = document["rows"][0]["facts"]
        self.assertEqual(facts["capability"]["state"], "available")
        self.assertEqual(facts["capability"]["detail"], f"Pinned SDK model observed for {SELECTOR}")
        self.assertEqual(facts["authentication"]["state"], "unavailable")
        self.assertEqual(
            facts["authentication"]["detail"], "Pinned SDK execution-eligible credentials absent"
        )
        self.assertEqual(facts["quota"]["state"], "unknown")
        self.assertEqual(facts["generation"]["state"], "unknown")
        self.assertEqual(document["rows"][0]["credential_basis"]["basis"], "subscription")

    def test_login_does_not_erase_generation_or_quota_refusal(self):
        subject = self.fresh()["rows"][0]["subject"]
        execution_status.record_generation(subject, succeeded=False, condition="quota")
        facts = self.fresh()["rows"][0]["facts"]
        self.assertEqual(facts["authentication"]["state"], "available")
        self.assertEqual(facts["generation"]["state"], "unavailable")
        self.assertEqual(facts["quota"]["state"], "unavailable")
        self.assertEqual(facts["quota"]["condition"], "quota")
        self.assertEqual(facts["generation"]["source"]["method"], "provider-generation")
        # A later actual answer is not a measurement of remaining headroom.
        execution_status.record_generation(subject, succeeded=True)
        facts = execution_status.cached_status(**self.query())["rows"][0]["facts"]
        self.assertEqual(facts["generation"]["state"], "available")
        self.assertEqual(facts["quota"]["state"], "unknown")

    def test_expired_failure_is_retained_with_age_not_reclassified_as_success(self):
        subject = self.fresh()["rows"][0]["subject"]
        execution_status.record_generation(subject, succeeded=False, condition="auth")
        with mock.patch(
            "agent_execution.execution_status.time.time", return_value=time.time() + 700
        ):
            facts = execution_status.cached_status(**self.query())["rows"][0]["facts"]
        self.assertEqual(facts["generation"]["state"], "unavailable")
        self.assertTrue(facts["generation"]["stale"])
        self.assertGreater(facts["generation"]["age_seconds"], 600)

    def test_unknown_credentials_do_not_join_unscoped_or_other_execution_contexts(self):
        subject = self.fresh()["rows"][0]["subject"]
        execution_status.record_generation(subject, succeeded=False, condition="quota")
        provider_status.record_refusal("anthropic", "rate limit exceeded")
        for changes in (
            {"requester_user": "another-user"},
            {"requester_host": "another-host"},
            {"selectors": ["openai-codex/gpt-6-astra"]},
            {"surface": "worker", "tool_policy": "read-only-no-shell"},
            {"environment": {**os.environ, "HOME": str(self.root / "another-home")}},
        ):
            arguments = self.query()
            arguments.update(changes)
            with self.subTest(changes=changes):
                facts = execution_status.cached_status(**arguments)["rows"][0]["facts"]
                self.assertEqual(facts["generation"]["state"], "unknown")
                self.assertEqual(facts["quota"]["state"], "unknown")
        # Exact refusals do not poison the legacy route-wide availability map.
        exact = execution_status.cached_status(**self.query())["rows"][0]
        self.assertEqual(exact["facts"]["quota"]["state"], "unavailable")

    def test_read_only_selector_does_not_need_writer_membership(self):
        readonly = self.fresh(
            surface="worker", selectors=["kimi-code/kimi-k2.5"], tool_policy="read-only-no-shell"
        )
        self.assertEqual(readonly["rows"][0]["facts"]["capability"]["state"], "available")
        writer = self.fresh(selectors=["kimi-code/kimi-k2.5"])
        self.assertEqual(writer["rows"][0]["facts"]["capability"]["state"], "unavailable")

    def test_native_worker_requires_exact_model_and_reports_transport_independently(self):
        with (
            mock.patch(
                "agent_execution.claude_execution.native_claude_launch",
                side_effect=FileNotFoundError,
            ),
            self.assertRaisesRegex(ValueError, "exact selector"),
        ):
            execution_status.probe_status(harness="claude", surface="worker", transport="weft")
        with mock.patch(
            "agent_execution.claude_execution.native_claude_launch",
            side_effect=FileNotFoundError,
        ):
            document = execution_status.probe_status(
                harness="claude",
                surface="worker",
                selectors=[SELECTOR],
                transport="weft",
            )
        facts = document["rows"][0]["facts"]
        self.assertEqual(facts["capability"]["state"], "unavailable")
        self.assertEqual(facts["transport"]["state"], "available")
        self.assertEqual(facts["authentication"]["state"], "unknown")
        self.assertEqual(document["rows"][0]["credential_basis"]["basis"], "unobserved")

    def test_native_claude_worker_support_matrix_and_independent_observations(self):
        executable = self.root / "claude-native"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        launch = {"PATH": str(self.root), "HOME": str(self.root)}
        basis = credentials.CredentialBasis(
            "claude", credentials.BASIS_SUBSCRIPTION, "none", time.time(), "safe-fingerprint"
        )
        observed = mock.Mock(state="unavailable", answered=True, basis=basis)
        with (
            mock.patch(
                "agent_execution.claude_execution.native_claude_launch",
                return_value=(str(executable), launch),
            ),
            mock.patch(
                "agent_execution.execution_status.run_in_process_group",
                return_value=subprocess.CompletedProcess(
                    ["claude", "--help"], 0, "--safe-mode --tools --restricted", ""
                ),
            ),
            mock.patch(
                "agent_execution.credentials.observe_claude_auth", return_value=observed
            ) as auth,
        ):
            readonly = execution_status.probe_status(
                harness="claude",
                surface="worker",
                selectors=[SELECTOR],
                transport="weft",
                requester_host="requester",
                requester_user="reviewer",
            )
            packet = execution_status.probe_status(
                harness="claude-packet",
                surface="worker",
                selectors=[SELECTOR],
                transport="weft",
                requester_host="requester",
                requester_user="reviewer",
            )
            mismatched = execution_status.probe_status(
                harness="claude-packet",
                surface="worker",
                selectors=[SELECTOR],
                tool_policy="read-only-no-shell",
                transport="weft",
            )
            offload = execution_status.probe_status(
                harness="claude",
                surface="offload-task",
                selectors=[SELECTOR],
                tool_policy="workspace-write-no-shell",
            )
        self.assertEqual(readonly["rows"][0]["subject"]["tool_policy"], "read-only-no-shell")
        self.assertEqual(readonly["rows"][0]["subject"]["effective_route"], "anthropic")
        self.assertEqual(readonly["rows"][0]["facts"]["capability"]["state"], "available")
        self.assertEqual(readonly["rows"][0]["facts"]["transport"]["state"], "available")
        self.assertEqual(readonly["rows"][0]["facts"]["authentication"]["state"], "unavailable")
        self.assertEqual(readonly["rows"][0]["credential_basis"]["basis"], "subscription")
        self.assertEqual(packet["rows"][0]["subject"]["tool_policy"], "packet-only-no-tools")
        self.assertEqual(packet["rows"][0]["facts"]["capability"]["state"], "available")
        self.assertEqual(packet["rows"][0]["credential_basis"]["harness"], "claude")
        self.assertEqual(mismatched["rows"][0]["facts"]["capability"]["state"], "unavailable")
        self.assertEqual(offload["rows"][0]["facts"]["capability"]["state"], "unavailable")
        self.assertEqual(auth.call_count, 2)

    def test_native_worker_refuses_profiles_and_non_native_selectors(self):
        with self.assertRaisesRegex(ValueError, "wrapper profiles"):
            execution_status.probe_status(
                harness="claude",
                surface="worker",
                selectors=[SELECTOR],
                profile="review",
            )
        with mock.patch(
            "agent_execution.claude_execution.native_claude_launch",
            side_effect=FileNotFoundError,
        ):
            document = execution_status.probe_status(
                harness="claude",
                surface="worker",
                selectors=["anthropic/opus"],
                transport="weft",
            )
        self.assertEqual(document["rows"][0]["facts"]["capability"]["state"], "unavailable")

    def test_malformed_native_auth_stays_unknown_without_erasing_capability_or_basis(self):
        executable = self.root / "claude"
        executable.write_text("#!/bin/sh\nprintf '%s' '{\"loggedIn\":true'\n")
        executable.chmod(0o755)
        launch = {"PATH": str(self.root), "HOME": str(self.root)}
        with (
            mock.patch(
                "agent_execution.claude_execution.native_claude_launch",
                return_value=(str(executable), launch),
            ),
            mock.patch(
                "agent_execution.execution_status.run_in_process_group",
                return_value=subprocess.CompletedProcess(
                    ["claude", "--help"], 0, "--safe-mode --tools --restricted", ""
                ),
            ),
        ):
            document = execution_status.probe_status(
                harness="claude-packet",
                surface="worker",
                selectors=[SELECTOR],
                transport="weft",
            )
        row = document["rows"][0]
        self.assertEqual(row["facts"]["capability"]["state"], "available")
        self.assertEqual(row["facts"]["authentication"]["state"], "unknown")
        self.assertEqual(row["credential_basis"]["basis"], "unobserved")
        self.assertEqual(row["credential_basis"]["harness"], "claude")

    def test_native_fingerprint_tracks_resolved_binary_and_scrubbed_environment(self):
        first = self.root / "native-one"
        second = self.root / "native-two"
        first.write_text("#!/bin/sh\n")
        second.write_text("#!/bin/sh\n")
        first.chmod(0o755)
        second.chmod(0o755)
        with mock.patch(
            "agent_execution.claude_execution.native_claude_launch",
            side_effect=[
                (str(first), {"PATH": str(self.root), "HOME": str(self.root / "home-a")}),
                (str(second), {"PATH": str(self.root), "HOME": str(self.root / "home-b")}),
            ],
        ):
            one = execution_status._subject(
                "claude", "worker", SELECTOR, "weft", None, None, "a" * 64, None, None, {}, None
            )
            two = execution_status._subject(
                "claude", "worker", SELECTOR, "weft", None, None, "a" * 64, None, None, {}, None
            )
        self.assertNotEqual(one["environment_fingerprint"], two["environment_fingerprint"])

    def test_native_quota_observation_survives_a_new_ssh_connection(self):
        executable = self.root / "claude"
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        launch = {"PATH": str(self.root), "HOME": str(self.root)}
        with mock.patch(
            "agent_execution.claude_execution.native_claude_launch",
            side_effect=[
                (str(executable), {**launch, "SSH_CONNECTION": "client 1000 server 22"}),
                (str(executable), {**launch, "SSH_CONNECTION": "client 2000 server 22"}),
            ],
        ):
            subject = execution_status.cached_status(
                harness="claude", surface="worker", selectors=[SELECTOR], transport="weft"
            )["rows"][0]["subject"]
            execution_status.record_generation(subject, succeeded=False, condition="quota")
            row = execution_status.cached_status(
                harness="claude", surface="worker", selectors=[SELECTOR], transport="weft"
            )["rows"][0]
        self.assertEqual(row["facts"]["quota"]["state"], "unavailable")
        self.assertFalse(row["facts"]["quota"]["stale"])

    def test_exact_observations_do_not_rewrite_the_legacy_projection(self):
        provider_status.record_success("anthropic")
        subject = self.fresh()["rows"][0]["subject"]
        provider_status.snapshot()
        projection = self.root / "observations" / "projection.json"
        before = projection.read_bytes()
        execution_status.record_generation(subject, succeeded=False, condition="quota")
        self.assertEqual(projection.read_bytes(), before)
        self.assertEqual(provider_status.snapshot()["unavailable_routes"], {})
        self.assertEqual(projection.read_bytes(), before)

    def test_wrong_pinned_build_is_rejected_before_probing(self):
        with mock.patch("agent_execution.execution_status.run_in_process_group") as run:
            with self.assertRaisesRegex(ValueError, "source identity"):
                execution_status.probe_status(**self.query(), expected_execution_sha256="0" * 64)
            run.assert_not_called()

    def test_validator_rejects_forged_freshness_identity_and_missing_facts(self):
        original = self.fresh()
        mutations = (
            lambda value: value["rows"][0]["facts"].pop("quota"),
            lambda value: value["rows"][0].pop("credential_basis"),
            lambda value: value["rows"][0]["facts"]["generation"].pop("observed_at"),
            lambda value: value["rows"][0]["facts"]["generation"].update(state=[]),
            lambda value: value["rows"][0]["facts"]["generation"].update(condition={}),
            lambda value: value["worker_identity"].update(package_version=None),
            lambda value: value["rows"][0]["subject"].update(fingerprint_scope="global"),
            lambda value: value["rows"][0]["credential_basis"].update(harness="claude"),
            lambda value: value["rows"][0]["facts"]["generation"].update(state="available"),
            lambda value: value["rows"][0]["facts"]["authentication"].update(age_seconds=-1),
            lambda value: value["rows"][0]["facts"]["authentication"].update(stale=True),
            lambda value: value["rows"][0]["subject"].update(execution_sha256="0" * 64),
            lambda value: value["rows"][0]["subject"].update(os_user="someone-else"),
            lambda value: value["rows"].append(copy.deepcopy(value["rows"][0])),
            lambda value: value["rows"][0]["facts"]["generation"].update(condition="made-up"),
        )
        for mutation in mutations:
            value = copy.deepcopy(original)
            mutation(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                execution_status.validate_status(value)

    def test_remote_checker_rejects_valid_but_wrong_request_context(self):
        value = self.fresh(requester_host="other-laptop", requester_user="other-user")
        output = subprocess.CompletedProcess(["ssh"], 0, json.dumps(value), "")
        with (
            mock.patch("agent_execution.execution_status.subprocess.run", return_value=output),
            self.assertRaisesRegex(ValueError, "context differs"),
        ):
            execution_status.probe_remote_status(
                host="studio", harness="omp", surface="offload-task", selectors=[SELECTOR]
            )

    def test_native_auth_uses_scrubbed_profile_environment_without_retaining_secrets(self):
        executable = self.root / "claude"
        capture = self.root / "launch"
        executable.write_text(
            "#!/bin/sh\n"
            f'printf \'%s|%s|%s\' "${{ANTHROPIC_API_KEY-unset}}" "${{ANTHROPIC_AUTH_TOKEN-unset}}" "$CLAUDE_PROFILE" > \'{capture}\'\n'
            'printf \'%s\' \'{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"max","apiKeySource":"none","email":"PRIVATE-EMAIL","token":"PRIVATE-TOKEN"}\'\n'
        )
        executable.chmod(0o755)
        environment = {
            **os.environ,
            "PATH": str(self.root),
            "ANTHROPIC_API_KEY": "PRIVATE-KEY",
            "ANTHROPIC_AUTH_TOKEN": "PRIVATE-AUTH",
        }

        def native_status(*, cached=False):
            observer = execution_status.cached_status if cached else execution_status.probe_status
            return observer(
                harness="claude",
                surface="native",
                selectors=[SELECTOR],
                profile="subscription",
                environment=environment,
                tool_policy="read-only-no-shell",
                cwd=self.root,
            )

        document = native_status()
        row = document["rows"][0]
        self.assertEqual(capture.read_text(), "unset|unset|subscription")
        # Auth bypasses wrapper routing. Positive sign-in is not generation
        # eligibility or subscription billing for the configured route.
        observed = credentials.observe_claude_auth(
            environment=environment, profile="subscription", cwd=self.root
        )
        self.assertEqual(observed.state, "available")
        self.assertEqual(observed.basis.basis, "subscription")
        self.assertEqual(row["facts"]["authentication"]["state"], "unknown")
        self.assertEqual(row["facts"]["capability"]["state"], "unknown")
        self.assertEqual(row["credential_basis"]["basis"], "unobserved")
        self.assertIsNone(row["subject"]["effective_route"])
        with self.assertRaisesRegex(ValueError, "verified launch route"):
            execution_status.record_generation(row["subject"], succeeded=False, condition="auth")
        cached = native_status(cached=True)
        self.assertEqual(cached["rows"][0]["facts"]["generation"]["state"], "unknown")
        self.assertNotIn("PRIVATE-", json.dumps(document))
        for path in (self.root / "observations").rglob("*.json"):
            self.assertNotIn("PRIVATE-", path.read_text())

    def test_agy_sign_in_observation_and_independence(self):
        agy_bin = self.root / "agy"
        agy_bin.write_text("#!/bin/sh\nexit 0\n")
        agy_bin.chmod(0o755)
        launch_env = {"PATH": str(self.root), "HOME": str(self.root)}

        # 1. Signed in
        with mock.patch(
            "agent_execution.credentials._run_status_command",
            return_value=subprocess.CompletedProcess(
                ["agy", "models"], 0, "gemini-3.1-pro-high\tGemini 3.1 Pro\n", ""
            ),
        ):
            status = execution_status.probe_status(
                harness="agy",
                surface="worker",
                selectors=["google-antigravity/gemini-3.1-pro-high"],
                tool_policy="packet-only-no-tools",
                environment=launch_env,
            )
            row = status["rows"][0]
            self.assertEqual(row["facts"]["capability"]["state"], "available")
            self.assertEqual(row["facts"]["authentication"]["state"], "available")
            self.assertEqual(
                row["facts"]["authentication"]["detail"],
                "Antigravity CLI authentication is available",
            )
            self.assertEqual(row["credential_basis"]["basis"], "not_observable")

        # 2. Signed out (please sign in)
        with mock.patch(
            "agent_execution.credentials._run_status_command",
            return_value=subprocess.CompletedProcess(
                ["agy", "models"], 1, "Error: Please sign in to view available models.\n", ""
            ),
        ):
            status = execution_status.probe_status(
                harness="agy",
                surface="worker",
                selectors=["google-antigravity/gemini-3.1-pro-high"],
                tool_policy="packet-only-no-tools",
                environment=launch_env,
            )
            row = status["rows"][0]
            self.assertEqual(row["facts"]["capability"]["state"], "available")
            self.assertEqual(row["facts"]["authentication"]["state"], "unavailable")
            self.assertEqual(
                row["facts"]["authentication"]["detail"],
                "Antigravity CLI authentication is unavailable",
            )
            self.assertEqual(row["credential_basis"]["basis"], "not_observable")

        # 3. agy binary missing
        empty_env = {"PATH": "/nonexistent", "HOME": str(self.root)}
        status = execution_status.probe_status(
            harness="agy",
            surface="worker",
            selectors=["google-antigravity/gemini-3.1-pro-high"],
            tool_policy="packet-only-no-tools",
            environment=empty_env,
        )
        row = status["rows"][0]
        self.assertEqual(row["facts"]["capability"]["state"], "unavailable")
        self.assertEqual(row["facts"]["authentication"]["state"], "unavailable")
        self.assertEqual(row["credential_basis"]["basis"], "not_observable")


if __name__ == "__main__":
    unittest.main()
