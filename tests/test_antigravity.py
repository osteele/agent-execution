"""Antigravity is an OAuth-only OMP route with independent model-family quota."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_execution import costs
from agent_execution.credentials import CredentialBasis
from agent_execution.omp_execution import (
    GROUNDED_OMP_SELECTORS,
    GROUNDED_OMP_SELECTORS_BY_PROVIDER,
    OMP_WRITER_SELECTORS,
    omp_launch_environment,
    omp_transcript,
    validate_omp_command,
)
from tests.support.omp import omp_command, omp_output


class AntigravityTest(unittest.TestCase):
    def test_selector_registries_commands_and_transcripts(self) -> None:
        gemini = "google-antigravity/gemini-3.1-pro"
        opus = "google-antigravity/claude-opus-4-6"
        sonnet = "google-antigravity/claude-sonnet-4-6"
        self.assertEqual(GROUNDED_OMP_SELECTORS_BY_PROVIDER["google-antigravity"], gemini)
        self.assertEqual(
            {s for s in GROUNDED_OMP_SELECTORS if s.startswith("google-antigravity/")},
            {gemini, opus, sonnet},
        )
        self.assertEqual(
            {s for s in OMP_WRITER_SELECTORS if s.startswith("google-antigravity/")},
            {gemini, opus},
        )
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            for selector in (gemini, opus, sonnet):
                with self.subTest(selector=selector):
                    self.assertEqual(
                        validate_omp_command(
                            omp_command(directory, selector=selector), cwd
                        ).selector,
                        selector,
                    )
                    self.assertEqual(
                        omp_transcript(omp_output(cwd=directory, selector=selector))["header"][
                            "selector"
                        ],
                        selector,
                    )
            for selector in (gemini, opus):
                with self.subTest(writer=selector):
                    validate_omp_command(
                        omp_command(directory, selector=selector, writer=True), cwd
                    )
                    omp_transcript(
                        omp_output(
                            cwd=directory, selector=selector, policy="workspace-write-no-shell"
                        )
                    )
            with self.assertRaisesRegex(ValueError, "writer OMP selector"):
                validate_omp_command(omp_command(directory, selector=sonnet, writer=True), cwd)
            with self.assertRaisesRegex(ValueError, "grounded OMP selector"):
                validate_omp_command(
                    omp_command(directory, selector="google-antigravity/gemini-unknown"), cwd
                )

    def test_launch_never_forwards_google_keys(self) -> None:
        launch = omp_launch_environment(
            {
                "HOME": "/tmp/home",
                "GOOGLE_API_KEY": "key",
                "GEMINI_API_KEY": "key",
                "GOOGLE_GENERATIVE_AI_API_KEY": "key",
                "ZAI_API_KEY": "coding-plan",
            }
        )
        self.assertEqual(launch["ZAI_API_KEY"], "coding-plan")
        self.assertFalse(any("GOOGLE" in key or "GEMINI" in key for key in launch))

    def test_both_adapters_probe_all_antigravity_models_and_never_price_unknown_as_free(
        self,
    ) -> None:
        self.assertNotIn("google-antigravity", costs.OMP_SUBSCRIPTION_PROVIDERS)
        for adapter in ("omp", "omp-packet"):
            for model in ("gemini-3.1-pro", "claude-opus-4-6"):
                selector = f"google-antigravity/{model}"
                profile = {"adapter": adapter, "provider": "google-antigravity", "model": selector}
                for basis, source, expected in (
                    ("subscription", "oauth", 0.0),
                    ("api_key", "api_key", None),
                    ("unobserved", None, None),
                ):
                    with self.subTest(adapter=adapter, selector=selector, source=source):
                        observation = CredentialBasis("omp", basis, source, 0.0, "fingerprint")
                        with mock.patch(
                            "agent_execution.costs.observe_credential_basis",
                            return_value=observation,
                        ) as probe:
                            estimate = costs.estimate_execution_cost(profile, refresh=True)
                        probe.assert_called_once_with(
                            "omp",
                            profile=selector,
                            state_root=None,
                            cwd=None,
                            environment=None,
                            refresh=True,
                        )
                        self.assertEqual(estimate.maximum_incremental_usd, expected)
                        self.assertEqual(estimate.probe_answered, source is not None)
