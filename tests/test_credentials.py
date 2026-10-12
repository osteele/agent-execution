from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from agent_execution import credentials


class OmpCredentialBasisTests(unittest.TestCase):
    def test_coding_plan_api_key_is_subscription_basis_but_source_stays_api_key(self) -> None:
        command = ["bun", "omp-auth-status"]
        response = json.dumps(
            {
                "schema_version": "agent-execution.omp-auth/v1",
                "provider": "zhipu-coding-plan",
                "model": "glm-5.3-flash",
                "credential_type": "api_key",
            }
        )
        with (
            mock.patch(
                "agent_execution.omp_execution.omp_auth_status_command", return_value=command
            ),
            mock.patch(
                "agent_execution.credentials._resolved_status_executable", return_value="bun"
            ),
            mock.patch(
                "agent_execution.credentials._omp_cache_profile", return_value="test-sdk"
            ),
            mock.patch(
                "agent_execution.credentials._cache_fingerprint", return_value="test-fingerprint"
            ),
            mock.patch(
                "agent_execution.credentials._run_status_command",
                return_value=subprocess.CompletedProcess(command, 0, response, ""),
            ),
        ):
            basis = credentials.observe_credential_basis(
                "omp",
                state_root=None,
                environment={"PATH": "/bin", "ZAI_API_KEY": "plan-key"},
                profile="zhipu-coding-plan/glm-5.3-flash",
            )

        self.assertEqual(basis.basis, credentials.BASIS_SUBSCRIPTION)
        self.assertEqual(basis.reported_source, "api_key")
        self.assertEqual(basis.to_dict()["schema_version"], "harness-credential-basis/v1")

    def test_metered_zai_api_key_remains_api_key_basis(self) -> None:
        with (
            mock.patch(
                "agent_execution.credentials._AUTH_STATUS_PROBES",
                {"omp": lambda *args, **kwargs: "api_key"},
            ),
            mock.patch(
                "agent_execution.credentials._resolved_status_executable", return_value="bun"
            ),
            mock.patch(
                "agent_execution.credentials._omp_cache_profile", return_value="test-sdk"
            ),
            mock.patch(
                "agent_execution.credentials._cache_fingerprint", return_value="test-fingerprint"
            ),
        ):
            basis = credentials.observe_credential_basis(
                "omp",
                state_root=None,
                environment={"PATH": "/bin"},
                profile="zai/glm-5.3-flash",
            )
        self.assertEqual(basis.basis, credentials.BASIS_API_KEY)
        self.assertEqual(basis.reported_source, "api_key")

    def test_old_cached_basis_semantics_are_reprobed(self) -> None:
        self.assertEqual(credentials.CREDENTIAL_CACHE_VERSION, "credential-basis-cache/v4")
        cached = {
            "cache_version": "credential-basis-cache/v3",
            "basis": credentials.BASIS_API_KEY,
            "reported_source": "api_key",
            "observed_at": 10.0,
            "fingerprint": "same",
        }
        self.assertIsNone(
            credentials._cached_observation(
                cached,
                fingerprint="same",
                now=11.0,
                ttl=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
