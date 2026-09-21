from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from agent_execution import cli, provider_status


class ProviderStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patch = mock.patch.dict(
            os.environ,
            {
                "AGENT_PROVIDER_STATUS_DIR": str(self.root),
                "AGENT_PROVIDER_STATUS_SALT": "test-shared-salt",
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_probe_records_redacted_inventory_and_independent_quota_pools(self) -> None:
        payload = {
            "reports": [
                {
                    "provider": "google-antigravity",
                    "metadata": {"email": "ol*", "projectId": "ai*"},
                    "limits": [
                        {
                            "id": "google-antigravity:google:default:gemini-weekly",
                            "label": "Gemini",
                            "scope": {"windowId": "weekly"},
                            "window": {"resetsAt": 1_800_000_000_000},
                            "amount": {"remainingFraction": 0.05},
                            "status": "warning",
                        },
                        {
                            "id": "google-antigravity:anthropic:default:3p-weekly",
                            "label": "Claude & GPT (shared)",
                            "scope": {
                                "windowId": "weekly",
                                "sharedGroup": "3p-weekly:weekly",
                            },
                            "window": {"resetsAt": 1_800_000_000_000},
                            "amount": {"remainingFraction": 1.0},
                            "status": "ok",
                        },
                    ],
                }
            ],
            "accountsWithoutUsage": [],
            "disabledCredentials": [],
        }

        events = provider_status.observe_omp_usage(payload, host="studio", now=1_790_000_000)
        snapshot = provider_status.snapshot(host="studio", now=1_790_000_001)
        providers = cast(list[dict[str, object]], snapshot["providers"])

        self.assertEqual(len(events), 1)
        self.assertEqual(providers[0]["route"], "google-antigravity")
        self.assertEqual(providers[0]["state"], "available")
        quota = cast(list[dict[str, object]], providers[0]["quota"])
        self.assertEqual(
            [entry["id"] for entry in quota],
            [
                "google-antigravity:google:default:gemini-weekly",
                "google-antigravity:anthropic:default:3p-weekly",
            ],
        )
        serialized = json.dumps(snapshot)
        self.assertNotIn('"ol*"', serialized)
        self.assertIn("hmac-sha256:", serialized)

    def test_success_supersedes_an_older_refusal_for_the_same_subject(self) -> None:
        with mock.patch("agent_execution.provider_status.time.time", return_value=1_790_000_000):
            provider_status.record_refusal("kimi-code", "You've reached your 5-hour usage limit")
        with mock.patch("agent_execution.provider_status.time.time", return_value=1_790_000_010):
            provider_status.record_success("kimi-code")

        self.assertIsNone(provider_status.unavailable_reason("kimi-code", now=1_790_000_020))
        self.assertNotIn("kimi-code", provider_status.quota_blocked_routes(now=1_790_000_020))

    def test_refusal_keeps_provider_text_out_of_the_registry(self) -> None:
        signature = "You've reached your 5-hour usage limit for alice@example.test"

        provider_status.record_refusal("kimi-code", signature)

        serialized = json.dumps(provider_status.snapshot())
        self.assertNotIn("alice@example.test", serialized)
        self.assertIn("provider reported quota refusal", serialized)

    def test_dispatch_observation_starts_a_background_outbox_push(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"},
            ),
            mock.patch("agent_execution.provider_status.subprocess.Popen") as launch,
        ):
            provider_status.record_success("anthropic")

        command = launch.call_args.args[0]
        self.assertEqual(
            command[-4:],
            ["provider", "sync", "--push-only", "--json"],
        )

    def test_expired_refusal_is_not_authoritative(self) -> None:
        provider_status.observe(
            "openai-codex",
            kind="quota",
            state="unavailable",
            source_tool="test",
            source_method="fixture",
            ttl_seconds=10,
            now=1_790_000_000,
            detail={"reason": "weekly cap"},
        )

        snapshot = provider_status.snapshot(now=1_790_000_011)
        providers = cast(list[dict[str, object]], snapshot["providers"])

        self.assertEqual(snapshot["unavailable_routes"], {})
        self.assertTrue(providers[0]["stale"])

    def test_malformed_event_is_skipped_without_erasing_valid_observations(self) -> None:
        provider_status.observe(
            "anthropic",
            kind="inventory",
            state="available",
            source_tool="test",
            source_method="fixture",
            now=1_790_000_000,
        )
        malformed = self.root / "events" / "malformed.json"
        malformed.parent.mkdir(parents=True, exist_ok=True)
        malformed.write_text('{"schema_version":"provider-status/v2"}')

        snapshot = provider_status.snapshot(now=1_790_000_001)
        providers = cast(list[dict[str, object]], snapshot["providers"])

        self.assertEqual([item["route"] for item in providers], ["anthropic"])

    def test_sync_without_transport_preserves_outbox_and_reports_diagnostic(self) -> None:
        event = provider_status.observe(
            "anthropic",
            kind="inventory",
            state="available",
            source_tool="test",
            source_method="fixture",
        )

        diagnostics = provider_status.sync()

        self.assertEqual(diagnostics, ["no provider-status R2 transport is configured"])
        self.assertTrue((self.root / "outbox" / f"{event['event_id']}.json").exists())
        self.assertEqual(provider_status.snapshot()["pending_observations"], 1)

    def test_transport_timeout_preserves_outbox_and_becomes_a_diagnostic(self) -> None:
        event = provider_status.observe(
            "anthropic",
            kind="inventory",
            state="available",
            source_tool="test",
            source_method="fixture",
        )
        with (
            mock.patch.dict(
                os.environ,
                {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"},
            ),
            mock.patch(
                "agent_execution.provider_status._run",
                side_effect=subprocess.TimeoutExpired(["weft"], 20),
            ),
        ):
            diagnostics = provider_status.sync(push=True, pull=False)

        self.assertIn("timed out after 20s", diagnostics[0])
        self.assertTrue((self.root / "outbox" / f"{event['event_id']}.json").exists())

    def test_probe_output_is_scoped_to_the_observing_host(self) -> None:
        event = {"subject": {"host": "studio"}}
        response = {
            "schema_version": provider_status.SNAPSHOT_SCHEMA_VERSION,
            "providers": [],
        }
        with (
            mock.patch(
                "agent_execution.cli.provider_status.probe_omp",
                return_value=({}, [event]),
            ),
            mock.patch(
                "agent_execution.cli.provider_status.snapshot",
                return_value=response,
            ) as status,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = cli.main(["provider", "probe", "--json"])

        self.assertEqual(result, 0)
        status.assert_called_once_with(host="studio", diagnostics=[])


if __name__ == "__main__":
    unittest.main()
