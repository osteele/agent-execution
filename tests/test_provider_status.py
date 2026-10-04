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

    def test_dispatch_billing_pools_are_split_only_for_antigravity(self) -> None:
        for route, model, pool in (
            ("google-antigravity", "gemini-3.1-pro", "google-antigravity/gemini"),
            (
                "google-antigravity",
                "google-antigravity/claude-opus-4-6",
                "google-antigravity/other",
            ),
            ("anthropic", "claude-opus-5-5", "anthropic"),
            ("openai-codex", "gpt-6-sol", "openai-codex"),
            ("zhipu-coding-plan", "glm-5.3-flash", "zhipu-coding-plan"),
            ("kimi-code", "k3", "kimi-code"),
        ):
            with self.subTest(route=route, model=model):
                success = provider_status.record_success(route, model=model)
                refusal = provider_status.record_refusal(route, "usage limit", model=model)
                for event in (success, refusal):
                    self.assertEqual(event["subject"]["route"], route)
                    self.assertEqual(event["subject"]["billing_pool"], pool)

    def test_antigravity_success_does_not_clear_other_family_refusal(self) -> None:
        with mock.patch("agent_execution.provider_status.time.time", return_value=1_790_000_000):
            provider_status.record_refusal(
                "google-antigravity",
                "You've reached your 5-hour usage limit",
                model="claude-opus-4-6",
            )
        with mock.patch("agent_execution.provider_status.time.time", return_value=1_790_000_010):
            provider_status.record_success("google-antigravity", model="gemini-3.1-pro")
        providers = cast(
            list[dict[str, object]], provider_status.snapshot(now=1_790_000_020)["providers"]
        )
        self.assertTrue(
            any(
                item["billing_pool"] == "google-antigravity/other"
                and item["state"] == "unavailable"
                for item in providers
            )
        )
        self.assertTrue(
            any(
                item["billing_pool"] == "google-antigravity/gemini" and item["state"] == "available"
                for item in providers
            )
        )

    def test_success_supersedes_an_older_refusal_for_the_same_subject(self) -> None:
        with mock.patch("agent_execution.provider_status.time.time", return_value=1_790_000_000):
            provider_status.record_refusal("kimi-code", "You've reached your 5-hour usage limit")
        with mock.patch("agent_execution.provider_status.time.time", return_value=1_790_000_010):
            provider_status.record_success("kimi-code")

        self.assertIsNone(provider_status.unavailable_reason("kimi-code", now=1_790_000_020))
        self.assertNotIn("kimi-code", provider_status.quota_blocked_routes(now=1_790_000_020))

    def test_observation_order_is_by_instant_not_by_timestamp_spelling(self) -> None:
        """Events arrive from other hosts; a valid RFC 3339 offset must order by instant."""
        refusal = provider_status.observe(
            "kimi-code",
            kind="quota",
            state="unavailable",
            source_tool="test",
            source_method="fixture",
            host="other-host",
            os_user="agent",
            now=1_790_000_000,
            detail={"reason": "weekly cap"},
        )
        success = provider_status.observe(
            "kimi-code",
            kind="availability",
            state="available",
            source_tool="test",
            source_method="fixture",
            host="other-host",
            os_user="agent",
            now=1_790_000_000,
        )
        # 10:00Z spelled in +08:00 (sorts after "11:00Z" as text), then 11:00Z.
        for event, observed in (
            (refusal, "2026-10-04T18:00:00+08:00"),
            (success, "2026-10-04T11:00:00Z"),
        ):
            event["observed_at"] = observed
            event["expires_at"] = "2026-10-05T00:00:00Z"
            path = self.root / "outbox" / f"{event['event_id']}.json"
            path.write_text(json.dumps(event))

        now = provider_status._timestamp("2026-10-04T12:00:00Z")
        self.assertIsNone(provider_status.unavailable_reason("kimi-code", now=now))

    def test_the_latest_observation_of_one_kind_is_chosen_by_instant(self) -> None:
        older = provider_status.observe(
            "kimi-code",
            kind="quota",
            state="available",
            source_tool="test",
            source_method="fixture",
            host="other-host",
            os_user="agent",
            now=1_790_000_000,
        )
        newer = provider_status.observe(
            "kimi-code",
            kind="quota",
            state="unavailable",
            source_tool="test",
            source_method="fixture",
            host="other-host",
            os_user="agent",
            now=1_790_000_000,
            detail={"reason": "weekly cap"},
        )
        # The older event's spelling sorts after the newer one's as text.
        for event, observed in (
            (older, "2026-10-04T18:00:00+08:00"),
            (newer, "2026-10-04T11:00:00Z"),
        ):
            event["observed_at"] = observed
            event["expires_at"] = "2026-10-05T00:00:00Z"
            (self.root / "outbox" / f"{event['event_id']}.json").write_text(json.dumps(event))

        now = provider_status._timestamp("2026-10-04T12:00:00Z")
        self.assertEqual(provider_status.unavailable_reason("kimi-code", now=now), "weekly cap")

    def test_an_unparseable_timestamp_is_not_a_valid_observation(self) -> None:
        event = provider_status.observe(
            "anthropic",
            kind="inventory",
            state="available",
            source_tool="test",
            source_method="fixture",
            now=1_790_000_000,
        )
        for field in ("observed_at", "expires_at"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                provider_status.validate_event({**event, field: "yesterday"})
            # Without an offset, the same text names a different instant per host.
            with (
                self.subTest(field=field, form="no offset"),
                self.assertRaisesRegex(ValueError, field),
            ):
                provider_status.validate_event({**event, field: "2026-10-04T11:00:00"})

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
