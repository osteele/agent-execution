from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from unittest import mock

from agent_execution import provider_status


class ProviderStatusIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(
            os.environ,
            {
                "AGENT_PROVIDER_STATUS_DIR": str(self.root),
                "AGENT_PROVIDER_STATUS_SALT": "",
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def test_concurrent_salt_creator_keeps_the_first_published_salt(self) -> None:
        salt_path = self.root / "fingerprint-salt"
        winning_salt = b"a" * 32

        def publish_competing_salt(size: int) -> bytes:
            self.assertEqual(size, 32)
            salt_path.write_bytes(winning_salt)
            return b"b" * 32

        with mock.patch(
            "agent_execution.provider_status.os.urandom", side_effect=publish_competing_salt
        ):
            fingerprint, scope = provider_status.credential_fingerprint(
                "email:alice@example.test", host="studio"
            )

        digest = hmac.new(winning_salt, b"email:alice@example.test", hashlib.sha256).hexdigest()
        self.assertEqual(salt_path.read_bytes(), winning_salt)
        self.assertEqual(fingerprint, f"hmac-sha256:{digest}")
        self.assertEqual(scope, "host")
        self.assertEqual(list(self.root.glob(".fingerprint-salt.*")), [])

    def test_unreadable_existing_salt_is_not_replaced(self) -> None:
        salt_path = self.root / "fingerprint-salt"
        original = b"original-host-salt"
        salt_path.write_bytes(original)
        with (
            mock.patch.object(Path, "read_bytes", side_effect=PermissionError("denied")),
            self.assertRaises(PermissionError),
        ):
            provider_status.credential_fingerprint("accountId:one", host="studio")
        self.assertEqual(salt_path.read_bytes(), original)

    def test_host_local_salts_keep_independent_hosts_separate(self) -> None:
        fingerprints = []
        for host in ("studio", "laptop"):
            with (
                self.subTest(host=host),
                mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_DIR": str(self.root / host)}),
            ):
                fingerprint = provider_status.credential_fingerprint("accountId:one", host=host)
                self.assertEqual(fingerprint[1], "host")
                self.assertEqual(
                    provider_status.credential_fingerprint("accountId:one", host=host), fingerprint
                )
                self.assertEqual(
                    (self.root / host / "fingerprint-salt").stat().st_mode & 0o777, 0o600
                )
                fingerprints.append(fingerprint[0])
        self.assertNotEqual(*fingerprints)

    def test_shared_salt_matches_only_known_identities_across_hosts(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_SALT": "shared-fixture-salt"}):
            first = provider_status.credential_fingerprint("accountId:one", host="studio")
            second = provider_status.credential_fingerprint("accountId:one", host="laptop")
            different = provider_status.credential_fingerprint("accountId:two", host="studio")
            self.assertEqual(first, second)
            self.assertEqual(first[1], "global")
            self.assertNotEqual(first, different)
            self.assertEqual(
                provider_status.credential_fingerprint(None, host="studio"),
                ("host:studio:unknown", "host"),
            )
        self.assertFalse((self.root / "fingerprint-salt").exists())

    def test_omp_exact_identities_preserve_all_account_collections(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_SALT": "shared-fixture-salt"}):
            for field in ("email", "accountId", "projectId", "orgId"):
                for collection in ("reports", "accountsWithoutUsage", "disabledCredentials"):
                    with self.subTest(field=field, collection=collection):
                        entry: dict[str, object] = {"provider": "anthropic"}
                        identity = "alice@example.test" if field == "email" else "account-one"
                        if collection == "reports":
                            entry["metadata"] = {field: identity}
                        else:
                            entry[field] = identity
                        payload: dict[str, object] = {"reports": []}
                        payload[collection] = [entry]
                        fingerprints = []
                        for host in ("studio", "laptop"):
                            events = provider_status.observe_omp_usage(
                                payload, host=host, now=1_790_000_000
                            )
                            self.assertEqual(len(events), 1)
                            subject = cast(dict[str, object], events[0]["subject"])
                            fingerprints.append(subject["credential_fingerprint"])
                            self.assertEqual(subject["fingerprint_scope"], "global")
                            self.assertNotIn(identity, json.dumps(events))
                        self.assertEqual(*fingerprints)
                        self.assertEqual(
                            fingerprints[0],
                            provider_status.credential_fingerprint(
                                f"{field}:{identity}", host="studio"
                            )[0],
                        )

    def test_probe_uses_exact_source_and_returns_only_private_account_partitions(self) -> None:
        identities = ("alice@example.test", "alina@example.test")
        reports: list[dict[str, object]] = [
            {
                "provider": "openai-codex",
                "metadata": {"email": identities[0], "allowed": False},
                "limits": [{"id": "openai-codex:5h", "label": "5 Hour"}],
            },
            {
                "provider": "openai-codex",
                "metadata": {"email": identities[1], "allowed": True},
                "limits": [{"id": "openai-codex:5h", "label": "5 Hour"}],
            },
        ]
        inventories = iter((reports, reports[:1]))

        def usage(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
            inventory = next(inventories)
            if "--redact" in command:
                # OMP masks depend on the inventory, not just the exact identity.
                masks = ("al*c*", "al*n*") if len(inventory) == 2 else ("al*",)
                inventory = [
                    {
                        **report,
                        "metadata": {
                            **cast(dict[str, object], report["metadata"]),
                            "email": mask,
                        },
                    }
                    for report, mask in zip(inventory, masks, strict=True)
                ]
            return subprocess.CompletedProcess(command, 0, json.dumps({"reports": inventory}), "")

        with (
            mock.patch("agent_execution.provider_status._command", side_effect=usage),
            mock.patch("agent_execution.provider_status.time.time", return_value=1_790_000_000),
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_SALT": "shared-fixture-salt"}),
        ):
            events = provider_status.probe_omp(host="studio")
            snapshot = provider_status.snapshot(host="studio", now=1_790_000_001)
            laptop_events = provider_status.probe_omp(host="laptop")
            expected = provider_status.credential_fingerprint(
                f"email:{identities[0]}", host="studio"
            )[0]

        self.assertIsInstance(events, list)
        self.assertIsInstance(laptop_events, list)
        self.assertEqual(len(events), 2)
        self.assertEqual(len(laptop_events), 1)
        providers = cast(list[dict[str, object]], snapshot["providers"])
        self.assertEqual(len(providers), 2)
        self.assertEqual(
            {provider["state"] for provider in providers}, {"available", "unavailable"}
        )
        subjects = [cast(dict[str, object], event["subject"]) for event in events]
        self.assertEqual(len({subject["credential_fingerprint"] for subject in subjects}), 2)
        laptop_subject = cast(dict[str, object], laptop_events[0]["subject"])
        self.assertEqual(subjects[0]["credential_fingerprint"], expected)
        self.assertEqual(laptop_subject["credential_fingerprint"], expected)
        for subject in [*subjects, laptop_subject]:
            self.assertEqual(subject["fingerprint_scope"], "global")
        for event in [*events, *laptop_events]:
            self.assertEqual(event["source"], {"tool": "omp", "method": "usage --json"})
        serialized = json.dumps({"events": [*events, *laptop_events], "snapshot": snapshot})
        persisted = "\n".join(path.read_text() for path in self.root.rglob("*.json"))
        for identity in identities:
            self.assertNotIn(identity, serialized)
            self.assertNotIn(identity, persisted)

    def test_documented_signature_forms_keep_their_classification(self) -> None:
        cases = (
            ("您已达到每周/每月使用上限", ("quota", "weekly-or-monthly", 86400, None)),
            ("You've reached your 5-hour usage limit", ("quota", "session", 18000, None)),
            ("monthly usage limit exceeded", ("quota", "monthly", 604800, None)),
            ("quota exhausted for the MONTHLY plan", ("quota", "monthly", 604800, None)),
            ("provider.auth_error: 403", ("quota", "session", 18000, None)),
            ("OAuth request for access token failed", ("auth", None, 86400, None)),
            ("Cannot connect to API", ("network", None, 300, None)),
            ("unrecognized failure", ("unknown", None, 600, None)),
        )
        for signature, expected in cases:
            with self.subTest(signature=signature):
                self.assertEqual(provider_status.classify(signature), expected)

    def test_provider_reset_date_accepts_both_documented_separators(self) -> None:
        now = 1_790_000_000
        reset = datetime(2026, 10, 6, 12).astimezone().timestamp()
        for separator in (" ", "T"):
            reset_at = f"2026-10-06{separator}12:00:00"
            with (
                self.subTest(separator=separator),
                mock.patch("agent_execution.provider_status.time.time", return_value=now),
            ):
                self.assertEqual(
                    provider_status.classify(
                        f"您已达到每周/每月使用上限; 限额将在 {reset_at} 重置"
                    ),
                    ("quota", "weekly-or-monthly", max(3600, int(reset - now)), reset_at),
                )

    def test_signature_keywords_do_not_match_larger_words_or_status_codes(self) -> None:
        for signature in (
            "bimonthly quota exhausted",
            "monthly recap",
            "limitless monthly plan",
            "provider.auth_error: 4030",
        ):
            with self.subTest(signature=signature):
                self.assertEqual(provider_status.classify(signature), ("unknown", None, 600, None))

    def test_specific_quota_signatures_keep_precedence_over_generic_signatures(self) -> None:
        self.assertEqual(
            provider_status.classify("You've reached your 5-hour usage limit; monthly quota"),
            ("quota", "session", 18000, None),
        )
        self.assertEqual(
            provider_status.classify("monthly quota exhausted; provider.auth_error: 403"),
            ("quota", "monthly", 604800, None),
        )

    def test_invalid_provider_reset_date_uses_existing_cooldown(self) -> None:
        self.assertEqual(
            provider_status.classify("限额将在 2026-99-99 25:61:61 重置")[:3],
            ("quota", "weekly-or-monthly", 86400),
        )

    def test_pathological_hour_counts_still_produce_serializable_refusals(self) -> None:
        for hours in ("0", "9" * 5000, "99999999999999999999"):
            with (
                self.subTest(hours=hours[:20]),
                mock.patch("agent_execution.provider_status.publish_async"),
                mock.patch("agent_execution.provider_status.time.time", return_value=1_790_000_000),
            ):
                event = provider_status.record_refusal(
                    "kimi-code", f"You've reached your {hours}-hour usage limit"
                )
                fact = cast(dict[str, object], event["fact"])
                self.assertEqual(fact["kind"], "quota")
                expires_at = cast(str, event["expires_at"])
                expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                self.assertGreater(expiry, datetime.fromtimestamp(1_790_000_000, timezone.utc))


if __name__ == "__main__":
    unittest.main()
