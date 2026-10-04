from __future__ import annotations

import copy
import itertools
import json
import os
import subprocess
import tempfile
import unittest
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from unittest import mock

from agent_execution import provider_status


class ProviderStatusSyncTest(unittest.TestCase):
    NOW = 1_790_000_000

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(
            os.environ,
            {
                "AGENT_PROVIDER_STATUS_DIR": str(self.root),
                "AGENT_PROVIDER_STATUS_SALT": "test-shared-salt",
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        clock = mock.patch("agent_execution.provider_status.time.time", return_value=self.NOW)
        clock.start()
        self.addCleanup(clock.stop)

    def event(self, number: int, *, route: str = "anthropic") -> dict[str, object]:
        return {
            "schema_version": "provider-status/v2",
            "event_id": f"1790000000000-{number:032x}",
            "subject": {
                "route": route,
                "billing_pool": route,
                "credential_fingerprint": "host:fixture-host:unknown",
                "fingerprint_scope": "host",
                "host": "fixture-host",
                "os_user": "agent",
            },
            "fact": {"kind": "availability", "state": "available"},
            "observed_at": datetime.fromtimestamp(self.NOW, timezone.utc).isoformat(),
            "expires_at": datetime.fromtimestamp(self.NOW + 600, timezone.utc).isoformat(),
            "source": {"tool": "test", "method": "fixture"},
        }

    def save(self, event: dict[str, object], directory: str = "outbox") -> Path:
        path = self.root / directory / f"{event['event_id']}.json"
        provider_status._atomic_json(path, event)
        return path

    def key(self, event: dict[str, object]) -> str:
        return f"provider-status/v2/observations/fixture-host/{event['event_id']}.json"

    def routes(self) -> set[str]:
        providers = cast(list[dict[str, object]], provider_status.snapshot()["providers"])
        return {cast(str, provider["route"]) for provider in providers}

    def runner(
        self,
        remote: dict[str, str | UnicodeError | OSError],
        *,
        upload_errors: set[str] | None = None,
    ) -> provider_status.Runner:
        def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(args[:2], ["weft", "r2"])
            if args[2] == "ls":
                # weft/cmd/r2.go prints fmt.Printf("%10d  %s\n", size, key).
                text = "".join(f"{100:10d}  {key}\n" for key in remote)
                return subprocess.CompletedProcess(args, 0, stdout=text, stderr="")
            if args[2] == "cat":
                value = remote[args[3]]
                if isinstance(value, (UnicodeError, OSError)):
                    raise value
                return subprocess.CompletedProcess(args, 0, stdout=value, stderr="")
            if args[2] == "put-content":
                path = Path(args[args.index("--path") + 1])
                key = args[args.index("--key") + 1]
                if upload_errors and path.name in upload_errors:
                    return subprocess.CompletedProcess(args, 1, stdout="", stderr="upload refused")
                remote[key] = path.read_text(encoding="utf-8")
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            self.fail(f"unexpected transport command: {args!r}")

        return run

    def test_publish_failure_retry_and_replay_preserve_every_event(self) -> None:
        for failed in itertools.product((False, True), repeat=2):
            with (
                self.subTest(failed=failed),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_DIR": directory}),
            ):
                events = [
                    provider_status.observe(
                        route,
                        kind="availability",
                        state="available",
                        source_tool="test",
                        source_method="fixture",
                        host="fixture-host",
                        os_user="agent",
                    )
                    for route in ("anthropic", "openai-codex")
                ]
                paths = [Path(directory) / "outbox" / f"{e['event_id']}.json" for e in events]
                originals = [path.read_bytes() for path in paths]
                failures = {path.name for path, fail in zip(paths, failed) if fail}
                remote: dict[str, str | UnicodeError | OSError] = {}
                with (
                    mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
                    mock.patch(
                        "agent_execution.provider_status._run",
                        self.runner(remote, upload_errors=failures),
                    ),
                ):
                    diagnostics = provider_status.sync(pull=False)
                    self.assertEqual(len(diagnostics), sum(failed))
                    self.assertEqual(
                        provider_status.snapshot()["pending_observations"], sum(failed)
                    )
                    self.assertEqual(self.routes(), {"anthropic", "openai-codex"})
                    for path, original, fail in zip(paths, originals, failed):
                        retained = path if fail else Path(directory) / "events" / path.name
                        self.assertEqual(retained.read_bytes(), original)
                    self.assertEqual(len(remote), 2 - sum(failed))
                    failures.clear()
                    self.assertEqual(provider_status.sync(), [])
                    self.assertEqual(provider_status.sync(), [])
                self.assertEqual(len(remote), 2)
                self.assertEqual(provider_status.snapshot()["pending_observations"], 0)
                self.assertEqual(self.routes(), {"anthropic", "openai-codex"})
                for event in events:
                    self.assertEqual(json.loads(cast(str, remote[self.key(event)])), event)

    def test_uploaded_event_survives_cache_move_failure_and_sibling_publishes(self) -> None:
        first = self.save(self.event(1))
        second = self.save(self.event(2, route="openai-codex"))
        original = first.read_bytes()
        remote: dict[str, str | UnicodeError | OSError] = {}
        replace = Path.replace

        def fail_first(path: Path, target: Path) -> Path:
            if path == first:
                raise PermissionError("injected cache move failure")
            return replace(path, target)

        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", self.runner(remote)),
        ):
            with mock.patch.object(Path, "replace", fail_first):
                diagnostics = provider_status.sync(pull=False)
            self.assertTrue(
                any(
                    first.name in item and "injected cache move failure" in item
                    for item in diagnostics
                )
            )
            self.assertEqual(first.read_bytes(), original)
            self.assertFalse(second.exists())
            self.assertEqual(len(remote), 2)
            self.assertEqual(self.routes(), {"anthropic", "openai-codex"})
            self.assertEqual(provider_status.snapshot()["pending_observations"], 1)
            self.assertEqual(provider_status.sync(), [])
        self.assertFalse(first.exists())
        self.assertEqual(len(remote), 2)
        self.assertEqual(provider_status.snapshot()["pending_observations"], 0)

    def test_unhashable_fact_fields_do_not_block_local_or_outbox_siblings(self) -> None:
        for directory, field, malformed in itertools.product(
            ("events", "outbox"), ("kind", "state"), ([], {})
        ):
            with self.subTest(directory=directory, field=field, value=malformed):
                good = self.event(2, route="openai-codex")
                good_path = self.save(good, directory)
                bad = self.event(1)
                cast(dict[str, object], bad["fact"])[field] = malformed
                bad_path = self.save(bad, directory)
                remote: dict[str, str | UnicodeError | OSError] = {}
                try:
                    self.assertEqual(self.routes(), {"openai-codex"})
                    with (
                        mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
                        mock.patch("agent_execution.provider_status._run", self.runner(remote)),
                    ):
                        diagnostics = provider_status.sync(pull=False)
                    if directory == "outbox":
                        self.assertTrue(any(bad_path.name in item for item in diagnostics))
                        self.assertEqual(json.loads(cast(str, remote[self.key(good)])), good)
                    self.assertEqual(self.routes(), {"openai-codex"})
                finally:
                    bad_path.unlink(missing_ok=True)
                    good_path.unlink(missing_ok=True)
                    (self.root / "events" / good_path.name).unlink(missing_ok=True)

    def test_remote_unhashable_fact_fields_do_not_block_valid_siblings(self) -> None:
        for field, malformed, reverse in itertools.product(
            ("kind", "state"), ([], {}), (False, True)
        ):
            with self.subTest(field=field, value=malformed, reverse=reverse):
                bad = self.event(1)
                cast(dict[str, object], bad["fact"])[field] = malformed
                good = self.event(2, route="openai-codex")
                events = (good, bad) if reverse else (bad, good)
                remote: dict[str, str | UnicodeError | OSError] = {
                    self.key(event): json.dumps(event) for event in events
                }
                with (
                    mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
                    mock.patch("agent_execution.provider_status._run", self.runner(remote)),
                ):
                    diagnostics = provider_status.sync(push=False)
                self.assertTrue(any(self.key(bad) in item for item in diagnostics))
                self.assertEqual(self.routes(), {"openai-codex"})
                self.assertFalse((self.root / "events" / f"{bad['event_id']}.json").exists())
                (self.root / "events" / f"{good['event_id']}.json").unlink()

    def test_bounded_required_field_malformations_are_validation_errors(self) -> None:
        paths = [
            ("schema_version",),
            ("event_id",),
            ("observed_at",),
            ("expires_at",),
            ("subject",),
            ("fact",),
            ("source",),
            ("source", "tool"),
            ("fact", "kind"),
            ("fact", "state"),
            *(
                ("subject", field)
                for field in ("route", "billing_pool", "credential_fingerprint", "host", "os_user")
            ),
        ]
        for path, value in itertools.product(paths, (None, False, 17, "", [], {})):
            with self.subTest(path=path, value=value):
                event = copy.deepcopy(self.event(1))
                target = event if len(path) == 1 else cast(dict[str, object], event[path[0]])
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    provider_status.validate_event(event)

    def test_invalid_utf8_local_event_does_not_block_valid_publication(self) -> None:
        good = self.event(2, route="openai-codex")
        self.save(good)
        bad_path = self.save(self.event(1))
        temporary = bad_path.with_suffix(".tmp")
        temporary.write_bytes(b"\xff")
        temporary.replace(bad_path)
        remote: dict[str, str | UnicodeError | OSError] = {}
        self.assertEqual(self.routes(), {"openai-codex"})
        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", self.runner(remote)),
        ):
            diagnostics = provider_status.sync(pull=False)
        self.assertTrue(any(bad_path.name in item for item in diagnostics))
        self.assertEqual(bad_path.read_bytes(), b"\xff")
        self.assertEqual(json.loads(cast(str, remote[self.key(good)])), good)
        self.assertEqual(self.routes(), {"openai-codex"})

    def test_invalid_utf8_download_does_not_block_valid_remote_sibling(self) -> None:
        bad, good = self.event(1), self.event(2, route="openai-codex")
        remote: dict[str, str | UnicodeError | OSError] = {
            self.key(bad): UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
            self.key(good): json.dumps(good),
        }
        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", self.runner(remote)),
        ):
            diagnostics = provider_status.sync(push=False)
        self.assertTrue(any(self.key(bad) in item for item in diagnostics))
        self.assertEqual(self.routes(), {"openai-codex"})

    def test_invalid_utf8_listing_reports_failure_without_erasing_local_facts(self) -> None:
        self.save(self.event(1), "events")
        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch(
                "agent_execution.provider_status._run",
                side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
            ),
        ):
            diagnostics = provider_status.sync(push=False)
        self.assertTrue(diagnostics)
        self.assertEqual(self.routes(), {"anthropic"})

    def test_remote_cache_write_failure_does_not_block_valid_sibling(self) -> None:
        bad, good = self.event(1), self.event(2, route="openai-codex")
        remote: dict[str, str | UnicodeError | OSError] = {
            self.key(event): json.dumps(event) for event in (bad, good)
        }
        bad_destination = self.root / "events" / f"{bad['event_id']}.json"
        replace = Path.replace

        def fail_first(path: Path, target: Path) -> Path:
            if target == bad_destination:
                raise PermissionError("injected remote cache write failure")
            return replace(path, target)

        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", self.runner(remote)),
            mock.patch.object(Path, "replace", fail_first),
        ):
            diagnostics = provider_status.sync(push=False)
        self.assertTrue(
            any(
                self.key(bad) in item and "injected remote cache write failure" in item
                for item in diagnostics
            )
        )
        self.assertFalse(bad_destination.exists())
        self.assertEqual(self.routes(), {"openai-codex"})

    def test_remote_key_must_match_event_id_and_host(self) -> None:
        for mismatch in ("event_id", "host"):
            with self.subTest(mismatch=mismatch):
                bad, good = self.event(1), self.event(2, route="openai-codex")
                bad_key = self.key(bad)
                if mismatch == "event_id":
                    bad["event_id"] = self.event(3)["event_id"]
                else:
                    cast(dict[str, object], bad["subject"])["host"] = "different-host"
                remote: dict[str, str | UnicodeError | OSError] = {
                    bad_key: json.dumps(bad),
                    self.key(good): json.dumps(good),
                }
                with (
                    mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
                    mock.patch("agent_execution.provider_status._run", self.runner(remote)),
                ):
                    diagnostics = provider_status.sync(push=False)
                self.assertTrue(any(bad_key in item for item in diagnostics))
                self.assertEqual(self.routes(), {"openai-codex"})
                self.assertFalse((self.root / "events" / Path(bad_key).name).exists())
                (self.root / "events" / f"{good['event_id']}.json").unlink()

    def test_malformed_listing_is_diagnosed_without_discarding_valid_rows(self) -> None:
        good = self.event(1)
        run = self.runner({self.key(good): json.dumps(good)})

        def malformed_listing(
            args: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            result = run(args, **kwargs)
            if args[2] == "ls":
                result.stdout = "not a size-and-key row\n" + result.stdout
            return result

        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", malformed_listing),
        ):
            diagnostics = provider_status.sync(push=False)
        self.assertTrue(diagnostics)
        self.assertEqual(self.routes(), {"anthropic"})

    def test_outbox_filename_must_match_event_identity(self) -> None:
        bad, good = self.event(1), self.event(2, route="openai-codex")
        bad_path = self.save(bad)
        bad["event_id"] = self.event(3)["event_id"]
        provider_status._atomic_json(bad_path, bad)
        original = bad_path.read_bytes()
        self.save(good)
        remote: dict[str, str | UnicodeError | OSError] = {}
        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", self.runner(remote)),
        ):
            diagnostics = provider_status.sync(pull=False)
        self.assertTrue(any(bad_path.name in item for item in diagnostics))
        self.assertEqual(bad_path.read_bytes(), original)
        self.assertNotIn(self.key(bad), remote)
        self.assertEqual(json.loads(cast(str, remote[self.key(good)])), good)

    def test_uploaded_event_survives_cache_directory_failure_and_retry(self) -> None:
        first = self.save(self.event(1))
        second = self.save(self.event(2, route="openai-codex"))
        original = first.read_bytes()
        remote: dict[str, str | UnicodeError | OSError] = {}
        mkdir = Path.mkdir
        failed = False

        def fail_once(
            path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
        ) -> None:
            nonlocal failed
            if path == self.root / "events" and not failed:
                failed = True
                raise PermissionError("injected cache directory failure")
            mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", self.runner(remote)),
        ):
            with mock.patch.object(Path, "mkdir", fail_once):
                diagnostics = provider_status.sync(pull=False)
            self.assertTrue(
                any(
                    first.name in item and "injected cache directory failure" in item
                    for item in diagnostics
                )
            )
            self.assertEqual(first.read_bytes(), original)
            self.assertFalse(second.exists())
            self.assertEqual(len(remote), 2)
            self.assertEqual(self.routes(), {"anthropic", "openai-codex"})
            self.assertEqual(provider_status.sync(), [])
        self.assertFalse(first.exists())
        self.assertEqual(provider_status.snapshot()["pending_observations"], 0)

    def test_snapshot_keeps_event_moved_during_outbox_enumeration(self) -> None:
        pending = self.save(self.event(1))
        cached = self.root / "events" / pending.name
        cached.parent.mkdir()
        glob = Path.glob

        def move_before_listing(path: Path, pattern: str) -> Iterator[Path]:
            if path == pending.parent and pending.exists():
                pending.replace(cached)
            return glob(path, pattern)

        with mock.patch.object(Path, "glob", move_before_listing):
            snapshot = provider_status.snapshot()
        providers = cast(list[dict[str, object]], snapshot["providers"])
        self.assertEqual([item["route"] for item in providers], ["anthropic"])
        self.assertEqual(snapshot["diagnostics"], [])

    def test_snapshot_keeps_event_moved_after_outbox_enumeration(self) -> None:
        pending = self.save(self.event(1))
        cached = self.root / "events" / pending.name
        cached.parent.mkdir()
        read_text = Path.read_text

        def move_before_read(
            path: Path, encoding: str | None = None, errors: str | None = None
        ) -> str:
            if path == pending and pending.exists():
                pending.replace(cached)
            return read_text(path, encoding=encoding, errors=errors)

        with mock.patch.object(Path, "read_text", move_before_read):
            snapshot = provider_status.snapshot()
        providers = cast(list[dict[str, object]], snapshot["providers"])
        self.assertEqual([item["route"] for item in providers], ["anthropic"])
        self.assertEqual(snapshot["diagnostics"], [])


if __name__ == "__main__":
    unittest.main()
