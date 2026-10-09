from __future__ import annotations

import builtins
import contextlib
import fcntl
import hashlib
import io
import json
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import cast
from unittest import mock

from agent_execution import provider_status

NOW = 1_790_000_000
RETENTION = 30 * 24 * 60 * 60
_REPLAY_SEEDS = (7, 41, 902)
Event = dict[str, object]


def _instant(value: object) -> float:
    return datetime.fromisoformat(cast(str, value).replace("Z", "+00:00")).timestamp()


def _event(
    number: int,
    *,
    route: str = "anthropic",
    kind: str = "quota",
    state: str = "unavailable",
    observed: int = NOW,
    ttl: int = 600,
    host: str = "host-a",
    user: str = "agent",
    fingerprint: str = "hmac-sha256:fixture-a",
    pool: str | None = None,
    offset_hours: int = 0,
) -> Event:
    zone = timezone(timedelta(hours=offset_hours))
    return {
        "schema_version": "provider-status/v2",
        "event_id": f"1790000000000-{number:032x}",
        "subject": {
            "route": route,
            "billing_pool": pool or route,
            "credential_fingerprint": fingerprint,
            "fingerprint_scope": "shared",
            "host": host,
            "os_user": user,
        },
        "fact": {"kind": kind, "state": state, "reason": f"{route} limited"},
        "observed_at": datetime.fromtimestamp(observed, zone).isoformat(),
        "expires_at": datetime.fromtimestamp(observed + ttl, zone).isoformat(),
        "source": {"tool": "test", "method": "projection-fixture"},
    }


def _publish_bytes(path: Path, contents: bytes) -> None:
    """Legacy writers publish immutable files without acquiring projection.lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish(root: Path, event: Event, directory: str = "events") -> Path:
    path = root / directory / f"{event['event_id']}.json"
    _publish_bytes(path, json.dumps(event).encode())
    return path


def _reference(
    events: list[Event], now: int, host: str | None
) -> tuple[list[Event], dict[str, str]]:
    """Independent full-history oracle; never calls registry replay/render helpers."""
    latest: dict[tuple[str, ...], Event] = {}
    for event in events:
        subject = cast(Event, event["subject"])
        fact = cast(Event, event["fact"])
        if _instant(event["observed_at"]) < now - RETENTION:
            continue
        if host is not None and subject["host"] != host:
            continue
        key = tuple(
            cast(str, subject[field])
            for field in ("route", "billing_pool", "credential_fingerprint", "host", "os_user")
        ) + (cast(str, fact["kind"]),)
        previous = latest.get(key)
        rank = (_instant(event["observed_at"]), cast(str, event["event_id"]))
        if previous is None or rank > (
            _instant(previous["observed_at"]),
            cast(str, previous["event_id"]),
        ):
            latest[key] = event
    successes = {
        key[:-1]: _instant(event["observed_at"])
        for key, event in latest.items()
        if cast(Event, event["fact"])["kind"] == "availability"
        and cast(Event, event["fact"])["state"] == "available"
    }
    providers: list[Event] = []
    unavailable: dict[str, str] = {}
    for key, event in latest.items():
        subject = cast(Event, event["subject"])
        fact = cast(Event, event["fact"])
        if (
            fact["state"] == "unavailable"
            and key[:-1] in successes
            and successes[key[:-1]] > _instant(event["observed_at"])
        ):
            continue
        stale = _instant(event["expires_at"]) <= now
        provider: Event = {
            **subject,
            "kind": fact["kind"],
            "state": fact["state"],
            "observed_at": event["observed_at"],
            "expires_at": event["expires_at"],
            "stale": stale,
            "source": event["source"],
        }
        for field in ("condition", "window", "reset_at", "reason", "quota", "quota_observable"):
            if field in fact:
                provider[field] = fact[field]
        providers.append(provider)
        if not stale and fact["state"] == "unavailable" and fact["kind"] != "transport":
            unavailable[cast(str, subject["route"])] = cast(str, fact["reason"])
    return providers, unavailable


@contextlib.contextmanager
def _history_io(root: Path) -> Iterator[dict[str, list[Path]]]:
    """Count real file-system boundaries, not calls to implementation replay helpers."""
    directories = {root / "events", root / "outbox"}
    counts: dict[str, list[Path]] = {"reads": [], "stats": [], "scans": []}

    def filesystem_path(value: object) -> Path | None:
        if isinstance(value, (str, bytes, os.PathLike)):
            return Path(os.fsdecode(value))
        return None

    def opening(original: Callable[..., object]) -> Callable[..., object]:
        def counted(path: object, *args: object, **kwargs: object) -> object:
            file = filesystem_path(path)
            mode = kwargs.get("mode", args[0] if args else "r")
            reading = (
                mode & os.O_ACCMODE != os.O_WRONLY
                if isinstance(mode, int)
                else "r" in cast(str, mode) or "+" in cast(str, mode)
            )
            if (
                file is not None
                and file.parent in directories
                and file.suffix == ".json"
                and reading
            ):
                counts["reads"].append(file)
            return original(path, *args, **kwargs)

        return counted

    def checking(original: Callable[..., object], *, scan: bool) -> Callable[..., object]:
        def counted(path: object, *args: object, **kwargs: object) -> object:
            file = filesystem_path(path)
            if file is not None:
                if scan and file in directories:
                    counts["scans"].append(file)
                elif not scan and file.parent in directories and file.suffix == ".json":
                    counts["stats"].append(file)
            return original(path, *args, **kwargs)

        return counted

    class CountedEntry:
        def __init__(self, entry: os.DirEntry[str]) -> None:
            self.entry = entry

        def __getattr__(self, name: str) -> object:
            return getattr(self.entry, name)

        def __fspath__(self) -> str:
            return self.entry.path

        def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
            file = Path(self.entry.path)
            if file.parent in directories and file.suffix == ".json":
                counts["stats"].append(file)
            return self.entry.stat(follow_symlinks=follow_symlinks)

    scandir = os.scandir

    def scanning(path: str | os.PathLike[str]) -> object:
        file = filesystem_path(path)
        if file in directories:
            counts["scans"].append(file)
        entries = scandir(path)

        class CountedScan:
            def __iter__(self) -> CountedScan:
                return self

            def __next__(self) -> CountedEntry:
                return CountedEntry(next(entries))

            def __enter__(self) -> Iterator[CountedEntry]:
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

            def close(self) -> None:
                entries.close()

        return CountedScan()

    with (
        mock.patch("builtins.open", opening(builtins.open)),
        mock.patch("io.open", opening(io.open)),
        mock.patch("os.open", opening(os.open)),
        mock.patch("os.stat", checking(os.stat, scan=False)),
        mock.patch("os.lstat", checking(os.lstat, scan=False)),
        mock.patch("os.scandir", scanning),
        mock.patch("os.listdir", checking(os.listdir, scan=True)),
        mock.patch.object(Path, "glob", checking(Path.glob, scan=True)),
    ):
        yield counts


def _concurrent_writer(root: str, writer: int, barrier: Barrier) -> None:
    os.environ.update(
        {
            "AGENT_PROVIDER_STATUS_DIR": root,
            "AGENT_PROVIDER_STATUS_SALT": "projection-test-salt",
            "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
        }
    )
    for number in range(8):
        barrier.wait(timeout=20)
        provider_status.observe(
            f"concurrent-{writer}-{number}",
            kind="availability",
            state="available",
            source_tool="test",
            source_method="concurrent-writer",
            host=f"writer-{writer}",
            os_user="isolated-test",
            now=NOW + number,
        )


class ProviderStatusProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(
            os.environ,
            {
                "AGENT_PROVIDER_STATUS_DIR": str(self.root),
                "AGENT_PROVIDER_STATUS_SALT": "projection-test-salt",
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        clock = mock.patch("agent_execution.provider_status.time.time", return_value=NOW)
        clock.start()
        self.addCleanup(clock.stop)

    def assert_projection(
        self, events: list[Event], *, now: int = NOW, host: str | None = None
    ) -> Event:
        snapshot = provider_status.snapshot(now=now, host=host)
        providers, unavailable = _reference(events, now, host)
        self.assertCountEqual(cast(list[Event], snapshot["providers"]), providers)
        self.assertEqual(snapshot["unavailable_routes"], unavailable)
        self.assertEqual(snapshot["diagnostics"], [])
        if host is None:
            self.assertEqual(
                provider_status.quota_blocked_routes(now=now),
                frozenset(
                    cast(str, provider["route"])
                    for provider in providers
                    if provider["kind"] == "quota"
                    and provider["state"] == "unavailable"
                    and not provider["stale"]
                ),
            )
        return snapshot

    def assert_no_history_io(self, counts: dict[str, list[Path]]) -> None:
        self.assertEqual(counts, {"reads": [], "stats": [], "scans": []})

    def assert_inventory_only_history_io(
        self,
        counts: dict[str, list[Path]],
        *,
        touched: list[Event],
        mutations: int = 1,
        listing_scans: int = 0,
    ) -> None:
        names = {f"{event['event_id']}.json" for event in touched}
        for operation in ("reads", "stats"):
            self.assertEqual(
                [path for path in counts[operation] if path.name not in names],
                [],
                f"mutation performed historical event {operation}",
            )
        # Baseline/post-write inventories may retry, but may not rescan per event.
        self.assertLessEqual(len(counts["scans"]), 12 * mutations + listing_scans)

    def assert_reconciled_projection(self, events: list[Event], *, pending: int) -> None:
        self.assertEqual(self.assert_projection(events)["pending_observations"], pending)
        with _history_io(self.root) as counts:
            self.assertEqual(self.assert_projection(events)["pending_observations"], pending)
        self.assert_no_history_io(counts)

    def test_warm_reads_skip_history_and_writes_only_inventory_it(self) -> None:
        sizes: list[int] = []
        for count in (24, 768):
            root = self.root / str(count)
            with (
                self.subTest(history=count),
                mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_DIR": str(root)}),
            ):
                history = [_event(number, observed=NOW - count + number) for number in range(count)]
                for number, event in enumerate(history):
                    _publish(root, event, "events" if number % 2 else "outbox")
                with _history_io(root) as cold_counts:
                    self.assertEqual(
                        self.assert_projection(history)["pending_observations"], count // 2
                    )
                self.assertGreaterEqual(len(cold_counts["reads"]), count)
                self.assertIn(root / "outbox", cold_counts["scans"])
                self.assertIn(root / "events", cold_counts["scans"])
                projection = root / "projection.json"
                self.assertTrue(projection.is_file())
                sizes.append(projection.stat().st_size)
                with _history_io(root) as read_counts:
                    for _ in range(3):
                        warm = self.assert_projection(history)
                        self.assertEqual(warm["pending_observations"], count // 2)
                self.assert_no_history_io(read_counts)
                with _history_io(root) as update_counts:
                    added = provider_status.observe(
                        "anthropic",
                        kind="availability",
                        state="available",
                        source_tool="test",
                        source_method="warm-update",
                        host="host-a",
                        os_user="agent",
                        now=NOW,
                    )
                    current = self.assert_projection([*history, added])
                    self.assertEqual(current["pending_observations"], count // 2 + 1)
                self.assert_inventory_only_history_io(update_counts, touched=[added])
        self.assertLessEqual(sizes[1], sizes[0] + 1024, "projection retained historical events")

    def test_warm_out_of_order_observations_preserve_latest_raw_kind(self) -> None:
        arrived: list[Event] = []
        self.assert_projection([])
        for observed, kind, state in (
            (NOW + 10, "quota", "unavailable"),
            (NOW - 10, "quota", "available"),
            (NOW, "availability", "available"),
            (NOW + 20, "availability", "available"),
            (NOW + 15, "quota", "unavailable"),
            (NOW + 30, "quota", "unavailable"),
            (NOW + 25, "availability", "unknown"),
            (NOW + 15, "availability", "available"),
        ):
            with self.subTest(observed=observed, kind=kind), _history_io(self.root) as counts:
                event = provider_status.observe(
                    "anthropic",
                    kind=kind,
                    state=state,
                    source_tool="test",
                    source_method="out-of-order-warm-writer",
                    detail={"reason": "anthropic limited"},
                    host="host-a",
                    os_user="agent",
                    now=observed,
                )
                arrived.append(event)
                result = self.assert_projection(arrived)
                self.assertEqual(result["pending_observations"], len(arrived))
            self.assert_inventory_only_history_io(counts, touched=[event])

    def test_seeded_replay_matches_independent_model_across_queries_and_arrival_order(self) -> None:
        for seed in _REPLAY_SEEDS:
            root = self.root / str(seed)
            randomizer = random.Random(seed)
            events = [
                _event(
                    number,
                    route=randomizer.choice(("anthropic", "openai-codex", "kimi-code")),
                    kind=randomizer.choice(
                        ("availability", "quota", "authentication", "transport", "inventory")
                    ),
                    state=randomizer.choice(("available", "unavailable", "unknown")),
                    observed=NOW + randomizer.choice((-RETENTION - 1, -180, -30, 0, 30)),
                    ttl=randomizer.choice((10, 120, 900)),
                    host=randomizer.choice(("host-a", "host-b")),
                    user=randomizer.choice(("agent", "operator")),
                    fingerprint=randomizer.choice(
                        ("hmac-sha256:fixture-a", "hmac-sha256:fixture-b")
                    ),
                    pool=randomizer.choice(("pool-a", "pool-b")),
                    offset_hours=randomizer.choice((0, 8, -4)),
                )
                for number in range(72)
            ]
            randomizer.shuffle(events)
            with (
                self.subTest(seed=seed),
                mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_DIR": str(root)}),
            ):
                arrived: list[Event] = []
                for number, event in enumerate(events):
                    _publish(root, event, "outbox" if number % 3 == 0 else "events")
                    arrived.append(event)
                    if number % 9 == 8:
                        self.assert_projection(arrived, host="host-a")
                for now in (NOW, NOW + 120, NOW + RETENTION + 60, NOW - 60):
                    for host in ("host-b", None, "missing-host", "host-a"):
                        with self.subTest(now=now, host=host):
                            self.assert_projection(events, now=now, host=host)
                (root / "projection.json").unlink()
                # A cold read beyond retention must not discard the raw facts from cache.
                self.assert_projection(events, now=NOW + RETENTION + 60)
                with _history_io(root) as counts:
                    self.assert_projection(events, now=NOW)
                self.assert_no_history_io(counts)

    def test_equal_instant_latest_kind_uses_event_id_not_arrival_or_timestamp_spelling(
        self,
    ) -> None:
        lower = _event(1, state="available", offset_hours=8)
        higher = _event(2, state="unavailable")
        for order in ((lower, higher), (higher, lower)):
            root = self.root / cast(str, order[0]["event_id"])
            with mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_DIR": str(root)}):
                _publish(root, order[0], "outbox")
                self.assert_projection([order[0]])
                _publish(root, order[1])
                value = self.assert_projection(list(order))
                self.assertEqual(value["unavailable_routes"], {"anthropic": "anthropic limited"})
                with _history_io(root) as counts:
                    self.assert_projection(list(order))
                self.assert_no_history_io(counts)

    def test_ttl_retention_and_success_are_recomputed_from_raw_latest_facts(self) -> None:
        events = [
            _event(1, observed=NOW - 20, ttl=900),
            _event(2, observed=NOW - 10, state="available", ttl=5),
            _event(3, route="kimi-code", observed=NOW - 20),
            _event(
                4,
                route="kimi-code",
                kind="availability",
                state="available",
                observed=NOW - 10,
                ttl=5,
            ),
            _event(5, route="openai-codex", observed=NOW - RETENTION, ttl=RETENTION + 30),
            _event(6, route="transport-only", kind="transport"),
            _event(7, route="same-instant"),
            _event(8, route="same-instant", kind="availability", state="available"),
        ]
        for event in events:
            _publish(self.root, event)
        self.assert_projection(events)
        with _history_io(self.root) as counts:
            for now in (NOW - 15, NOW, NOW + 1, NOW + 600, NOW + RETENTION + 1, NOW):
                value = self.assert_projection(events, now=now)
                if now == NOW:
                    self.assertNotIn("anthropic", cast(Event, value["unavailable_routes"]))
                    self.assertNotIn("kimi-code", cast(Event, value["unavailable_routes"]))
                    self.assertNotIn("transport-only", cast(Event, value["unavailable_routes"]))
                    self.assertIn("same-instant", cast(Event, value["unavailable_routes"]))
                    self.assertIn("openai-codex", cast(Event, value["unavailable_routes"]))
        self.assert_no_history_io(counts)

    def test_missing_or_corrupt_projection_recovers_authoritative_events(self) -> None:
        events = [_event(1), _event(2, route="openai-codex", kind="inventory", state="available")]
        for event in events:
            _publish(self.root, event, "outbox")
        self.assert_projection(events)
        projection = self.root / "projection.json"
        for corruption in (
            "missing",
            "truncated",
            "utf8",
            "wrong-shape",
            "version",
            "checked-content",
        ):
            with self.subTest(corruption=corruption):
                if corruption == "missing":
                    projection.unlink()
                else:
                    damaged = {
                        "truncated": b"{",
                        "utf8": b"\xff",
                        "wrong-shape": b"[]",
                        "version": b'{"schema_version":"provider-status-projection/unsupported"}',
                    }.get(corruption)
                    if damaged is None:
                        original = projection.read_bytes()
                        self.assertIn(b"anthropic limited", original)
                        damaged = original.replace(b"anthropic limited", b"counterfeit reason")
                    _publish_bytes(projection, damaged)
                with _history_io(self.root) as recovery_io:
                    recovered = provider_status.snapshot(now=NOW)
                expected, unavailable = _reference(events, NOW, None)
                self.assertCountEqual(cast(list[Event], recovered["providers"]), expected)
                self.assertEqual(recovered["unavailable_routes"], unavailable)
                self.assertEqual(recovered["pending_observations"], 2)
                self.assertGreaterEqual(len(recovery_io["reads"]), len(events))
                self.assertTrue(projection.is_file())
                with _history_io(self.root) as warm_io:
                    self.assert_projection(events)
                self.assert_no_history_io(warm_io)

    def test_projection_publish_failure_does_not_lose_committed_observation(self) -> None:
        previous = _event(1, route="anthropic", kind="inventory", state="available")
        _publish(self.root, previous)
        self.assert_projection([previous])
        replace = Path.replace
        failures: list[Path] = []

        def deny_projection(path: Path, target: Path) -> Path:
            if target == self.root / "projection.json":
                failures.append(path)
                raise PermissionError("injected projection publication failure")
            return replace(path, target)

        with mock.patch.object(Path, "replace", deny_projection):
            try:
                provider_status.observe(
                    "openai-codex",
                    kind="quota",
                    state="unavailable",
                    source_tool="test",
                    source_method="committed-before-cache-failure",
                    detail={"reason": "openai-codex limited"},
                    host="host-a",
                    os_user="agent",
                    now=NOW,
                )
            except PermissionError as error:
                self.assertIn("injected projection publication failure", str(error))
        self.assertTrue(failures, "failure injection did not reach the projection publisher")
        committed = [
            cast(Event, json.loads(path.read_text()))
            for path in (self.root / "outbox").glob("*.json")
        ]
        self.assertEqual(len(committed), 1)
        self.assertEqual(cast(Event, committed[0]["subject"])["route"], "openai-codex")
        with _history_io(self.root) as recovery_io:
            recovered = self.assert_projection([previous, *committed])
        self.assertEqual(recovered["pending_observations"], 1)
        self.assertGreaterEqual(len(recovery_io["reads"]), 2)
        with _history_io(self.root) as warm_io:
            self.assert_projection([previous, *committed])
        self.assert_no_history_io(warm_io)

    def test_interruption_after_event_rename_cannot_bless_an_incomplete_projection(self) -> None:
        previous = _event(1, kind="inventory", state="available")
        _publish(self.root, previous, "outbox")
        self.assert_projection([previous])
        replace = Path.replace
        stat = os.stat
        outbox_status = (self.root / "outbox").stat()

        def coarse_stat(
            path: str | os.PathLike[str],
            *,
            dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> os.stat_result:
            # A filesystem clock tick can cover both adjacent directory changes.
            if path == self.root / "outbox":
                return outbox_status
            return stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

        def fail_after_commit(path: Path, target: Path) -> Path:
            result = replace(path, target)
            if target.parent == self.root / "outbox":
                raise PermissionError("injected interruption after event rename")
            return result

        with (
            mock.patch.object(Path, "replace", fail_after_commit),
            self.assertRaisesRegex(PermissionError, "interruption after event rename"),
        ):
            provider_status.observe(
                "openai-codex",
                kind="quota",
                state="unavailable",
                source_tool="test",
                source_method="interrupted-publication",
                detail={"reason": "openai-codex limited"},
                now=NOW,
            )
        with mock.patch("os.stat", coarse_stat):
            committed = [
                cast(Event, json.loads(path.read_text()))
                for path in (self.root / "outbox").glob("*.json")
            ]
            self.assertEqual(len(committed), 2)
            recovered = self.assert_projection(committed)
            self.assertEqual(recovered["pending_observations"], 2)

    def test_unwritable_projection_returns_real_events_and_transient_diagnostics(self) -> None:
        event = _event(1)
        _publish(self.root, event)
        replace = Path.replace

        def deny_projection(path: Path, target: Path) -> Path:
            if target == self.root / "projection.json":
                raise PermissionError("injected read-only projection")
            return replace(path, target)

        with mock.patch.object(Path, "replace", deny_projection):
            for _ in range(2):
                result = provider_status.snapshot(now=NOW)
                self.assertCountEqual(
                    cast(list[Event], result["providers"]), _reference([event], NOW, None)[0]
                )
                self.assertEqual(result["unavailable_routes"], {"anthropic": "anthropic limited"})
                self.assertEqual(result["pending_observations"], 0)
                self.assertTrue(
                    any(
                        "injected read-only projection" in item
                        for item in cast(list[str], result["diagnostics"])
                    )
                )
        self.assert_projection([event])
        with _history_io(self.root) as counts:
            self.assert_projection([event])
        self.assert_no_history_io(counts)

    def test_unwritable_lock_does_not_hide_real_event_projection(self) -> None:
        event = _event(1)
        _publish(self.root, event)
        open_file = os.open
        denied: list[Path] = []

        def deny_lock(
            path: str | os.PathLike[str],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if Path(path) == self.root / "projection.lock":
                denied.append(Path(path))
                raise PermissionError("injected read-only registry")
            return open_file(path, flags, mode, dir_fd=dir_fd)

        with mock.patch("os.open", deny_lock):
            result = provider_status.snapshot(now=NOW)
            with self.assertRaisesRegex(OSError, "injected read-only registry"):
                provider_status.observe(
                    "openai-codex",
                    kind="availability",
                    state="available",
                    source_tool="test",
                    source_method="lock-denied",
                    now=NOW,
                )
        self.assertTrue(denied)
        self.assertCountEqual(
            cast(list[Event], result["providers"]), _reference([event], NOW, None)[0]
        )
        self.assertEqual(result["unavailable_routes"], {"anthropic": "anthropic limited"})
        self.assertTrue(
            any(
                "injected read-only registry" in item
                for item in cast(list[str], result["diagnostics"])
            )
        )
        self.assert_projection([event])

    def test_contended_registry_refuses_writes_without_hiding_committed_facts(self) -> None:
        event = _event(1)
        _publish(self.root, event)
        self.assert_projection([event])
        script = f"""
import json
from agent_execution import provider_status

provider_status._LOCK_TIMEOUT_SECONDS = 0.05
try:
    provider_status.observe(
        "openai-codex", kind="availability", state="available",
        source_tool="test", source_method="contended-writer", now={NOW},
    )
except OSError:
    print(json.dumps(provider_status.snapshot(now={NOW})))
else:
    raise AssertionError("a contended writer published without owning the lock")
"""
        with (self.root / "projection.lock").open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        result = json.loads(completed.stdout)
        self.assertEqual(result["unavailable_routes"], {"anthropic": "anthropic limited"})
        self.assertCountEqual(result["providers"], _reference([event], NOW, None)[0])
        self.assert_projection([event])
        added = provider_status.observe(
            "openai-codex",
            kind="availability",
            state="available",
            source_tool="test",
            source_method="released-writer",
            now=NOW,
        )
        self.assert_projection([event, added])

    def test_malformed_event_diagnostic_persists_until_atomic_repair_or_removal(self) -> None:
        previous = _event(1, kind="inventory", state="available")
        repaired = _event(2, route="openai-codex")
        _publish(self.root, previous)
        self.assert_projection([previous])
        bad_path = self.root / "outbox" / f"{repaired['event_id']}.json"
        _publish_bytes(bad_path, b"{")
        damaged = provider_status.snapshot(now=NOW)
        self.assertCountEqual(
            cast(list[Event], damaged["providers"]), _reference([previous], NOW, None)[0]
        )
        self.assertEqual(damaged["pending_observations"], 1)
        self.assertTrue(
            any(bad_path.name in item for item in cast(list[str], damaged["diagnostics"]))
        )
        with _history_io(self.root) as counts:
            warm = provider_status.snapshot(now=NOW)
        self.assertEqual(warm, damaged)
        self.assert_no_history_io(counts)

        _publish(self.root, repaired, "outbox")
        fixed = self.assert_projection([previous, repaired])
        self.assertEqual(fixed["pending_observations"], 1)
        _publish_bytes(bad_path, b"\xff")
        damaged_again = provider_status.snapshot(now=NOW)
        self.assertTrue(
            any(bad_path.name in item for item in cast(list[str], damaged_again["diagnostics"]))
        )
        bad_path.unlink()
        removed = self.assert_projection([previous])
        self.assertEqual(removed["pending_observations"], 0)
        with _history_io(self.root) as counts:
            self.assert_projection([previous])
        self.assert_no_history_io(counts)

    def test_external_directory_and_event_changes_invalidate_projection(self) -> None:
        self.assert_projection([])
        event = _event(1)
        pending = _publish(self.root, event, "outbox")
        self.assertEqual(self.assert_projection([event])["pending_observations"], 1)
        settled = self.root / "events" / pending.name
        settled.parent.mkdir()
        pending.replace(settled)
        self.assertEqual(self.assert_projection([event])["pending_observations"], 0)

        replacement = _event(1, state="available")
        _publish(self.root, replacement)
        self.assert_projection([replacement])
        settled.unlink()
        self.assert_projection([])

        replacement_directory = self.root / "replacement"
        fresh = _event(2, route="openai-codex")
        _publish(self.root, fresh, "replacement")
        (self.root / "events").rename(self.root / "retired-events")
        replacement_directory.rename(self.root / "events")
        self.assert_projection([fresh])
        with _history_io(self.root) as counts:
            self.assert_projection([fresh])
        self.assert_no_history_io(counts)

    def test_legacy_v1_projection_cannot_hide_blessed_missing_event(self) -> None:
        visible = _event(1, kind="inventory", state="available")
        omitted = _event(2, route="openai-codex")
        _publish(self.root, visible)
        pending = _publish(self.root, omitted, "outbox")
        generations: dict[str, list[int]] = {}
        for directory in ("outbox", "events"):
            status = (self.root / directory).stat()
            generations[directory] = [
                status.st_dev,
                status.st_ino,
                status.st_mtime_ns,
                status.st_ctime_ns,
                status.st_nlink,
                status.st_size,
            ]
        # The unsafe predecessor could save a valid checksum/current generations
        # while omitting an event and its pending count. Integrity is not freshness.
        payload: Event = {
            "schema_version": "provider-status-projection/v1",
            "generations": generations,
            "latest": [visible],
            "pending_observations": 0,
            "diagnostics": [],
        }
        checksum = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _publish_bytes(
            self.root / "projection.json",
            json.dumps({**payload, "checksum": f"sha256:{checksum}"}).encode(),
        )
        with _history_io(self.root) as counts:
            value = provider_status.snapshot(now=NOW)
        providers, unavailable = _reference([visible, omitted], NOW, None)
        self.assertCountEqual(cast(list[Event], value["providers"]), providers)
        self.assertEqual(value["unavailable_routes"], unavailable)
        self.assertTrue(
            any("unsupported schema" in item for item in cast(list[str], value["diagnostics"]))
        )
        self.assertEqual(value["pending_observations"], 1)
        self.assertIn(pending, counts["reads"], "migration trusted the unsafe v1 projection")
        self.assert_reconciled_projection([visible, omitted], pending=1)

    def test_legacy_append_during_observe_reconciles_facts_and_pending(self) -> None:
        # Both schedules insert an unlocked append before projection publication.
        # Its fact and pending count must survive the locked writer's update.
        for directory in ("events", "outbox"):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                previous = _event(1, kind="inventory", state="available")
                legacy = _event(2, route="openai-codex")
                _publish(root, previous)
                replace = Path.replace
                injected = False

                def append_after_commit(
                    path: Path,
                    target: Path,
                    *,
                    root: Path = root,
                    legacy: Event = legacy,
                    directory: str = directory,
                    replace: Callable[[Path, Path], Path] = replace,
                ) -> Path:
                    nonlocal injected
                    result = replace(path, target)
                    if target.parent == root / "outbox" and not injected:
                        injected = True
                        _publish(root, legacy, directory)
                    return result

                with mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_DIR": str(root)}):
                    self.assert_projection([previous])
                    with mock.patch.object(Path, "replace", append_after_commit):
                        added = provider_status.observe(
                            "kimi-code",
                            kind="availability",
                            state="available",
                            source_tool="test",
                            source_method="legacy-append-during-observe",
                            now=NOW,
                        )
                    self.assertTrue(injected, "legacy append hook never reached event rename")
                    expected = [previous, legacy, added]
                    pending = 1 + (directory == "outbox")
                    self.assertEqual(
                        self.assert_projection(expected)["pending_observations"], pending
                    )
                    with _history_io(root) as counts:
                        self.assertEqual(
                            self.assert_projection(expected)["pending_observations"], pending
                        )
                    self.assert_no_history_io(counts)

    def test_legacy_settlement_during_observe_reconciles_pending(self) -> None:
        previous = _event(1, kind="inventory", state="available")
        legacy = _event(2, route="openai-codex")
        _publish(self.root, previous)
        pending = _publish(self.root, legacy, "outbox")
        self.assertEqual(self.assert_projection([previous, legacy])["pending_observations"], 1)
        replace = Path.replace
        injected = False

        def settle_after_commit(path: Path, target: Path) -> Path:
            nonlocal injected
            result = replace(path, target)
            if target.parent == self.root / "outbox" and not injected:
                injected = True
                pending.replace(self.root / "events" / pending.name)
            return result

        with mock.patch.object(Path, "replace", settle_after_commit):
            added = provider_status.observe(
                "kimi-code",
                kind="availability",
                state="available",
                source_tool="test",
                source_method="legacy-settlement-during-observe",
                now=NOW,
            )
        self.assertTrue(injected, "legacy settlement hook never reached event rename")
        # The predecessor keeps both pending counts despite the legacy move.
        self.assert_reconciled_projection([previous, legacy, added], pending=1)

    def test_legacy_replacement_during_observe_reconciles_changed_fact(self) -> None:
        previous = _event(1)
        replacement = _event(1, state="available")
        replaced_path = _publish(self.root, previous)
        old_inode = replaced_path.stat().st_ino
        self.assert_projection([previous])
        replace = Path.replace
        injected = False

        def replace_after_commit(path: Path, target: Path) -> Path:
            nonlocal injected
            result = replace(path, target)
            if target.parent == self.root / "outbox" and not injected:
                injected = True
                _publish(self.root, replacement)
                self.assertNotEqual(replaced_path.stat().st_ino, old_inode)
            return result

        with mock.patch.object(Path, "replace", replace_after_commit):
            added = provider_status.observe(
                "openai-codex",
                kind="availability",
                state="available",
                source_tool="test",
                source_method="legacy-replacement-during-observe",
                now=NOW,
            )
        self.assertTrue(injected, "legacy replacement hook never reached event rename")
        # Names and counts did not change: the new inode is the distinguishing fact.
        # The predecessor incorrectly retains the unavailable historical body.
        self.assert_reconciled_projection([replacement, added], pending=1)

    def test_legacy_replacement_of_new_observation_uses_own_original_inode(self) -> None:
        previous = _event(1, kind="inventory", state="available")
        _publish(self.root, previous)
        self.assert_projection([previous])
        replace = Path.replace
        replacements: list[Event] = []

        def replace_new_observation(path: Path, target: Path) -> Path:
            result = replace(path, target)
            if target.parent == self.root / "outbox" and not replacements:
                original_inode = target.stat().st_ino
                replacement = cast(Event, json.loads(target.read_text()))
                replacement["fact"] = {
                    **cast(Event, replacement["fact"]),
                    "state": "unavailable",
                    "reason": "legacy replacement",
                }
                replacements.append(replacement)
                _publish(self.root, replacement, "outbox")
                self.assertNotEqual(target.stat().st_ino, original_inode)
            return result

        with mock.patch.object(Path, "replace", replace_new_observation):
            provider_status.observe(
                "openai-codex",
                kind="availability",
                state="available",
                source_tool="test",
                source_method="legacy-replacement-of-new-observation",
                now=NOW,
            )
        self.assertEqual(len(replacements), 1, "new observation replacement hook never ran")
        # A post-rename stat must not bless this foreign body as the writer's inode.
        self.assert_reconciled_projection([previous, *replacements], pending=1)

    def test_legacy_replacement_of_settlement_uses_original_source_inode(self) -> None:
        previous = _event(1, kind="inventory", state="available")
        _publish(self.root, previous)
        pending_event = _event(2, route="openai-codex", state="available")
        source = _publish(self.root, pending_event, "outbox")
        original_inode = source.stat().st_ino
        destination = self.root / "events" / source.name
        replacement = _event(2, route="openai-codex")
        self.assert_projection([previous, pending_event])
        replace = Path.replace
        injected = False

        def replace_settled_event(path: Path, target: Path) -> Path:
            nonlocal injected
            result = replace(path, target)
            if path == source and target == destination and not injected:
                injected = True
                _publish(self.root, replacement)
                self.assertNotEqual(destination.stat().st_ino, original_inode)
            return result

        def upload(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(args[:3], ["weft", "r2", "put-content"])
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", upload),
            mock.patch.object(Path, "replace", replace_settled_event),
        ):
            self.assertEqual(provider_status.sync(pull=False), [])
        self.assertTrue(injected, "settled event replacement hook never reached the move")
        # The expected destination inode must come from the captured source, not
        # a stat after rename that would accidentally authenticate the replacement.
        self.assert_reconciled_projection([previous, replacement], pending=0)

    def test_legacy_append_after_projection_capture_invalidates_next_reader(self) -> None:
        previous = _event(1, kind="inventory", state="available")
        legacy = _event(2, route="openai-codex")
        _publish(self.root, previous)
        self.assert_projection([previous])
        replace = Path.replace
        committed = False
        injected = False

        def append_before_projection_rename(path: Path, target: Path) -> Path:
            nonlocal committed, injected
            if target == self.root / "projection.json" and not injected:
                self.assertTrue(committed, "hook ran before the authoritative mutation")
                captured = cast(Event, json.loads(path.read_text()))
                self.assertEqual(captured["pending_observations"], 1)
                injected = True
                _publish(self.root, legacy, "outbox")
            result = replace(path, target)
            if target.parent == self.root / "outbox":
                committed = True
            return result

        with mock.patch.object(Path, "replace", append_before_projection_rename):
            added = provider_status.observe(
                "kimi-code",
                kind="availability",
                state="available",
                source_tool="test",
                source_method="legacy-append-after-projection-capture",
                now=NOW,
            )
        self.assertTrue(injected, "late legacy append hook never reached projection rename")
        # Captured generations make a post-capture change visible to the next reader.
        self.assert_reconciled_projection([previous, legacy, added], pending=2)

    def test_legacy_append_after_inventory_capture_keeps_the_observed_generation(self) -> None:
        previous = _event(1, kind="inventory", state="available")
        legacy = _event(2, route="openai-codex")
        _publish(self.root, previous)
        self.assert_projection([previous])
        replace = Path.replace
        stat = cast(Callable[..., os.stat_result], os.stat)
        committed = False
        injected = False
        inventory_reads = 0

        def mark_commit(path: Path, target: Path) -> Path:
            nonlocal committed
            result = replace(path, target)
            if target.parent == self.root / "outbox":
                committed = True
            return result

        def append_after_captured_stat(
            path: object, *args: object, **kwargs: object
        ) -> os.stat_result:
            nonlocal injected, inventory_reads
            captured = stat(path, *args, **kwargs)
            if committed and not injected and path == self.root / "events":
                inventory_reads += 1
                if inventory_reads == 2:
                    # The post-inventory stat is already captured. A later fresh
                    # stamp must not silently include this unadmitted event.
                    injected = True
                    _publish(self.root, legacy)
            return captured

        with (
            mock.patch.object(Path, "replace", mark_commit),
            mock.patch("os.stat", append_after_captured_stat),
        ):
            added = provider_status.observe(
                "kimi-code",
                kind="availability",
                state="available",
                source_tool="test",
                source_method="legacy-append-after-inventory-capture",
                now=NOW,
            )
        self.assertTrue(injected, "legacy append never reached the post-inventory stat")
        self.assert_reconciled_projection([previous, legacy, added], pending=1)

    def test_coarse_directory_clock_cannot_hide_a_legacy_append(self) -> None:
        previous = _event(1, kind="inventory", state="available")
        legacy = _event(2, route="openai-codex")
        _publish(self.root, previous)
        self.assert_projection([previous])
        replace = Path.replace
        stat = cast(Callable[..., os.stat_result], os.stat)
        fstat = os.fstat
        outbox_status: os.stat_result | None = None
        clock_tick = 0
        directory_tick = 0
        injected = False

        def at_tick(status: os.stat_result, tick: int) -> os.stat_result:
            assert outbox_status is not None
            ctime = outbox_status.st_ctime_ns + tick
            return os.stat_result(
                status,
                {
                    "st_atime": status.st_atime,
                    "st_mtime": status.st_mtime,
                    "st_ctime": ctime / 1_000_000_000,
                    "st_atime_ns": status.st_atime_ns,
                    "st_mtime_ns": status.st_mtime_ns,
                    "st_ctime_ns": ctime,
                },
            )

        def append_before_projection_rename(path: Path, target: Path) -> Path:
            nonlocal outbox_status, directory_tick, injected
            if (
                outbox_status is not None
                and target == self.root / "projection.json"
                and not injected
            ):
                injected = True
                _publish(self.root, legacy, "outbox")
            result = replace(path, target)
            if target.parent == self.root / "outbox":
                if outbox_status is None:
                    outbox_status = stat(self.root / "outbox")
                directory_tick = clock_tick
            return result

        def coarse_directory_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            if outbox_status is not None and path == self.root / "outbox":
                return at_tick(outbox_status, directory_tick)
            return stat(path, *args, **kwargs)

        def coarse_filesystem_clock(descriptor: int) -> os.stat_result:
            captured = fstat(descriptor)
            return captured if outbox_status is None else at_tick(captured, clock_tick)

        def advance_clock(seconds: float) -> None:
            nonlocal clock_tick
            # A scheduler wait advances the simulated filesystem quantum.
            # Already-committed directory metadata keeps its recorded tick.
            clock_tick += 1_000_000

        with (
            mock.patch.object(Path, "replace", append_before_projection_rename),
            mock.patch("os.stat", coarse_directory_stat),
            mock.patch("os.fstat", coarse_filesystem_clock),
            mock.patch("agent_execution.provider_status.time.sleep", advance_clock),
        ):
            added = provider_status.observe(
                "kimi-code",
                kind="availability",
                state="available",
                source_tool="test",
                source_method="coarse-clock-legacy-append",
                now=NOW,
            )
            self.assertTrue(injected, "coarse-clock append hook never executed")
            self.assert_reconciled_projection([previous, legacy, added], pending=2)

    def test_settlement_updates_pending_without_replay_or_persisting_transport_errors(self) -> None:
        history = [_event(number, observed=NOW - 100 + number) for number in range(32)]
        for event in history:
            _publish(self.root, event)
        self.assert_projection(history)
        pending = [
            provider_status.observe(
                route,
                kind="availability",
                state="available",
                source_tool="test",
                source_method="settlement",
                host="host-a",
                os_user="agent",
                now=NOW,
            )
            for route in ("openai-codex", "kimi-code")
        ]
        self.assertEqual(self.assert_projection([*history, *pending])["pending_observations"], 2)
        failed_name = f"{pending[0]['event_id']}.json"
        failures = {failed_name}
        remote: dict[str, bytes] = {}
        arrived_during_upload: list[Event] = []

        def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(args[:3], ["weft", "r2", "put-content"])
            # A second open-file description cannot acquire flock while sync owns it,
            # even in the same process. Fail immediately instead of hanging on observe.
            with (self.root / "projection.lock").open("rb") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            path = Path(args[args.index("--path") + 1])
            key = args[args.index("--key") + 1]
            if path.name in failures:
                return subprocess.CompletedProcess(
                    args, 1, stdout="", stderr="injected upload failure"
                )
            remote[key] = path.read_bytes()
            if not arrived_during_upload:
                arrived_during_upload.append(
                    provider_status.observe(
                        "zhipu-coding-plan",
                        kind="availability",
                        state="available",
                        source_tool="test",
                        source_method="arrived-during-upload",
                        host="host-a",
                        os_user="agent",
                        now=NOW,
                    )
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", run),
        ):
            with _history_io(self.root) as settlement_io:
                diagnostics = provider_status.sync(pull=False)
            self.assertTrue(any("injected upload failure" in item for item in diagnostics))
            self.assertTrue(arrived_during_upload)
            self.assertEqual(len(remote), 1)
            self.assertFalse(
                any(path.parent == self.root / "events" for path in settlement_io["reads"])
            )
            self.assert_inventory_only_history_io(
                settlement_io,
                touched=[*pending, *arrived_during_upload],
                mutations=2,
                listing_scans=2,
            )
            with _history_io(self.root) as warm_io:
                value = self.assert_projection([*history, *pending, *arrived_during_upload])
                self.assertEqual(value["pending_observations"], 2)
            self.assert_no_history_io(warm_io)

            failures.clear()
            with _history_io(self.root) as retry_io:
                self.assertEqual(provider_status.sync(pull=False), [])
            self.assertFalse(any(path.parent == self.root / "events" for path in retry_io["reads"]))
            self.assert_inventory_only_history_io(
                retry_io,
                touched=[*pending, *arrived_during_upload],
                mutations=2,
                listing_scans=2,
            )
            self.assertEqual(len(remote), 3)
            with _history_io(self.root) as warm_io:
                value = self.assert_projection([*history, *pending, *arrived_during_upload])
                self.assertEqual(value["pending_observations"], 0)
            self.assert_no_history_io(warm_io)
        self.assertCountEqual(
            [json.loads(contents) for contents in remote.values()],
            [*pending, *arrived_during_upload],
        )

    def test_concurrent_duplicate_settlement_keeps_the_projection_warm(self) -> None:
        history = [_event(number, observed=NOW - 100 + number) for number in range(32)]
        for event in history:
            _publish(self.root, event)
        self.assert_projection(history)
        pending = provider_status.observe(
            "openai-codex",
            kind="availability",
            state="available",
            source_tool="test",
            source_method="duplicate-publication",
            now=NOW,
        )
        uploading = threading.Barrier(2)

        def upload(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(args[:3], ["weft", "r2", "put-content"])
            uploading.wait(timeout=5)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", upload),
            _history_io(self.root) as counts,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            publications = [executor.submit(provider_status.sync, pull=False) for _ in range(2)]
            for publication in publications:
                self.assertEqual(publication.result(timeout=10), [])
        self.assert_inventory_only_history_io(
            counts,
            touched=[pending],
            mutations=2,
            listing_scans=4,
        )
        self.assertFalse(any(path.parent == self.root / "events" for path in counts["reads"]))
        with _history_io(self.root) as final_counts:
            value = self.assert_projection([*history, pending])
        self.assertEqual(value["pending_observations"], 0)
        self.assert_no_history_io(final_counts)

    def test_event_added_after_rebuild_enumeration_is_not_hidden_by_cached_generation(self) -> None:
        original = _event(1, kind="inventory", state="available")
        later = _event(2, route="openai-codex")
        _publish(self.root, original)
        scandir = os.scandir
        injected = False

        def add_after_enumeration(
            path: str | os.PathLike[str],
        ) -> contextlib.AbstractContextManager[Iterator[os.DirEntry[str]]]:
            nonlocal injected
            entries = scandir(path)
            iterator: Iterator[os.DirEntry[str]] = entries
            if Path(path) == self.root / "events" and not injected:
                iterator = iter(list(entries))
                injected = True
                _publish(self.root, later)

            class CapturedScan(
                contextlib.AbstractContextManager[Iterator[os.DirEntry[str]]],
                Iterator[os.DirEntry[str]],
            ):
                def __next__(self) -> os.DirEntry[str]:
                    return next(iterator)

                def __enter__(self) -> Iterator[os.DirEntry[str]]:
                    return self

                def __exit__(self, *args: object) -> None:
                    self.close()

                def close(self) -> None:
                    entries.close()

            return CapturedScan()

        with mock.patch("os.scandir", add_after_enumeration):
            first = provider_status.snapshot(now=NOW)
        self.assertTrue(injected, "mutation hook did not run at the directory I/O boundary")
        self.assertIn(
            "anthropic",
            {provider["route"] for provider in cast(list[Event], first["providers"])},
        )
        # Either retry the moving replay immediately, or leave it invalid for the
        # next reader. It must never bless the incomplete listing with a new signature.
        self.assert_projection([original, later])
        with _history_io(self.root) as counts:
            self.assert_projection([original, later])
        self.assert_no_history_io(counts)

    def test_unreadable_event_directory_is_diagnosed_without_caching_an_empty_view(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory-read permission checks")
        visible = _event(1, kind="inventory", state="available")
        hidden = _event(2, route="openai-codex")
        _publish(self.root, visible, "outbox")
        _publish(self.root, hidden)
        directory = self.root / "events"
        directory.chmod(0o300)
        try:
            with self.assertRaises(PermissionError), os.scandir(directory):
                pass
            value = provider_status.snapshot(now=NOW)
            providers, unavailable = _reference([visible], NOW, None)
            self.assertCountEqual(cast(list[Event], value["providers"]), providers)
            self.assertEqual(value["unavailable_routes"], unavailable)
            self.assertEqual(value["pending_observations"], 1)
            self.assertTrue(any("events" in item for item in cast(list[str], value["diagnostics"])))
            self.assertFalse((self.root / "projection.json").exists())
        finally:
            directory.chmod(0o700)
        self.assert_reconciled_projection([visible, hidden], pending=1)

    def test_transient_event_read_error_does_not_cache_permanent_missing_fact(self) -> None:
        events = [_event(1), _event(2, route="openai-codex")]
        paths = [_publish(self.root, event) for event in events]
        open_file = cast(Callable[..., object], io.open)
        failures: list[Path] = []

        def fail_once(path: object, *args: object, **kwargs: object) -> object:
            if path == paths[0] and not failures:
                failures.append(paths[0])
                raise PermissionError("injected transient event read")
            return open_file(path, *args, **kwargs)

        before = (self.root / "events").stat()
        with mock.patch("io.open", fail_once):
            provider_status.snapshot(now=NOW)
        self.assertEqual(failures, [paths[0]])
        after = (self.root / "events").stat()
        self.assertEqual(
            (before.st_ino, before.st_mtime_ns, before.st_ctime_ns),
            (after.st_ino, after.st_mtime_ns, after.st_ctime_ns),
        )
        self.assert_projection(events)
        with _history_io(self.root) as counts:
            self.assert_projection(events)
        self.assert_no_history_io(counts)

    def test_multiprocess_writers_preserve_every_subject_and_pending_count(self) -> None:
        self.assert_projection([])
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(3)
        processes = [
            context.Process(target=_concurrent_writer, args=(str(self.root), writer, barrier))
            for writer in range(3)
        ]
        started = []
        try:
            for process in processes:
                process.start()
                started.append(process)
            for process in started:
                process.join(timeout=30)
                self.assertFalse(
                    process.is_alive(), "isolated concurrent writer exceeded its bound"
                )
                self.assertEqual(process.exitcode, 0)
            committed = [
                cast(Event, json.loads(path.read_text()))
                for path in (self.root / "outbox").glob("*.json")
            ]
            self.assertEqual(len(committed), 24)
            # Do not let a recovery replay conceal lost cache updates from the writers.
            with _history_io(self.root) as counts:
                value = self.assert_projection(committed, now=NOW + 8)
                self.assertEqual(value["pending_observations"], 24)
                self.assertEqual(
                    {provider["route"] for provider in cast(list[Event], value["providers"])},
                    {f"concurrent-{writer}-{number}" for writer in range(3) for number in range(8)},
                )
            self.assert_no_history_io(counts)
        finally:
            for process in started:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                process.close()

    def test_observe_and_status_finish_while_sync_transport_is_waiting(self) -> None:
        initial = provider_status.observe(
            "anthropic",
            kind="availability",
            state="available",
            source_tool="test",
            source_method="before-network-wait",
            host="host-a",
            os_user="agent",
            now=NOW,
        )
        entered = threading.Event()
        release = threading.Event()
        uploaded: list[bytes] = []

        def waiting_upload(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.assertEqual(args[:3], ["weft", "r2", "put-content"])
            path = Path(args[args.index("--path") + 1])
            uploaded.append(path.read_bytes())
            entered.set()
            if not release.wait(timeout=15):
                raise AssertionError("test failed to release blocked transport")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def concurrent_observation() -> tuple[Event, Event]:
            event = provider_status.observe(
                "openai-codex",
                kind="quota",
                state="unavailable",
                source_tool="test",
                source_method="during-network-wait",
                detail={"reason": "openai-codex limited"},
                host="host-a",
                os_user="agent",
                now=NOW,
            )
            return event, provider_status.snapshot(now=NOW)

        with (
            mock.patch.dict(os.environ, {"AGENT_PROVIDER_STATUS_TRANSPORT": "weft"}),
            mock.patch("agent_execution.provider_status._run", waiting_upload),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            syncing = executor.submit(provider_status.sync, pull=False)
            try:
                self.assertTrue(entered.wait(timeout=5), "sync never entered the transport")
                observing = executor.submit(concurrent_observation)
                added, during_wait = observing.result(timeout=5)
                self.assertFalse(release.is_set())
                self.assertEqual(during_wait["pending_observations"], 2)
                self.assertCountEqual(
                    cast(list[Event], during_wait["providers"]),
                    _reference([initial, added], NOW, None)[0],
                )
            finally:
                release.set()
            self.assertEqual(syncing.result(timeout=10), [])
        self.assertEqual([json.loads(contents) for contents in uploaded], [initial])
        with _history_io(self.root) as counts:
            final = self.assert_projection([initial, added])
            self.assertEqual(final["pending_observations"], 1)
        self.assert_no_history_io(counts)


if __name__ == "__main__":
    unittest.main()
