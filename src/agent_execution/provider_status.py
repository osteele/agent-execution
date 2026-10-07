"""Versioned provider observations shared across hosts.

The registry is advisory. Writers commit immutable observations locally before a
best-effort R2 sync; readers merge the local cache even when the remote store is
unreachable. Credentials never enter the registry. Account identities are
reduced to an HMAC fingerprint using a host-local salt.

The event files under ``outbox/`` and ``events/`` are the authority; each is
published once by atomic rename and never rewritten in place. Reads are served
from ``projection.json``, a rebuildable summary holding the latest raw event per
subject and kind, the pending count, and diagnostics for unusable files. It is
trusted only while both directories still have the stat generations it records,
so a warm read costs O(subjects) and opens, stats, and lists no event file;
otherwise the event files are replayed in O(history) reads.

Writers of this version hold ``projection.lock`` while they load the
projection, withdraw the saved copy, change the event directories, and publish
the updated projection; network I/O happens outside the lock. An interruption
therefore leaves replayable events and no trusted partial projection. Older
writers that append, settle, or atomically replace event files without the
lock may keep running during a rolling deployment, so a locked mutation lists
the name and inode of every event file before and after its own change. It
publishes its incremental update only when the second listing equals the first
plus its own writes, under the generations observed with that listing;
anything else is reconciled by replay. An uncontended mutation pays O(history)
directory-entry work but reads and stats no historical event. Changing a
published event file in place is not a supported writer operation.

Exact execution context. An observation may carry ``subject.execution``: the
harness, surface, selector, tool policy, transport, requester host/user,
execution build digest, wrapper profile, and launch-environment fingerprint it
was made under. Every one of those dimensions is part of the projection key, so
an exact observation never collapses with another context or with a legacy
unscoped observation, and it never enters the route-global
``unavailable_routes`` or quota-blocked views. Exact observations describe one
executor's host-local facts; they are committed directly to ``events/`` and are
never uploaded, so older readers elsewhere cannot mistake them for route-wide
evidence. ``capability``, ``generation`` and ``credential-basis`` facts exist
only in an exact context.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import getpass
import hashlib
import hmac
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

SCHEMA_VERSION = "provider-status/v2"
SNAPSHOT_SCHEMA_VERSION = "provider-status-snapshot/v2"
REMOTE_PREFIX = "provider-status/v2/observations"
REMOTE_BUCKET = "weft-results"
DEFAULT_EVENT_TTL_SECONDS = 10 * 60
RETENTION_SECONDS = 30 * 24 * 60 * 60
TRANSPORT_TIMEOUT_SECONDS = 20
_PROJECTION_SCHEMA_VERSION = "provider-status-projection/v3"
_EXECUTION_PROJECTION_SCHEMA = "provider-status-execution-projection/v1"
_REPLAY_ATTEMPTS = 3
_CLOCK_FENCE_TIMEOUT_SECONDS = 0.05

#: Fact kinds. ``capability``, ``generation`` and ``credential-basis`` are
#: meaningful only for an exact execution context (see the module docstring).
FACT_KINDS = frozenset(
    {
        "availability",
        "authentication",
        "inventory",
        "quota",
        "transport",
        "capability",
        "generation",
        "credential-basis",
    }
)
EXACT_ONLY_FACT_KINDS = frozenset({"capability", "generation", "credential-basis"})
FACT_STATES = frozenset({"available", "unavailable", "unknown"})
#: Typed refusal conditions, as ``classify`` names them.
REFUSAL_CONDITIONS = frozenset({"quota", "auth", "network", "unknown"})

#: Exact execution-context vocabulary. The labels are independent: native
#: Claude and a worker that cannot run Claude are different contexts.
EXECUTION_HARNESSES = ("claude", "codex", "omp", "omp-packet", "agy")
EXECUTION_SURFACES = ("native", "worker", "offload-task")
EXECUTION_TRANSPORTS = ("local", "weft")
EXECUTION_TOOL_POLICIES = (
    "packet-only-no-tools",
    "read-only-no-shell",
    "workspace-write-no-shell",
)
EXECUTION_IDENTITY_FIELDS = (
    "harness",
    "surface",
    "selector",
    "tool_policy",
    "transport",
    "requester_host",
    "requester_user",
    "execution_sha256",
    "profile",
    "environment_fingerprint",
    "effective_route",
)
EXACT_SELECTOR_PATTERN = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"

ROUTE_ALIASES = {
    "claude": "anthropic",
    "codex": "openai-codex",
    "glm": "zhipu-coding-plan",
    "kimi": "kimi-code",
    "zai": "zhipu-coding-plan",
}

_SIGNATURES: list[tuple[str, str, str | None, int | None]] = [
    (
        r"限额将在\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s*重置",
        "quota",
        "weekly-or-monthly",
        None,
    ),
    (r"您已达到每周/每月使用上限", "quota", "weekly-or-monthly", 24 * 3600),
    (r"You've reached your (\d+)-hour usage limit", "quota", "session", -1),
    # The weekly cap identifies a window, not its reset time.
    (r"You've reached your weekly(?: \(\d+-day\))? usage limit\b", "quota", "weekly", 600),
    (
        r"\bmonthly\b[^.\n]{0,40}\b(?:limit|allowance|quota|cap)\b",
        "quota",
        "monthly",
        7 * 24 * 3600,
    ),
    (
        r"\b(?:limit|allowance|quota|cap)\b[^.\n]{0,40}\bmonthly\b",
        "quota",
        "monthly",
        7 * 24 * 3600,
    ),
    (r"provider\.auth_error:\s*403\b", "quota", "session", 5 * 3600),
    (r"OAuth request .*token failed", "auth", None, 24 * 3600),
    (r"Cannot connect to API", "network", None, 300),
]

Runner = Callable[..., subprocess.CompletedProcess[str]]
_run: Runner = subprocess.run


def route_for(name: str) -> str:
    return ROUTE_ALIASES.get(name, name)


def billing_pool_for(route: str, model: str | None = None) -> str:
    """Map Antigravity model families to independent quota pools."""
    resolved = route_for(route)
    if resolved == "google-antigravity" and model:
        model_id = model.partition("/")[2] if "/" in model else model
        return f"google-antigravity/{'gemini' if model_id.startswith('gemini-') else 'other'}"
    return resolved


def state_dir() -> Path:
    override = os.environ.get("AGENT_PROVIDER_STATUS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "state" / "provider-status-v2"


def _events_dir() -> Path:
    return state_dir() / "events"


def _outbox_dir() -> Path:
    return state_dir() / "outbox"


def snapshot_path() -> Path:
    return state_dir() / "snapshot.json"


def _projection_path() -> Path:
    return state_dir() / "projection.json"


def _lock_path() -> Path:
    return state_dir() / "projection.lock"


def _salt_path() -> Path:
    return state_dir() / "fingerprint-salt"


def _atomic_json(path: Path, value: object) -> int:
    """Publish ``value`` at ``path`` by atomic rename, returning the new file's inode.

    The inode is read from the open file before the rename; afterwards the path
    may already name another writer's replacement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            inode = os.fstat(stream.fileno()).st_ino
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return inode


def _fingerprint_salt() -> bytes:
    override = os.environ.get("AGENT_PROVIDER_STATUS_SALT")
    if override:
        return override.encode()
    path = _salt_path()
    try:
        return path.read_bytes()
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    value = os.urandom(32)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        os.write(descriptor, value)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.chmod(temporary, 0o600)
        try:
            # Publish a complete salt without replacing a concurrent creator.
            os.link(temporary, path)
        except FileExistsError:
            pass
        return path.read_bytes()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def credential_fingerprint(identity: str | None, *, host: str) -> tuple[str, str]:
    if not identity:
        return f"host:{host}:unknown", "host"
    digest = hmac.new(_fingerprint_salt(), identity.encode(), hashlib.sha256).hexdigest()
    scope = "global" if os.environ.get("AGENT_PROVIDER_STATUS_SALT") else "host"
    return f"hmac-sha256:{digest}", scope


def _iso(timestamp: float | None = None) -> str:
    return (
        datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _timestamp(value: object) -> float | None:
    """The instant an RFC 3339 timestamp names, or None when it names none.

    A timestamp without an offset is refused: read in each host's local time,
    the same text would name different instants on different hosts.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return None if parsed.tzinfo is None else parsed.timestamp()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return cleaned[:100] or "unknown"


def classify(signature: str) -> tuple[str, str | None, int, str | None]:
    for pattern, condition, window, cooldown in _SIGNATURES:
        match = re.search(pattern, signature or "", re.IGNORECASE)
        if not match:
            continue
        if cooldown is None:
            reset_at = match.group(1)
            try:
                reset = datetime.strptime(
                    reset_at.replace("T", " "), "%Y-%m-%d %H:%M:%S"
                ).astimezone()
                seconds = max(3600, int(reset.timestamp() - time.time()))
            except (ValueError, OverflowError, OSError):
                seconds = 24 * 3600
            return condition, window, seconds, reset_at
        if cooldown == -1:
            try:
                seconds = int(match.group(1)) * 3600
                # Refusal expiration must remain representable by the event format.
                datetime.fromtimestamp(time.time() + seconds, timezone.utc)
            except (ValueError, OverflowError, OSError):
                seconds = DEFAULT_EVENT_TTL_SECONDS
            if seconds <= 0:
                seconds = DEFAULT_EVENT_TTL_SECONDS
            return condition, window, seconds, None
        return condition, window, cooldown, None
    return "unknown", None, 600, None


def validate_execution_identity(value: object) -> dict[str, object]:
    """Validate exact context before accepting an immutable observation."""
    if not isinstance(value, dict) or set(value) != set(EXECUTION_IDENTITY_FIELDS):
        raise ValueError("invalid execution identity fields")
    for field, choices in (
        ("harness", EXECUTION_HARNESSES),
        ("surface", EXECUTION_SURFACES),
        ("transport", EXECUTION_TRANSPORTS),
        ("tool_policy", EXECUTION_TOOL_POLICIES),
    ):
        if not isinstance(value[field], str) or value[field] not in choices:
            raise ValueError(f"invalid execution identity {field}")
    for field in ("requester_host", "requester_user", "environment_fingerprint"):
        if not isinstance(value[field], str) or not value[field] or len(value[field]) > 512:
            raise ValueError(f"invalid execution identity {field}")
    if not isinstance(value["execution_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["execution_sha256"]
    ):
        raise ValueError("invalid execution identity build")
    selector = value["selector"]
    if selector is not None and (
        not isinstance(selector, str) or not re.fullmatch(EXACT_SELECTOR_PATTERN, selector)
    ):
        raise ValueError("invalid execution identity selector")
    for field in ("profile", "effective_route"):
        item = value[field]
        if item is not None and (not isinstance(item, str) or not item or len(item) > 128):
            raise ValueError(f"invalid execution identity {field}")
    return cast(dict[str, object], value)


def _execution_scope(value: object) -> str:
    return (
        ""
        if value is None
        else json.dumps(validate_execution_identity(value), sort_keys=True, separators=(",", ":"))
    )


def validate_event(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("provider observation has an unsupported schema")
    if (
        not isinstance(value.get("event_id"), str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value["event_id"]) is None
    ):
        raise ValueError("provider observation has invalid event_id")
    # Observations arrive from other hosts and other versions, and they are
    # ordered by these instants, so the text must name one.
    for field in ("observed_at", "expires_at"):
        if _timestamp(value.get(field)) is None:
            raise ValueError(f"provider observation has invalid {field}")
    subject = value.get("subject")
    if not isinstance(subject, dict):
        raise ValueError("provider observation has no subject")
    for field in ("route", "billing_pool", "credential_fingerprint", "host", "os_user"):
        if not isinstance(subject.get(field), str) or not subject[field]:
            raise ValueError(f"provider observation subject has invalid {field}")
    fact = value.get("fact")
    if not isinstance(fact, dict):
        raise ValueError("provider observation has no fact")
    if not isinstance(fact.get("kind"), str) or fact["kind"] not in FACT_KINDS:
        raise ValueError("provider observation has invalid fact kind")
    if not isinstance(fact.get("state"), str) or fact["state"] not in {
        "available",
        "unavailable",
        "unknown",
    }:
        raise ValueError("provider observation has invalid fact state")
    execution = subject.get("execution")
    if execution is not None:
        validate_execution_identity(execution)
    elif fact["kind"] in EXACT_ONLY_FACT_KINDS:
        raise ValueError("exact fact lacks execution identity")
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("tool"), str)
        or not source["tool"]
    ):
        raise ValueError("provider observation has invalid source")
    return cast(dict[str, object], value)


def observe(
    route: str,
    *,
    kind: str,
    state: str,
    source_tool: str,
    source_method: str,
    host: str | None = None,
    os_user: str | None = None,
    billing_pool: str | None = None,
    model: str | None = None,
    credential_identity: str | None = None,
    ttl_seconds: int = DEFAULT_EVENT_TTL_SECONDS,
    detail: dict[str, object] | None = None,
    now: float | None = None,
    execution_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Commit one immutable observation.

    ``execution_identity`` scopes the observation to one exact execution
    context. Such an observation is a host-local executor fact: it is settled
    directly into the local event log and is never queued for upload.
    """
    moment = time.time() if now is None else now
    host_name = host or socket.gethostname()
    fingerprint, fingerprint_scope = credential_fingerprint(credential_identity, host=host_name)
    event_id = f"{int(moment * 1000):013d}-{uuid.uuid4().hex}"
    fact: dict[str, object] = {"kind": kind, "state": state}
    if detail:
        fact.update(detail)
    subject: dict[str, object] = {
        "route": route_for(route),
        "billing_pool": billing_pool or billing_pool_for(route, model),
        "credential_fingerprint": fingerprint,
        "fingerprint_scope": fingerprint_scope,
        "host": host_name,
        "os_user": os_user or getpass.getuser(),
    }
    if execution_identity is not None:
        subject["execution"] = dict(execution_identity)
    event: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "subject": subject,
        "fact": fact,
        "observed_at": _iso(moment),
        "expires_at": _iso(moment + ttl_seconds),
        "source": {"tool": source_tool, "method": source_method},
    }
    validate_event(event)
    if execution_identity is not None:
        cache: list[str] = []
        with _registry_lock(cache, create=True) as locked:
            if not locked:
                raise OSError(cache[-1])
            projection = _execution_projection(cache, locked=True)
            (state_dir() / "execution-projection.json").unlink(missing_ok=True)
            _atomic_json(_events_dir() / "execution" / f"{event_id}.json", event)
            projection.admit(event)
            projection.generations = {"execution": _generation(_events_dir() / "execution")}
            _publish_execution_projection(projection)
        return event
    with _mutation([], write_snapshot=True) as change:
        path = _outbox_dir() / f"{event_id}.json"
        change.created(path, _atomic_json(path, event))
        change.projection.admit(event)
        change.projection.pending += 1
    return event


def record_refusal(
    route: str,
    signature: str,
    *,
    observed_by: str = "agent-execution",
    model: str | None = None,
    billing_pool: str | None = None,
) -> dict[str, object]:
    condition, window, cooldown, reset_at = classify(signature)
    detail: dict[str, object] = {
        "condition": condition,
        "reason": f"provider reported {condition} refusal",
    }
    if window:
        detail["window"] = window
    if reset_at:
        detail["reset_at"] = reset_at
    event = observe(
        route,
        kind=(
            "quota"
            if condition == "quota"
            else "authentication"
            if condition == "auth"
            else "transport"
            if condition == "network"
            else "availability"
        ),
        state="unavailable",
        source_tool=observed_by,
        source_method="provider-refusal",
        model=model,
        billing_pool=billing_pool,
        ttl_seconds=cooldown,
        detail=detail,
    )
    publish_async()
    return event


def record_success(
    route: str,
    *,
    observed_by: str = "agent-execution",
    model: str | None = None,
    billing_pool: str | None = None,
) -> dict[str, object]:
    event = observe(
        route,
        kind="availability",
        state="available",
        source_tool=observed_by,
        source_method="successful-dispatch",
        model=model,
        billing_pool=billing_pool,
    )
    publish_async()
    return event


def _identity_from_metadata(metadata: dict[str, object]) -> str | None:
    """Select an exact identity from unredacted ``omp usage --json`` metadata."""
    for field in ("email", "accountId", "projectId", "orgId"):
        value = metadata.get(field)
        if isinstance(value, str) and value:
            return f"{field}:{value}"
    return None


def _quota_summary(limit: object) -> dict[str, object] | None:
    if not isinstance(limit, dict):
        return None
    summary: dict[str, object] = {}
    for source_name, target_name in (("id", "id"), ("label", "label"), ("status", "status")):
        value = limit.get(source_name)
        if isinstance(value, str) and value:
            summary[target_name] = value
    scope = limit.get("scope")
    if isinstance(scope, dict):
        for field in ("sharedGroup", "tier", "modelId", "windowId"):
            value = scope.get(field)
            if isinstance(value, str) and value:
                summary[field] = value
    amount = limit.get("amount")
    if isinstance(amount, dict):
        remaining = amount.get("remainingFraction")
        if isinstance(remaining, int | float):
            summary["remaining_fraction"] = float(remaining)
    window = limit.get("window")
    if isinstance(window, dict):
        resets_at = window.get("resetsAt")
        if isinstance(resets_at, int | float):
            summary["resets_at"] = _iso(float(resets_at) / 1000)
    return summary or None


def observe_omp_usage(
    payload: object, *, host: str | None = None, now: float | None = None
) -> list[dict[str, object]]:
    """Record unredacted OMP usage; hash identities before persisting observations."""
    if not isinstance(payload, dict) or not isinstance(payload.get("reports"), list):
        raise ValueError("omp usage did not return a reports array")
    events: list[dict[str, object]] = []
    for raw_report in payload["reports"]:
        if not isinstance(raw_report, dict):
            raise ValueError("omp usage report is not an object")
        provider = raw_report.get("provider")
        if not isinstance(provider, str) or not provider:
            raise ValueError("omp usage report has no provider")
        metadata = raw_report.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        allowed = metadata.get("allowed")
        limit_reached = metadata.get("limitReached")
        state = "unavailable" if allowed is False or limit_reached is True else "available"
        limits = raw_report.get("limits")
        quota = (
            [summary for item in limits if (summary := _quota_summary(item))]
            if isinstance(limits, list)
            else []
        )
        events.append(
            observe(
                provider,
                kind="inventory",
                state=state,
                source_tool="omp",
                source_method="usage --json",
                host=host,
                credential_identity=_identity_from_metadata(cast(dict[str, object], metadata)),
                detail={"quota": quota},
                now=now,
            )
        )
    for field, state in (
        ("accountsWithoutUsage", "available"),
        ("disabledCredentials", "unavailable"),
    ):
        entries = payload.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"omp usage {field} entry is not an object")
            provider = entry.get("provider")
            if not isinstance(provider, str) or not provider:
                raise ValueError(f"omp usage {field} entry has no provider")
            events.append(
                observe(
                    provider,
                    kind="authentication",
                    state=state,
                    source_tool="omp",
                    source_method="usage --json",
                    host=host,
                    credential_identity=_identity_from_metadata(cast(dict[str, object], entry)),
                    detail={"quota_observable": False},
                    now=now,
                )
            )
    return events


# route, billing pool, credential fingerprint, host, OS user, exact execution
# scope ("" for legacy unscoped events), fact kind.
_Key = tuple[str, str, str, str, str, str, str]
_Order = tuple[float, str]
# Stat fields of each event directory; None while the directory does not exist.
_Generations = dict[str, list[int] | None]
# The inode of every event file, keyed by its directory's and its own name.
_Inventory = dict[tuple[str, str], int]


def _subject_key(event: dict[str, object]) -> _Key:
    subject = cast(dict[str, object], event["subject"])
    fact = cast(dict[str, object], event["fact"])
    return (
        cast(str, subject["route"]),
        cast(str, subject["billing_pool"]),
        cast(str, subject["credential_fingerprint"]),
        cast(str, subject["host"]),
        cast(str, subject["os_user"]),
        _execution_scope(subject.get("execution")),
        cast(str, fact["kind"]),
    )


def _order(event: dict[str, object]) -> _Order:
    # By instant, not by text: the same instant has several valid spellings.
    # Equal instants fall back to event_id so every replay picks the same event.
    return cast(float, _timestamp(event["observed_at"])), cast(str, event["event_id"])


@dataclasses.dataclass
class _Projection:
    """What a snapshot needs from the event files, sized by subjects, not history.

    ``latest`` keeps the newest raw event per subject and kind before
    retention, expiry, host scope, or success-clears-refusal; those depend on
    the reader and are applied on every render. ``generations`` are the
    directory states this summarizes; None marks a projection that must never
    be published.
    """

    generations: _Generations | None
    latest: dict[_Key, tuple[_Order, dict[str, object]]] = dataclasses.field(default_factory=dict)
    pending: int = 0
    diagnostics: list[str] = dataclasses.field(default_factory=list)

    def admit(self, event: dict[str, object]) -> None:
        key, order = _subject_key(event), _order(event)
        previous = self.latest.get(key)
        if previous is None or order >= previous[0]:
            self.latest[key] = (order, event)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": _PROJECTION_SCHEMA_VERSION,
            "generations": self.generations,
            "latest": [self.latest[key][1] for key in sorted(self.latest)],
            "pending_observations": self.pending,
            "diagnostics": self.diagnostics,
        }


def _checksum(payload: dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def _note(diagnostics: list[str], message: str) -> None:
    if message not in diagnostics:
        diagnostics.append(message)


def _describe(error: Exception) -> str:
    # strerror omits the temporary file names that would make repeats distinct.
    if isinstance(error, OSError) and error.strerror:
        return error.strerror
    return str(error)


def _generation(directory: Path) -> list[int] | None:
    try:
        status = os.stat(directory)
    except FileNotFoundError:
        return None
    # Creating, renaming, or removing an entry advances mtime and ctime, and
    # ctime cannot be set back; device and inode catch a replaced directory.
    # APFS also counts entries in nlink and size.
    return [
        status.st_dev,
        status.st_ino,
        status.st_mtime_ns,
        status.st_ctime_ns,
        status.st_nlink,
        status.st_size,
    ]


def _generations() -> _Generations:
    return {"outbox": _generation(_outbox_dir()), "events": _generation(_events_dir())}


def _event_entries(directory: Path) -> Iterator[os.DirEntry[str]]:
    try:
        entries = os.scandir(directory)
    except FileNotFoundError:
        return
    with entries:
        for entry in entries:
            if entry.name.endswith(".json"):
                yield entry


def _list_events() -> _Inventory:
    """Every event file's inode, from directory entries alone.

    ``DirEntry.inode`` comes from the listing itself, so this opens and stats
    no event file: O(history) entry metadata, no event contents.
    """
    inventory: _Inventory = {}
    for directory in (_outbox_dir(), _events_dir()):
        for entry in _event_entries(directory):
            inventory[(directory.name, entry.name)] = entry.inode()
    return inventory


def _clock_fence(generations: _Generations) -> bool:
    """Observe this filesystem's clock strictly beyond the directories' ctimes.

    Equal timestamps can cover several mutations within one clock quantum.
    A same-device temporary file witnesses the filesystem clock before a scan,
    so any subsequent directory change must differ from the saved generations.
    The wait is bounded; failure to establish the fence forbids caching.
    """
    existing = [value for value in generations.values() if value is not None]
    if not existing:
        return True
    newest = max(value[3] for value in existing)
    deadline = time.monotonic() + _CLOCK_FENCE_TIMEOUT_SECONDS
    while True:
        descriptor, name = tempfile.mkstemp(dir=state_dir(), prefix=".projection-clock-")
        try:
            witness = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            Path(name).unlink(missing_ok=True)
        if any(value[0] != witness.st_dev for value in existing):
            return False
        if witness.st_ctime_ns > newest:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.001, remaining))


def _inventory() -> tuple[_Generations, _Inventory] | None:
    """The event files' inventory with the generations it belongs to.

    None when enumeration, generation stability, or the clock fence fails.
    """
    try:
        before = _generations()
        if not _clock_fence(before):
            return None
        inventory = _list_events()
        after = _generations()
    except OSError:
        return None
    return (after, inventory) if after == before else None


def _scan() -> tuple[_Projection, bool]:
    """Stream every local event file into a new projection.

    Time is O(history); memory holds one event per subject and kind plus the
    diagnostics. The flag reports complete enumeration and event reads.
    """
    projection = _Projection(None)
    complete = True
    outbox = _outbox_dir()
    # Sync moves outbox -> events; scanning in that order cannot miss a move
    # that occurs between the two directory enumerations.
    for directory in (outbox, _events_dir()):
        try:
            for entry in _event_entries(directory):
                path = Path(entry.path)
                pending = directory == outbox
                try:
                    try:
                        text = path.read_text()
                    except FileNotFoundError:
                        if not pending:
                            raise
                        # A publisher may settle an enumerated file before it is read.
                        pending = False
                        text = (_events_dir() / path.name).read_text()
                    event = validate_event(json.loads(text))
                    if path.name != f"{event['event_id']}.json":
                        raise ValueError("event_id does not match local filename")
                    projection.admit(event)
                except (OSError, ValueError) as error:
                    # Malformed content is permanent for an immutable file, but an
                    # I/O failure can heal without changing any generation.
                    complete = complete and not isinstance(error, OSError)
                    projection.diagnostics.append(
                        f"invalid local event {directory.name}/{path.name}: {error}"
                    )
                if pending:
                    projection.pending += 1
        except OSError as error:
            complete = False
            projection.diagnostics.append(
                f"unreadable local event directory {directory.name}: {error}"
            )
    return projection, complete


def _replay(*, locked: bool) -> tuple[_Projection, str | None]:
    """Rebuild the projection from the event files, with why it is unpublishable.

    A scan is publishable only if a clock fence preceded it, the directories'
    generations stayed equal, and enumeration and file reads were complete.
    Writers of this version wait on the lock; external changes cause retries.
    Replay attempts are bounded. A scan that hit an I/O failure is not repeated:
    the next load replays again.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            before = _generations()
            fenced = not locked or _clock_fence(before)
        except OSError as error:
            return _scan()[0], _describe(error)
        projection, complete = _scan()
        if not complete:
            return projection, "event history could not be read completely"
        try:
            after = _generations()
        except OSError as error:
            return projection, _describe(error)
        if after == before and fenced:
            projection.generations = after
            return projection, None
        if attempt == _REPLAY_ATTEMPTS:
            reason = (
                "event directories kept changing during replay"
                if after != before
                else "event directory clock could not be fenced"
            )
            return projection, reason


def _load_projection(
    *,
    path: Path | None = None,
    schema: str = _PROJECTION_SCHEMA_VERSION,
    generation_keys: tuple[str, ...] = ("outbox", "events"),
) -> _Projection:
    """The published projection; raises when it is absent or fails its checks."""
    with (path or _projection_path()).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError("unsupported schema")
    payload = {key: item for key, item in value.items() if key != "checksum"}
    if value.get("checksum") != _checksum(payload):
        raise ValueError("checksum mismatch")
    generations = value.get("generations")
    if (
        not isinstance(generations, dict)
        or set(generations) != set(generation_keys)
        or not all(
            item is None
            or (
                isinstance(item, list)
                and len(item) == 6
                and all(type(field) is int for field in item)
            )
            for item in generations.values()
        )
    ):
        raise ValueError("invalid generations")
    pending = value.get("pending_observations")
    if type(pending) is not int or pending < 0:
        raise ValueError("invalid pending count")
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, list) or not all(isinstance(item, str) for item in diagnostics):
        raise ValueError("invalid diagnostics")
    latest = value.get("latest")
    if not isinstance(latest, list):
        raise ValueError("invalid latest events")
    projection = _Projection(
        cast(_Generations, generations), pending=pending, diagnostics=diagnostics
    )
    for item in latest:
        event = validate_event(item)
        if _subject_key(event) in projection.latest:
            raise ValueError("repeated subject")
        projection.admit(event)
    return projection


def _withdraw(cache: list[str], reason: str) -> None:
    # Only the lock holder withdraws a projection that could not be published.
    _note(cache, f"provider-status projection not cached: {reason}")
    with contextlib.suppress(OSError):
        _projection_path().unlink(missing_ok=True)


def _publish(projection: _Projection, cache: list[str]) -> None:
    payload = projection.payload()
    try:
        _atomic_json(_projection_path(), {**payload, "checksum": _checksum(payload)})
    except (OSError, ValueError) as error:
        _withdraw(cache, _describe(error))


def _refresh(cache: list[str], *, locked: bool) -> _Projection:
    """The registry's projection as of now.

    The published projection serves while both directories keep the
    generations it records; otherwise the event files are replayed and a lock
    holder publishes the result. Nothing returned without the lock is
    publishable: its generations could be stamped over a locked writer's change.
    """
    try:
        current: _Generations | None = _generations()
    except OSError as error:
        current = None
        _note(cache, f"provider-status projection not cached: {_describe(error)}")
    if current is not None:
        try:
            projection = _load_projection()
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            _note(cache, f"provider-status projection unusable: {_describe(error)}")
        else:
            if projection.generations == current:
                if not locked:
                    projection.generations = None
                return projection
    projection, failure = _replay(locked=locked)
    if failure is not None:
        _note(cache, f"provider-status projection not cached: {failure}")
    elif locked:
        _publish(projection, cache)
    else:
        projection.generations = None
    return projection


def _open_lock() -> int:
    path = _lock_path()
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    except OSError as error:
        try:
            # flock needs no write access, so a user who may only read an
            # existing registry still serializes with its writers.
            return os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            raise error from None


@contextlib.contextmanager
def _registry_lock(cache: list[str], *, create: bool) -> Iterator[bool]:
    """Hold the registry's exclusive lock, yielding whether it is held.

    A reader takes nothing from a registry that does not exist yet. A registry
    that cannot be locked is still read in full, unpublished, with a
    diagnostic, rather than reading as empty.
    """
    if create:
        state_dir().mkdir(parents=True, exist_ok=True)
    elif not state_dir().is_dir():
        yield False
        return
    descriptor = -1
    try:
        descriptor = _open_lock()
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        _note(cache, f"provider-status registry lock unavailable: {_describe(error)}")
    try:
        yield descriptor >= 0
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclasses.dataclass
class _Change:
    """A locked mutation: the projection it updates and the files it must leave.

    ``expected`` starts as the inventory of exactly the files the projection
    summarizes and records each of the mutation's own writes. None marks a
    baseline that could not be listed, whose result is replayed, not published.
    """

    projection: _Projection
    expected: _Inventory | None

    def holds(self, path: Path) -> bool:
        """Whether an event file exists, from the baseline listing when there is one."""
        if self.expected is None:
            return path.exists()
        return (path.parent.name, path.name) in self.expected

    def created(self, path: Path, inode: int) -> None:
        if self.expected is not None:
            self.expected[(path.parent.name, path.name)] = inode

    def moved(self, source: Path, destination: Path) -> None:
        # The renamed file keeps the inode listed before the mutation; reading
        # the destination afterwards could bless a concurrent replacement.
        if self.expected is None:
            return
        inode = self.expected.pop((source.parent.name, source.name), None)
        if inode is None:
            self.expected = None
        else:
            self.expected[(destination.parent.name, destination.name)] = inode


def _baseline(cache: list[str]) -> _Change:
    """The current projection with an inventory of exactly the files it summarizes.

    The inventory counts only if the directories kept the projection's
    generations throughout the listing. A writer without the lock can change
    them in between; the pair is retaken a bounded number of times.
    """
    attempt = 0
    while True:
        attempt += 1
        projection = _refresh(cache, locked=True)
        if projection.generations is None:
            return _Change(projection, None)
        listed = _inventory()
        if listed is not None and listed[0] == projection.generations:
            return _Change(projection, listed[1])
        if attempt == _REPLAY_ATTEMPTS:
            return _Change(projection, None)


def _conclude(change: _Change, cache: list[str]) -> _Projection:
    """Publish a mutation's projection only if the files are exactly its result.

    It is stored with the generations observed alongside that exact listing,
    never a later reading that could cover another writer's change. Anything
    else, such as an unlocked append, settlement, or replacement, or a baseline
    that could not be listed, is reconciled by replaying the event files.
    """
    if change.expected is not None:
        listed = _inventory()
        if listed is not None and listed[1] == change.expected:
            change.projection.generations = listed[0]
            _publish(change.projection, cache)
            return change.projection
    return _refresh(cache, locked=True)


@contextlib.contextmanager
def _mutation(cache: list[str], *, write_snapshot: bool = False) -> Iterator[_Change]:
    """Serialize an authoritative mutation and its derived projection.

    Withdraw the saved projection before touching event files. A failure or
    process exit after an event file changes must force replay, even if the
    filesystem reports identical directory metadata for adjacent changes.
    """
    with _registry_lock(cache, create=True) as locked:
        if not locked:
            raise OSError(cache[-1])
        change = _baseline(cache)
        _projection_path().unlink(missing_ok=True)
        yield change
        projection = _conclude(change, cache)
        if write_snapshot:
            _atomic_json(snapshot_path(), _render(projection, cache=cache))


def _event_key(event: dict[str, object]) -> str:
    subject = cast(dict[str, object], event["subject"])
    return f"{REMOTE_PREFIX}/{_safe_component(cast(str, subject['host']))}/{event['event_id']}.json"


def _transport() -> str | None:
    override = os.environ.get("AGENT_PROVIDER_STATUS_TRANSPORT")
    if override:
        return None if override == "none" else override
    if shutil.which("weft") and (Path.home() / ".config" / "weft" / "config.toml").exists():
        return "weft"
    if shutil.which("rclone") and (Path.home() / ".config" / "rclone" / "rclone.conf").exists():
        return "rclone"
    return None


def publish_async() -> bool:
    """Start a finite best-effort outbox push without delaying the caller."""
    if _transport() is None:
        return False
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agent_execution.cli",
                "provider",
                "sync",
                "--push-only",
                "--json",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return False
    return True


def _remote_path(key: str) -> str:
    return f"r2:{REMOTE_BUCKET}/{key}"


def _command(
    args: list[str], *, timeout: int = TRANSPORT_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    try:
        return _run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args,
            124,
            stdout="",
            stderr=f"timed out after {timeout}s",
        )
    except (OSError, UnicodeError) as error:
        return subprocess.CompletedProcess(args, 127, stdout="", stderr=str(error))


def _upload(path: Path, key: str, transport: str) -> str | None:
    if transport == "weft":
        command = ["weft", "r2", "put-content", "--path", str(path), "--key", key]
    elif transport == "rclone":
        command = ["rclone", "copyto", str(path), _remote_path(key)]
    else:
        return f"unsupported provider-status transport {transport}"
    result = _command(command)
    return (
        None
        if result.returncode == 0
        else (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
    )


def _list_remote(transport: str) -> tuple[list[str], str | None]:
    if transport == "weft":
        result = _command(["weft", "r2", "ls", f"{REMOTE_PREFIX}/"])
        if result.returncode != 0:
            return [], result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        keys: list[str] = []
        malformed = 0
        for line in result.stdout.splitlines():
            match = re.match(r"^\s*\d+\s+(.+)$", line)
            if match:
                keys.append(match.group(1).strip())
            elif line.strip():
                malformed += 1
        return keys, f"{malformed} malformed observation listing row(s)" if malformed else None
    if transport == "rclone":
        result = _command(
            ["rclone", "lsf", "--files-only", "--recursive", _remote_path(REMOTE_PREFIX)]
        )
        if result.returncode != 0:
            return [], result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return [
            f"{REMOTE_PREFIX}/{line.strip()}" for line in result.stdout.splitlines() if line.strip()
        ], None
    return [], f"unsupported provider-status transport {transport}"


def _download(key: str, transport: str) -> tuple[str | None, str | None]:
    command = (
        ["weft", "r2", "cat", key] if transport == "weft" else ["rclone", "cat", _remote_path(key)]
    )
    result = _command(command)
    if result.returncode != 0:
        return None, result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
    return result.stdout, None


def sync(*, push: bool = True, pull: bool = True) -> list[str]:
    diagnostics: list[str] = []
    transport = _transport()
    if transport is None:
        return ["no provider-status R2 transport is configured"]
    if push and _outbox_dir().exists():
        for path in sorted(_outbox_dir().glob("*.json")):
            try:
                event = validate_event(json.loads(path.read_text()))
                if path.name != f"{event['event_id']}.json":
                    raise ValueError("event_id does not match outbox filename")
            except (OSError, ValueError) as error:
                diagnostics.append(f"invalid outbox event {path.name}: {error}")
                continue
            error = _upload(path, _event_key(event), transport)
            if error:
                diagnostics.append(f"publish {path.name}: {error}")
                continue
            destination = _events_dir() / path.name
            vanished = f"cache {path.name}: event disappeared before settlement"
            try:
                with _mutation(diagnostics) as change:
                    if change.holds(path):
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        path.replace(destination)
                        change.moved(path, destination)
                        # Already projected while pending; only the count changes.
                        change.projection.pending -= 1
                    elif not change.holds(destination):
                        diagnostics.append(vanished)
                    # Otherwise a concurrent sync settled this upload first: nothing changes.
            except FileNotFoundError:
                # A writer without the lock moved or removed it after the baseline listing.
                if not destination.exists():
                    diagnostics.append(vanished)
            except OSError as error:
                diagnostics.append(f"cache {path.name}: {error}")
    if pull:
        keys, error = _list_remote(transport)
        if error:
            diagnostics.append(f"list observations: {error}")
        for key in keys:
            if (
                re.fullmatch(
                    rf"{re.escape(REMOTE_PREFIX)}/[A-Za-z0-9._-]+/[A-Za-z0-9][A-Za-z0-9._-]*\.json",
                    key,
                )
                is None
            ):
                diagnostics.append(f"invalid remote observation key {key}")
                continue
            name = Path(key).name
            destination = _events_dir() / name
            if destination.exists() or (_outbox_dir() / name).exists():
                continue
            text, download_error = _download(key, transport)
            if download_error:
                diagnostics.append(f"read {key}: {download_error}")
                continue
            try:
                event = validate_event(json.loads(text or ""))
                if key != _event_key(event):
                    raise ValueError("key does not match observation host and event_id")
            except ValueError as parse_error:
                diagnostics.append(f"invalid remote observation {key}: {parse_error}")
                continue
            try:
                with _mutation(diagnostics) as change:
                    # Another sync may have cached it since the check above.
                    if not (change.holds(destination) or change.holds(_outbox_dir() / name)):
                        change.created(destination, _atomic_json(destination, event))
                        change.projection.admit(event)
            except OSError as error:
                diagnostics.append(f"cache {key}: {error}")
    try:
        value = _write_snapshot(diagnostics=diagnostics)
    except OSError as error:
        diagnostics.append(f"write snapshot: {error}")
    else:
        diagnostics = cast(list[str], value["diagnostics"])
    return diagnostics


def snapshot(
    *, host: str | None = None, now: float | None = None, diagnostics: list[str] | None = None
) -> dict[str, object]:
    cache: list[str] = []
    with _registry_lock(cache, create=False) as locked:
        projection = _refresh(cache, locked=locked)
    return _render(projection, host=host, now=now, diagnostics=diagnostics, cache=cache)


def _render(
    projection: _Projection,
    *,
    host: str | None = None,
    now: float | None = None,
    diagnostics: list[str] | None = None,
    cache: list[str] | None = None,
) -> dict[str, object]:
    """The public snapshot; O(subjects), independent of event history."""
    moment = time.time() if now is None else now
    collected_diagnostics = list(diagnostics or [])
    seen = set(collected_diagnostics)
    for diagnostic in (*projection.diagnostics, *(cache or [])):
        if diagnostic not in seen:
            seen.add(diagnostic)
            collected_diagnostics.append(diagnostic)
    # The newest event of a subject is retained exactly when any of its events
    # is, so filtering the projected latest matches filtering the history.
    latest: dict[_Key, tuple[float, dict[str, object]]] = {}
    for key in sorted(projection.latest):
        (observed_at, _), event = projection.latest[key]
        if observed_at < moment - RETENTION_SECONDS or (host and key[3] != host):
            continue
        latest[key] = (observed_at, event)
    successful_at: dict[tuple[str, ...], float] = {}
    for key, (observed_at, event) in latest.items():
        fact = cast(dict[str, object], event["fact"])
        if fact["kind"] == "availability" and fact["state"] == "available":
            successful_at[key[:-1]] = observed_at
    providers: list[dict[str, object]] = []
    unavailable_routes: dict[str, str] = {}
    # By host, then route.
    for key, (observed_at, event) in sorted(
        latest.items(), key=lambda item: (item[0][3], item[0][0])
    ):
        subject = cast(dict[str, object], event["subject"])
        fact = cast(dict[str, object], event["fact"])
        success = successful_at.get(key[:-1])
        if fact["state"] == "unavailable" and success is not None and success > observed_at:
            continue
        expires_at = _timestamp(event.get("expires_at"))
        stale = expires_at is None or expires_at <= moment
        provider: dict[str, object] = {
            **subject,
            "kind": fact["kind"],
            "state": fact["state"],
            "observed_at": event["observed_at"],
            "expires_at": event["expires_at"],
            "stale": stale,
            "source": event["source"],
        }
        for field in (
            "condition",
            "window",
            "reset_at",
            "reason",
            "quota",
            "quota_observable",
            "detail",
            "credential_basis",
        ):
            if field in fact:
                provider[field] = fact[field]
        providers.append(provider)
        route = cast(str, subject["route"])
        if (
            not stale
            and "execution" not in subject
            and fact["state"] == "unavailable"
            and fact["kind"] != "transport"
        ):
            reason = (
                cast(str | None, fact.get("reason"))
                or cast(str | None, fact.get("condition"))
                or "unavailable"
            )
            unavailable_routes.setdefault(route, reason)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": _iso(moment),
        "providers": providers,
        "unavailable_routes": unavailable_routes,
        "diagnostics": collected_diagnostics,
        "pending_observations": projection.pending,
    }


def _write_snapshot(*, diagnostics: list[str] | None = None) -> dict[str, object]:
    cache: list[str] = []
    with _registry_lock(cache, create=True) as locked:
        value = _render(_refresh(cache, locked=locked), diagnostics=diagnostics, cache=cache)
        _atomic_json(snapshot_path(), value)
    return value


def unavailable_reason(route: str, *, now: float | None = None) -> str | None:
    value = snapshot(now=now)
    unavailable = cast(dict[str, str], value["unavailable_routes"])
    return unavailable.get(route_for(route))


def quota_blocked_routes(*, now: float | None = None) -> frozenset[str]:
    value = snapshot(now=now)
    routes: set[str] = set()
    for provider in cast(list[dict[str, object]], value["providers"]):
        if (
            "execution" not in provider
            and provider["kind"] == "quota"
            and provider["state"] == "unavailable"
            and not provider["stale"]
        ):
            routes.add(cast(str, provider["route"]))
    return frozenset(routes)


def _publish_execution_projection(projection: _Projection) -> None:
    payload = {**projection.payload(), "schema_version": _EXECUTION_PROJECTION_SCHEMA}
    _atomic_json(
        state_dir() / "execution-projection.json", {**payload, "checksum": _checksum(payload)}
    )


def _execution_projection(cache: list[str], *, locked: bool) -> _Projection:
    # One authoritative event store and lock. Exact facts occupy a nested
    # namespace that immutable older workers cannot read as route-wide facts.
    # Their derived index also cannot invalidate the legacy workers' cache.
    directory = _events_dir() / "execution"
    generations = {"execution": _generation(directory)}
    try:
        projection = _load_projection(
            path=state_dir() / "execution-projection.json",
            schema=_EXECUTION_PROJECTION_SCHEMA,
            generation_keys=("execution",),
        )
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as error:
        _note(cache, f"execution projection unusable: {_describe(error)}")
    else:
        if projection.generations == generations:
            return projection
    projection = _Projection(generations)
    for entry in _event_entries(directory):
        try:
            event = validate_event(json.loads(Path(entry.path).read_text()))
            if entry.name != f"{event['event_id']}.json" or "execution" not in cast(
                dict, event["subject"]
            ):
                raise ValueError("invalid exact event identity")
            projection.admit(event)
        except (OSError, ValueError) as error:
            _note(projection.diagnostics, f"execution event {entry.name}: {_describe(error)}")
    if locked and _generation(directory) == generations["execution"]:
        _publish_execution_projection(projection)
    return projection


def exact_observations(
    *,
    route: str,
    billing_pool: str,
    host: str,
    os_user: str,
    execution: Mapping[str, object],
    now: float | None = None,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """The latest observation of each fact kind for one exact execution context.

    Only exact observations with the unidentified host-scoped credential
    fingerprint match: they never join another host, user, context, or a
    legacy unscoped observation. Expired observations are returned (the caller
    reports them stale); events beyond retention are not. Returns the
    observations by kind and the registry diagnostics of the read.
    """
    moment = time.time() if now is None else now
    validate_execution_identity(dict(execution))
    cache: list[str] = []
    with _registry_lock(cache, create=False) as locked:
        projection = _execution_projection(cache, locked=locked)
    fingerprint = credential_fingerprint(None, host=host)[0]
    if execution["harness"] == "claude" and execution["effective_route"] is None:
        candidates = []
        for (observed_at, _), event in projection.latest.values():
            subject = cast(dict[str, object], event["subject"])
            scope = cast(dict[str, object], subject["execution"])
            if observed_at < moment - RETENTION_SECONDS or any(
                subject[field] != value
                for field, value in (
                    ("route", route_for(route)),
                    ("host", host),
                    ("os_user", os_user),
                    ("credential_fingerprint", fingerprint),
                )
            ):
                continue
            if all(
                scope[key] == execution[key]
                for key in EXECUTION_IDENTITY_FIELDS
                if key != "effective_route"
            ):
                candidates.append((observed_at, subject))
        if candidates:
            subject = max(candidates, key=lambda item: item[0])[1]
            execution = cast(dict[str, object], subject["execution"])
            billing_pool = cast(str, subject["billing_pool"])
    wanted = (
        route_for(route),
        billing_pool,
        fingerprint,
        host,
        os_user,
        _execution_scope(dict(execution)),
    )
    found: dict[str, dict[str, object]] = {}
    for key, ((observed_at, _), event) in projection.latest.items():
        if key[:-1] != wanted or observed_at < moment - RETENTION_SECONDS:
            continue
        found[key[-1]] = event
    diagnostics: list[str] = []
    for diagnostic in (*projection.diagnostics, *cache):
        _note(diagnostics, diagnostic)
    return found, diagnostics


def healthy_routes(candidates: list[str], *, now: float | None = None) -> list[str]:
    unavailable = cast(dict[str, str], snapshot(now=now)["unavailable_routes"])
    return [route for route in candidates if route_for(route) not in unavailable]


def probe_omp(*, host: str | None = None, timeout: int = 45) -> list[dict[str, object]]:
    # Screenshot masks are inventory-dependent aliases, not account identities.
    result = _command(["omp", "usage", "--json"], timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or f"omp usage exited {result.returncode}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"omp usage returned invalid JSON: {error}") from error
    return observe_omp_usage(payload, host=host)
