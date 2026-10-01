"""Versioned provider observations shared across hosts.

The registry is advisory. Writers commit immutable observations locally before a
best-effort R2 sync; readers merge the local cache even when the remote store is
unreachable. Credentials never enter the registry. Account identities are
reduced to an HMAC fingerprint using a host-local salt.
"""

from __future__ import annotations

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
from collections.abc import Callable
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
    (r"monthly[^.\n]{0,40}(?:limit|allowance|quota|cap)\b", "quota", "monthly", 7 * 24 * 3600),
    (r"(?:limit|allowance|quota|cap)[^.\n]{0,40}\bmonthly\b", "quota", "monthly", 7 * 24 * 3600),
    (r"provider\.auth_error:\s*403", "quota", "session", 5 * 3600),
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


def _salt_path() -> Path:
    return state_dir() / "fingerprint-salt"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint_salt() -> bytes:
    override = os.environ.get("AGENT_PROVIDER_STATUS_SALT")
    if override:
        return override.encode()
    path = _salt_path()
    try:
        return path.read_bytes()
    except OSError:
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
            temporary.replace(path)
        except OSError:
            if not path.exists():
                raise
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
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


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
            except ValueError:
                seconds = 24 * 3600
            return condition, window, seconds, reset_at
        if cooldown == -1:
            return condition, window, int(match.group(1)) * 3600, None
        return condition, window, cooldown, None
    return "unknown", None, 600, None


def validate_event(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("provider observation has an unsupported schema")
    for field in ("event_id", "observed_at", "expires_at"):
        if not isinstance(value.get(field), str) or not value[field]:
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
    if fact.get("kind") not in {
        "availability",
        "authentication",
        "inventory",
        "quota",
        "transport",
    }:
        raise ValueError("provider observation has invalid fact kind")
    if fact.get("state") not in {"available", "unavailable", "unknown"}:
        raise ValueError("provider observation has invalid fact state")
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
) -> dict[str, object]:
    moment = time.time() if now is None else now
    host_name = host or socket.gethostname()
    fingerprint, fingerprint_scope = credential_fingerprint(credential_identity, host=host_name)
    event_id = f"{int(moment * 1000):013d}-{uuid.uuid4().hex}"
    fact: dict[str, object] = {"kind": kind, "state": state}
    if detail:
        fact.update(detail)
    event: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "subject": {
            "route": route_for(route),
            "billing_pool": billing_pool or billing_pool_for(route, model),
            "credential_fingerprint": fingerprint,
            "fingerprint_scope": fingerprint_scope,
            "host": host_name,
            "os_user": os_user or getpass.getuser(),
        },
        "fact": fact,
        "observed_at": _iso(moment),
        "expires_at": _iso(moment + ttl_seconds),
        "source": {"tool": source_tool, "method": source_method},
    }
    validate_event(event)
    _atomic_json(_outbox_dir() / f"{event_id}.json", event)
    _write_snapshot()
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
                source_method="usage --json --redact",
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
                    source_method="usage --json --redact",
                    host=host,
                    credential_identity=_identity_from_metadata(cast(dict[str, object], entry)),
                    detail={"quota_observable": False},
                    now=now,
                )
            )
    return events


def _read_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for directory in (_events_dir(), _outbox_dir()):
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                value = json.loads(path.read_text())
                events.append(validate_event(value))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
    unique: dict[str, dict[str, object]] = {}
    for event in events:
        unique[cast(str, event["event_id"])] = event
    return list(unique.values())


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
    except OSError as error:
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
        for line in result.stdout.splitlines():
            match = re.match(r"^\s*\d+\s+(.+)$", line)
            if match:
                keys.append(match.group(1).strip())
        return keys, None
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
            except (OSError, json.JSONDecodeError, ValueError) as error:
                diagnostics.append(f"invalid outbox event {path.name}: {error}")
                continue
            error = _upload(path, _event_key(event), transport)
            if error:
                diagnostics.append(f"publish {path.name}: {error}")
                continue
            destination = _events_dir() / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.replace(destination)
            except FileNotFoundError:
                pass
    if pull:
        keys, error = _list_remote(transport)
        if error:
            diagnostics.append(f"list observations: {error}")
        else:
            for key in keys:
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
                except (json.JSONDecodeError, ValueError) as parse_error:
                    diagnostics.append(f"invalid remote observation {key}: {parse_error}")
                    continue
                _atomic_json(destination, event)
    _write_snapshot(diagnostics=diagnostics)
    return diagnostics


def _subject_key(event: dict[str, object]) -> tuple[str, str, str, str, str, str]:
    subject = cast(dict[str, object], event["subject"])
    fact = cast(dict[str, object], event["fact"])
    return (
        cast(str, subject["route"]),
        cast(str, subject["billing_pool"]),
        cast(str, subject["credential_fingerprint"]),
        cast(str, subject["host"]),
        cast(str, subject["os_user"]),
        cast(str, fact["kind"]),
    )


def snapshot(
    *, host: str | None = None, now: float | None = None, diagnostics: list[str] | None = None
) -> dict[str, object]:
    moment = time.time() if now is None else now
    latest: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
    for event in _read_events():
        observed_at = _timestamp(event.get("observed_at"))
        if observed_at is None or observed_at < moment - RETENTION_SECONDS:
            continue
        subject = cast(dict[str, object], event["subject"])
        if host and subject["host"] != host:
            continue
        key = _subject_key(event)
        previous = latest.get(key)
        if previous is None or cast(str, event["observed_at"]) > cast(str, previous["observed_at"]):
            latest[key] = event
    successful_at: dict[tuple[str, str, str, str, str], str] = {}
    for key, event in latest.items():
        fact = cast(dict[str, object], event["fact"])
        if fact["kind"] == "availability" and fact["state"] == "available":
            successful_at[key[:-1]] = cast(str, event["observed_at"])
    providers: list[dict[str, object]] = []
    unavailable_routes: dict[str, str] = {}
    for event in sorted(
        latest.values(),
        key=lambda item: (
            cast(dict[str, object], item["subject"])["host"],
            cast(dict[str, object], item["subject"])["route"],
        ),
    ):
        subject = cast(dict[str, object], event["subject"])
        fact = cast(dict[str, object], event["fact"])
        success = successful_at.get(_subject_key(event)[:-1])
        if (
            fact["state"] == "unavailable"
            and success is not None
            and success > cast(str, event["observed_at"])
        ):
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
        for field in ("condition", "window", "reset_at", "reason", "quota", "quota_observable"):
            if field in fact:
                provider[field] = fact[field]
        providers.append(provider)
        route = cast(str, subject["route"])
        if not stale and fact["state"] == "unavailable" and fact["kind"] != "transport":
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
        "diagnostics": list(diagnostics or []),
        "pending_observations": (
            len(list(_outbox_dir().glob("*.json"))) if _outbox_dir().exists() else 0
        ),
    }


def _write_snapshot(*, diagnostics: list[str] | None = None) -> dict[str, object]:
    value = snapshot(diagnostics=diagnostics)
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
            provider["kind"] == "quota"
            and provider["state"] == "unavailable"
            and not provider["stale"]
        ):
            routes.add(cast(str, provider["route"]))
    return frozenset(routes)


def healthy_routes(candidates: list[str], *, now: float | None = None) -> list[str]:
    return [route for route in candidates if unavailable_reason(route, now=now) is None]


def probe_omp(
    *, host: str | None = None, timeout: int = 45
) -> tuple[dict[str, object], list[dict[str, object]]]:
    result = _command(["omp", "usage", "--json", "--redact"], timeout=timeout)
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
    events = observe_omp_usage(payload, host=host)
    return cast(dict[str, object], payload), events
