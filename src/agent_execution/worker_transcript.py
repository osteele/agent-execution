"""Database-free validation of the ctx transcript returned to a remote worker.

This module is part of the worker containment boundary.  It may parse the
versioned ctx JSON surface, but it must not know where ctx or its consumers keep
their databases. Consumers interpret the normalized calls after retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import cast

_UTC_RFC3339_MILLISECONDS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}(?:Z|\+00:00)"
)


def _is_schema_version_1(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


_DETAIL_KEYS = (
    "file_path",
    "notebook_path",
    "path",
    "pattern",
    "command",
    "cmd",
    "url",
    "query",
    "uri",
)


@dataclass(frozen=True)
class TranscriptCall:
    """One validated tool-call event from ctx's public transcript schema."""

    tool: str | None
    detail: str
    occurred_at: float
    fidelity: str


@dataclass(frozen=True)
class ValidatedTranscript:
    """The database-neutral facts needed to construct a context manifest."""

    calls: tuple[TranscriptCall, ...]
    fidelity: str


def _detail(payload: dict[str, object]) -> str:
    for key in _DETAIL_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _tool_call(event: dict[str, object]) -> tuple[str, str] | None:
    text = event.get("text")
    if not isinstance(text, str):
        return None
    try:
        decoded = cast(object, json.loads(text))
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    body = cast(dict[str, object], decoded)
    tool = body.get("name")
    if not isinstance(tool, str) or not tool:
        return None
    inputs = body.get("arguments", body.get("input"))
    if isinstance(inputs, dict):
        detail = _detail(cast(dict[str, object], inputs))
    elif isinstance(inputs, str):
        detail = inputs
    else:
        detail = ""
    return tool, detail


def _spell_out_utc_designator(value: str) -> str:
    """Rewrite a trailing ``Z`` for Python 3.10's ISO timestamp parser."""
    return f"{value[:-1]}+00:00" if value.endswith("Z") else value


def _timestamp(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("ctx tool-call event lacks a string occurred_at")
    if _UTC_RFC3339_MILLISECONDS.fullmatch(value) is None:
        raise ValueError("ctx tool-call event has an unparsable occurred_at")
    normalized = _spell_out_utc_designator(value)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("ctx tool-call event has an unparsable occurred_at") from error
    return parsed.timestamp()


def validate_ctx_transcript(
    raw: dict[str, object],
    *,
    provider: str,
    session_id: str,
    remote_cwd: str | None = None,
) -> ValidatedTranscript:
    """Validate and normalize ctx's versioned session JSON without local state."""
    if not _is_schema_version_1(raw.get("schema_version")):
        raise ValueError(f"unsupported ctx session schema: {raw.get('schema_version')!r}")
    if raw.get("payload_type") != "session_transcript" or raw.get("target") != "session":
        raise ValueError("ctx session export has the wrong payload type")
    if raw.get("provider") != provider or raw.get("provider_session_id") != session_id:
        raise ValueError("ctx session export does not match the requested provider session")
    if raw.get("format") != "json":
        raise ValueError("ctx session export must use JSON format")
    mode = raw.get("mode")
    if mode in {"full", "lite"}:
        raise ValueError(
            f"ctx session export mode {mode!r} excludes tool_call events; "
            "evidence manifests require mode 'log'"
        )
    if mode != "log":
        raise ValueError("ctx session export must use mode 'log' to carry tool_call events")
    if any(key in raw for key in ("truncated", "pagination", "has_more", "next_cursor")):
        raise ValueError("ctx session export is incomplete or paged")
    events_raw = raw.get("events")
    if not isinstance(events_raw, list):
        raise ValueError("ctx session export events must be an array")

    calls: list[TranscriptCall] = []
    fidelity = "full"
    observed_cwds: set[str] = set()
    for event_raw in cast(list[object], events_raw):
        if not isinstance(event_raw, dict):
            raise ValueError("ctx session export contains a non-object event")
        event = cast(dict[str, object], event_raw)
        event_schema = event.get("schema_version")
        if event_schema is not None and not _is_schema_version_1(event_schema):
            raise ValueError("ctx session export contains an unsupported event schema")
        if event.get("provider") != provider or event.get("provider_session_id") != session_id:
            raise ValueError("ctx session export contains an event from another session")
        activity = event.get("activity")
        if isinstance(activity, dict):
            facts = cast(dict[str, object], activity).get("facts")
            if isinstance(facts, list):
                for fact_raw in cast(list[object], facts):
                    if not isinstance(fact_raw, dict):
                        continue
                    fact = cast(dict[str, object], fact_raw)
                    if fact.get("kind") == "session_cwd" and isinstance(fact.get("value"), str):
                        observed_cwds.add(cast(str, fact["value"]))
        if event.get("event_type") != "tool_call":
            continue
        occurred_at = _timestamp(event.get("occurred_at"))
        content = event.get("content")
        event_fidelity = (
            "full"
            if isinstance(content, dict)
            and cast(dict[str, object], content).get("complete") is True
            else "partial"
        )
        if event_fidelity != "full":
            fidelity = "partial"
        parsed = _tool_call(event)
        calls.append(
            TranscriptCall(
                tool=parsed[0] if parsed is not None else None,
                detail=parsed[1] if parsed is not None else str(event.get("ctx_event_id", "")),
                occurred_at=occurred_at,
                fidelity=event_fidelity,
            )
        )
    if remote_cwd is not None and remote_cwd not in observed_cwds:
        raise ValueError("ctx session export does not corroborate the worker working directory")
    return ValidatedTranscript(calls=tuple(calls), fidelity=fidelity)
