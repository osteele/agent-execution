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
    """One validated tool-call event from ctx's public transcript schema.

    ``fidelity`` is ``full`` only when the event content is complete and the
    call rests on ctx's explicit capture or facts; anything ctx leaves
    uncertain degrades the call to ``partial``.
    """

    tool: str | None
    detail: str
    occurred_at: float
    fidelity: str


@dataclass(frozen=True)
class ValidatedTranscript:
    """The database-neutral facts needed to construct a context manifest.

    ``fidelity`` is ``full`` only when every call is; one uncertain call
    makes the whole transcript ``partial``.
    """

    calls: tuple[TranscriptCall, ...]
    fidelity: str


def _detail(payload: dict[str, object]) -> str:
    for key in _DETAIL_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


# ctx's public normalized capture states for `invocation.arguments`. Only
# `present` vouches for its accompanying `value`; the others say the value was
# not captured, and any bytes beside them are not transcript evidence.
_CAPTURE_PRESENT = "present"
CAPTURE_STATUSES = frozenset({"present", "absent", "unavailable", "omitted"})


def _has_normalized_activity(event: dict[str, object]) -> bool:
    """Whether the event carries ctx's normalized activity at all.

    An event with activity is interpreted only through it, however uncertain
    its fields. Legacy provider-text parsing is reserved for events that carry
    no activity field.
    """
    return "activity" in event


def _captured_argument_detail(arguments: object) -> str:
    """Read a detail from normalized arguments only when capture is `present`."""
    if not isinstance(arguments, dict):
        return ""
    captured = cast(dict[str, object], arguments)
    if captured.get("capture_status") != _CAPTURE_PRESENT:
        # Explicitly absent, unavailable, omitted, unrecognized, or missing
        # status: the accompanying value is never promoted.
        return ""
    value = captured.get("value")
    if isinstance(value, dict):
        return _detail(cast(dict[str, object], value))
    if isinstance(value, str):
        return value
    return ""


def _readable_file_facts(facts: object) -> tuple[list[str], bool]:
    """The detail strings ctx's readable ``file`` facts name, in order.

    The flag reports whether the facts also carry a recognized ``file`` fact
    whose value ctx left missing or unreadable.  Such a fact is an explicit
    uncertainty about the file identity, never a record to skip silently, so it
    can ground only partial fidelity.
    """
    if not isinstance(facts, list):
        return [], False
    details: list[str] = []
    unreadable = False
    for fact in cast(list[object], facts):
        if not isinstance(fact, dict) or fact.get("kind") != "file":
            continue
        file_value = fact.get("value")
        detail = ""
        if isinstance(file_value, str):
            detail = file_value
        elif isinstance(file_value, dict):
            detail = _detail(cast(dict[str, object], file_value))
        if detail:
            details.append(detail)
        else:
            unreadable = True
    return details, unreadable


def _normalized_tool_call(event: dict[str, object]) -> tuple[str, str, bool] | None:
    """Read ctx's public normalized activity; never consult provider text.

    Returns None when the activity names no usable tool, in which case the
    caller records an unattributed call at partial fidelity rather than
    falling back to legacy ``event.text`` parsing.  Otherwise it returns
    ``(tool, detail, grounded)``.  ``grounded`` is true only when ctx
    explicitly captured the arguments (``capture_status: present`` with a
    readable value) and every readable ``file`` fact agrees with the captured
    detail; an unsupported shape, an unknown or non-present capture state,
    conflicting evidence, or a recognized ``file`` fact ctx cannot read stays
    usable through its ctx-supplied detail but is never full-fidelity grounding.
    """
    activity = event.get("activity")
    if not isinstance(activity, dict) or not _is_schema_version_1(activity.get("revision")):
        return None
    invocation = activity.get("invocation")
    if not isinstance(invocation, dict):
        return None
    tool = invocation.get("tool")
    if not isinstance(tool, str) or not tool:
        return None
    # `value` carries the public normalized argument object, and only under
    # `capture_status: present`; `file` facts provide the canonical file
    # identity when provider arguments are opaque or uncaptured.
    fact_details, unreadable_file_fact = _readable_file_facts(activity.get("facts"))
    captured = _captured_argument_detail(invocation.get("arguments"))
    if captured:
        # The captured value is the only detail ctx vouches for.  A readable
        # `file` fact that names something else is a conflict, and a recognized
        # `file` fact ctx left missing or unreadable is an explicit uncertainty:
        # neither is evidence to prefer nor to drop, so keep the captured detail
        # and let fidelity record that the file evidence is unresolved.
        return (
            tool,
            captured,
            not unreadable_file_fact and all(detail == captured for detail in fact_details),
        )
    if fact_details and all(detail == fact_details[0] for detail in fact_details):
        # Uncaptured arguments still have ctx's own file identity, but without
        # a `present` capture it is corroboration, not full-fidelity grounding.
        return tool, fact_details[0], False
    # An unknown, missing, or non-present capture state, or facts that
    # contradict each other or that ctx cannot read, name no single detail this
    # SDK may claim.
    return tool, "", False


def _tool_call(event: dict[str, object]) -> tuple[str | None, str, bool] | None:
    if _has_normalized_activity(event):
        # Normalized activity is authoritative even when its fields are
        # uncertain; provider text must not fill in what it declined to vouch for.
        return _normalized_tool_call(event)
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
    # Legacy events carry no ctx activity, so their fidelity is the event
    # content's alone; parsing changed nothing about how much ctx vouches.
    return tool, detail, True


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
        parsed = _tool_call(event)
        # Full fidelity requires complete content AND, for an event carrying
        # ctx activity, an activity this SDK can vouch for: an unsupported
        # revision or invocation, an unknown capture state, or conflicting
        # evidence leaves the call partial.  Legacy events without activity
        # keep their content-only fidelity.
        grounded = parsed[2] if parsed is not None else not _has_normalized_activity(event)
        event_fidelity = (
            "full"
            if grounded
            and isinstance(content, dict)
            and cast(dict[str, object], content).get("complete") is True
            else "partial"
        )
        if event_fidelity != "full":
            fidelity = "partial"
        calls.append(
            TranscriptCall(
                tool=parsed[0] if parsed is not None else None,
                detail=parsed[1] if parsed is not None else str(event.get("ctx_event_id", "")),
                occurred_at=occurred_at,
                fidelity=event_fidelity,
            )
        )
    if remote_cwd is not None and observed_cwds != {remote_cwd}:
        raise ValueError("ctx session export does not corroborate the worker working directory")
    return ValidatedTranscript(calls=tuple(calls), fidelity=fidelity)
