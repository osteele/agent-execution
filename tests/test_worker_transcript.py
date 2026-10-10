from __future__ import annotations

import json
import unittest
from typing import cast

from agent_execution.worker_transcript import (
    CAPTURE_STATUSES,
    TranscriptCall,
    validate_ctx_transcript,
)

TEMPTING_PATH = "/remote/project/secrets.env"


def tool_event(
    *,
    occurred_at: str = "1970-01-01T00:00:10.500Z",
    complete: bool = True,
    text: str | None = None,
) -> dict[str, object]:
    """A legacy tool_call event: activity absent, provider text is the source."""
    return {
        "provider": "codex",
        "provider_session_id": "thread-7",
        "ctx_event_id": "event-7",
        "event_type": "tool_call",
        "occurred_at": occurred_at,
        "content": {"complete": complete},
        "text": text
        if text is not None
        else json.dumps(
            {
                "name": "view_image",
                "arguments": {"path": "/remote/project/figure.png"},
            }
        ),
    }


def cwd_event(cwd: str = "/remote/project") -> dict[str, object]:
    """A non-tool event whose normalized facts corroborate the worker cwd."""
    return {
        "provider": "codex",
        "provider_session_id": "thread-7",
        "ctx_event_id": "session-7",
        "event_type": "message",
        "occurred_at": "1970-01-01T00:00:10.000Z",
        "activity": {"facts": [{"kind": "session_cwd", "value": cwd}]},
    }


def transcript(*events: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "target": "session",
        "payload_type": "session_transcript",
        "provider": "codex",
        "provider_session_id": "thread-7",
        "mode": "log",
        "format": "json",
        "events": [*(events or (tool_event(),)), cwd_event()],
    }


def normalized_event(arguments: object, *, text: str | None = None) -> dict[str, object]:
    """A tool_call carrying ctx's public normalized activity, with no file facts."""
    event = tool_event(
        text=text
        if text is not None
        else json.dumps({"name": "Read", "input": {"file_path": TEMPTING_PATH}})
    )
    event["activity"] = {
        "revision": 1,
        "invocation": {"tool": "Read", "arguments": arguments},
        "facts": [{"kind": "session_cwd", "value": "/remote/project"}],
    }
    return event


def ctx_activity_event(
    arguments: object,
    *,
    facts: list[dict[str, object]] | None = None,
    text: str | None = None,
) -> dict[str, object]:
    """A tool_call in the retained ctx 1.0.2 revision-1 session-log shape.

    ``text`` defaults to the raw JSON tool arguments ctx 1.0.2 records beside
    normalized activity; it deliberately carries no tool name, so every detail
    must come from the activity and never from provider text.
    """
    event = tool_event(
        text=text
        if text is not None
        else json.dumps({"file_path": TEMPTING_PATH, "limit": 90, "offset": 1225})
    )
    event["activity"] = {
        "revision": 1,
        "invocation": {
            "tool": "Read",
            "arguments": arguments,
            "started_at_unix_ms": 10500,
        },
        "facts": [
            {"kind": "session_cwd", "value": "/remote/project"},
            *(facts or []),
        ],
    }
    return event


class CtxTranscriptBoundaryTests(unittest.TestCase):
    def validate(self, raw: dict[str, object]):
        return validate_ctx_transcript(
            raw,
            provider="codex",
            session_id="thread-7",
            remote_cwd="/remote/project",
        )

    def first_event(self, raw: dict[str, object]) -> dict[str, object]:
        events = cast(list[object], raw["events"])
        return cast(dict[str, object], events[0])

    def last_event(self, raw: dict[str, object]) -> dict[str, object]:
        events = cast(list[object], raw["events"])
        return cast(dict[str, object], events[-1])

    def test_schema_versions_require_the_exact_integer_version(self) -> None:
        for version in (True, False, 1.0, "1", 2):
            with self.subTest(scope="transcript", version=version):
                raw = transcript()
                raw["schema_version"] = version
                with self.assertRaisesRegex(ValueError, "unsupported ctx session schema"):
                    self.validate(raw)

            with self.subTest(scope="event", version=version):
                raw = transcript()
                self.first_event(raw)["schema_version"] = version
                with self.assertRaisesRegex(ValueError, "unsupported event schema"):
                    self.validate(raw)

    def test_envelope_requires_the_requested_json_surface(self) -> None:
        for format_ in (None, "jsonl", "markdown", True):
            with self.subTest(format=format_):
                raw = transcript()
                if format_ is None:
                    raw.pop("format")
                else:
                    raw["format"] = format_
                with self.assertRaisesRegex(ValueError, "must use JSON format"):
                    self.validate(raw)

    def test_complete_cli_transcript_refuses_pagination_markers(self) -> None:
        for marker, value in (
            ("truncated", {"events": True, "max_events": 1}),
            ("truncated", None),
            ("pagination", {"has_more": True}),
            ("has_more", False),
            ("next_cursor", "cursor-7"),
        ):
            with self.subTest(marker=marker, value=value):
                raw = transcript()
                raw[marker] = value
                with self.assertRaisesRegex(ValueError, "incomplete or paged"):
                    self.validate(raw)

    def test_tool_timestamps_use_exact_utc_millisecond_rfc3339(self) -> None:
        for occurred_at in (
            "1970-01-01T00:00:10.500Z",
            "1970-01-01T00:00:10.500+00:00",
        ):
            with self.subTest(accepted=occurred_at):
                normalized = self.validate(transcript(tool_event(occurred_at=occurred_at)))
                self.assertEqual(normalized.calls[0].occurred_at, 10.5)

        for occurred_at in (
            "1970-01-01 00:00:10.500Z",
            "1970-01-01T00:00:10Z",
            "1970-01-01T00:00:10.50Z",
            "1970-01-01T00:00:10.500000Z",
            "1970-01-01T01:00:10.500+01:00",
            "1970-01-01T00:00:10.500z",
            "1970-13-01T00:00:10.500Z",
        ):
            with (
                self.subTest(rejected=occurred_at),
                self.assertRaisesRegex(ValueError, "has an unparsable occurred_at"),
            ):
                self.validate(transcript(tool_event(occurred_at=occurred_at)))

    def test_event_shape_identity_and_worker_cwd_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-object event"):
            self.validate(transcript("not an event"))

        for key, value in (("provider", "other"), ("provider_session_id", "thread-8")):
            with self.subTest(key=key):
                raw = transcript()
                self.first_event(raw)[key] = value
                with self.assertRaisesRegex(ValueError, "event from another session"):
                    self.validate(raw)

        raw = transcript()
        activity = cast(dict[str, object], self.last_event(raw)["activity"])
        activity["facts"] = [{"kind": "session_cwd", "value": "/other/project"}]
        with self.assertRaisesRegex(ValueError, "does not corroborate"):
            self.validate(raw)

    def test_matching_cwd_does_not_override_a_conflicting_session_fact(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not corroborate"):
            self.validate(transcript(cwd_event("/other/project"), tool_event()))

        validated = self.validate(transcript(cwd_event(), tool_event()))
        self.assertEqual(validated.fidelity, "full")

    def test_public_normalized_activity_precedes_provider_text(self) -> None:
        event = tool_event(text=json.dumps({"not": "a Claude tool envelope"}))
        event["activity"] = {
            "revision": 1,
            "invocation": {
                "tool": "Read",
                "arguments": {
                    "capture_status": "present",
                    "value": {"file_path": "/remote/project/notes.md"},
                },
            },
            "facts": [
                {"kind": "session_cwd", "value": "/remote/project"},
                {"kind": "file", "value": "/remote/project/notes.md"},
            ],
        }
        normalized = self.validate(transcript(event))
        self.assertEqual(normalized.calls[0].tool, "Read")
        self.assertEqual(normalized.calls[0].detail, "/remote/project/notes.md")
        self.assertEqual(normalized.calls[0].occurred_at, 10.5)
        self.assertEqual(normalized.calls[0].fidelity, "full")

    def test_present_value_is_the_only_argument_detail_source(self) -> None:
        normalized = self.validate(
            transcript(
                normalized_event(
                    {"capture_status": "present", "value": {"file_path": "/remote/project/a.md"}},
                    text=json.dumps({"name": "Read", "input": {"file_path": TEMPTING_PATH}}),
                )
            )
        )
        self.assertEqual(
            normalized.calls[0],
            TranscriptCall(
                tool="Read", detail="/remote/project/a.md", occurred_at=10.5, fidelity="full"
            ),
        )

    def test_retained_revision_1_activity_shape_grounds_full_fidelity(self) -> None:
        runs_path = "/remote/project/src/agent_review/runs.py"
        event = ctx_activity_event(
            {
                "capture_status": "present",
                "value": {"file_path": runs_path, "limit": 90, "offset": 1225},
            },
            facts=[{"kind": "file", "value": runs_path}],
        )
        normalized = self.validate(transcript(event))
        self.assertEqual(
            normalized.calls[0],
            TranscriptCall(tool="Read", detail=runs_path, occurred_at=10.5, fidelity="full"),
        )
        self.assertEqual(normalized.fidelity, "full")
        # ctx 1.0.2 keeps the raw provider arguments in `event.text`; they
        # must not leak into the normalized call even under another path.
        self.assertNotIn(TEMPTING_PATH, json.dumps(normalized.calls[0].__dict__))

    def test_uncertain_capture_keeps_only_fact_identity_at_partial_fidelity(self) -> None:
        notes_path = "/remote/project/notes.md"
        legacy_text = json.dumps({"name": "Read", "input": {"file_path": TEMPTING_PATH}})
        cases: list[tuple[str, object]] = [
            ("absent", {"capture_status": "absent", "value": {"file_path": TEMPTING_PATH}}),
            ("unavailable", {"capture_status": "unavailable", "value": TEMPTING_PATH}),
            ("omitted", {"capture_status": "omitted", "value": TEMPTING_PATH}),
            ("unknown-status", {"capture_status": "captured", "value": TEMPTING_PATH}),
            ("missing-status", {"value": TEMPTING_PATH}),
            ("non-string-status", {"capture_status": 1, "value": TEMPTING_PATH}),
            ("present-empty-value", {"capture_status": "present", "value": {}}),
            ("malformed-list", [TEMPTING_PATH]),
            ("malformed-string", TEMPTING_PATH),
            ("malformed-null", None),
        ]
        for label, arguments in cases:
            with self.subTest(state=label):
                event = ctx_activity_event(
                    arguments,
                    facts=[{"kind": "file", "value": notes_path}],
                    text=legacy_text,
                )
                normalized = self.validate(transcript(event))
                self.assertEqual(
                    normalized.calls[0],
                    TranscriptCall(
                        tool="Read", detail=notes_path, occurred_at=10.5, fidelity="partial"
                    ),
                )
                self.assertEqual(normalized.fidelity, "partial")
                self.assertNotIn(TEMPTING_PATH, json.dumps(normalized.calls[0].__dict__))

    def test_disagreeing_arguments_and_file_facts_stay_partial(self) -> None:
        event = ctx_activity_event(
            {"capture_status": "present", "value": {"file_path": "/remote/project/a.md"}},
            facts=[{"kind": "file", "value": "/remote/project/b.md"}],
        )
        normalized = self.validate(transcript(event))
        self.assertEqual(
            normalized.calls[0],
            TranscriptCall(
                tool="Read", detail="/remote/project/a.md", occurred_at=10.5, fidelity="partial"
            ),
        )
        self.assertEqual(normalized.fidelity, "partial")

    def test_unreadable_file_fact_keeps_captured_detail_at_partial_fidelity(self) -> None:
        """A recognized ``file`` fact ctx cannot read is explicit uncertainty.

        Even beside a ``capture_status: present`` capture it must not be
        dropped as if absent, nor filled from the provider text recorded beside
        the activity: the ctx-vouched captured detail survives, but fidelity is
        partial.
        """
        notes_path = "/remote/project/notes.md"
        cases: tuple[tuple[str, list[dict[str, object]]], ...] = (
            ("non-string-value", [{"kind": "file", "value": 42}]),
            ("missing-value", [{"kind": "file"}]),
            ("unreadable-value-object", [{"kind": "file", "value": {"other": TEMPTING_PATH}}]),
            (
                "readable-and-unreadable",
                [{"kind": "file", "value": notes_path}, {"kind": "file", "value": 42}],
            ),
        )
        for label, facts in cases:
            with self.subTest(facts=label):
                event = ctx_activity_event(
                    {"capture_status": "present", "value": {"file_path": notes_path}},
                    facts=facts,
                )
                normalized = self.validate(transcript(event))
                self.assertEqual(
                    normalized.calls[0],
                    TranscriptCall(
                        tool="Read", detail=notes_path, occurred_at=10.5, fidelity="partial"
                    ),
                )
                self.assertEqual(normalized.fidelity, "partial")
                self.assertNotIn(TEMPTING_PATH, json.dumps(normalized.calls[0].__dict__))

    def test_readable_matching_file_facts_still_ground_full_fidelity(self) -> None:
        notes_path = "/remote/project/notes.md"
        cases: tuple[tuple[str, list[dict[str, object]]], ...] = (
            ("string-value", [{"kind": "file", "value": notes_path}]),
            ("object-value", [{"kind": "file", "value": {"file_path": notes_path}}]),
        )
        for label, facts in cases:
            with self.subTest(facts=label):
                event = ctx_activity_event(
                    {"capture_status": "present", "value": {"file_path": notes_path}},
                    facts=facts,
                )
                normalized = self.validate(transcript(event))
                self.assertEqual(
                    normalized.calls[0],
                    TranscriptCall(
                        tool="Read", detail=notes_path, occurred_at=10.5, fidelity="full"
                    ),
                )
                self.assertEqual(normalized.fidelity, "full")

    def test_conflicting_or_malformed_file_facts_claim_no_detail(self) -> None:
        cases: tuple[tuple[str, list[dict[str, object]]], ...] = (
            (
                "conflicting",
                [
                    {"kind": "file", "value": "/remote/project/a.md"},
                    {"kind": "file", "value": "/remote/project/b.md"},
                ],
            ),
            ("malformed", [{"kind": "file", "value": 42}]),
        )
        for label, facts in cases:
            with self.subTest(facts=label):
                event = ctx_activity_event({"capture_status": "omitted"}, facts=facts)
                normalized = self.validate(transcript(event))
                self.assertEqual(
                    normalized.calls[0],
                    TranscriptCall(tool="Read", detail="", occurred_at=10.5, fidelity="partial"),
                )
                self.assertEqual(normalized.fidelity, "partial")

    def test_uncaptured_states_never_promote_their_accompanying_value(self) -> None:
        self.assertEqual(CAPTURE_STATUSES, {"present", "absent", "unavailable", "omitted"})
        tempting_values: tuple[object, ...] = (TEMPTING_PATH, {"file_path": TEMPTING_PATH})
        arguments_cases: list[tuple[str, object]] = []
        for status in ("absent", "unavailable", "omitted"):
            for value in tempting_values:
                arguments_cases.append((status, {"capture_status": status, "value": value}))
        for value in tempting_values:
            # Unrecognized (including the invented `captured`), wrong-typed, or
            # missing status is not `present`.
            arguments_cases.append(("captured", {"capture_status": "captured", "value": value}))
            arguments_cases.append(("PRESENT", {"capture_status": "PRESENT", "value": value}))
            arguments_cases.append(("non-string", {"capture_status": True, "value": value}))
            arguments_cases.append(("missing", {"value": value}))
        # Malformed argument containers carrying the same tempting path.
        arguments_cases.append(("malformed-list", [TEMPTING_PATH]))
        arguments_cases.append(("malformed-string", TEMPTING_PATH))
        arguments_cases.append(("malformed-null", None))
        for label, arguments in arguments_cases:
            with self.subTest(state=label, arguments=arguments):
                normalized = self.validate(transcript(normalized_event(arguments)))
                call = normalized.calls[0]
                self.assertEqual(call.tool, "Read")
                self.assertEqual(call.detail, "")
                self.assertNotIn(TEMPTING_PATH, json.dumps(normalized.calls[0].__dict__))
                self.assertEqual(call.fidelity, "partial")
                self.assertEqual(normalized.fidelity, "partial")

    def test_uncertain_normalized_activity_never_falls_through_to_provider_text(self) -> None:
        legacy_text = json.dumps({"name": "Read", "input": {"file_path": TEMPTING_PATH}})
        for label, activity in (
            ("facts-only", {"facts": [{"kind": "session_cwd", "value": "/remote/project"}]}),
            (
                "wrong-revision",
                {
                    "revision": 2,
                    "invocation": {
                        "tool": "Read",
                        "arguments": {
                            "capture_status": "present",
                            "value": {"file_path": TEMPTING_PATH},
                        },
                    },
                },
            ),
            ("no-invocation", {"revision": 1}),
            ("malformed-invocation", {"revision": 1, "invocation": "Read"}),
            (
                "no-tool",
                {
                    "revision": 1,
                    "invocation": {
                        "arguments": {
                            "capture_status": "omitted",
                            "value": {"file_path": TEMPTING_PATH},
                        }
                    },
                },
            ),
            ("malformed-activity", "Read"),
            ("empty-activity", {}),
            ("null-activity", None),
        ):
            with self.subTest(activity=label):
                event = tool_event(text=legacy_text)
                event["activity"] = activity
                normalized = self.validate(transcript(event))
                self.assertEqual(
                    normalized.calls[0],
                    TranscriptCall(
                        tool=None, detail="event-7", occurred_at=10.5, fidelity="partial"
                    ),
                )
                self.assertEqual(normalized.fidelity, "partial")

    def test_activity_absent_events_keep_legacy_provider_text_parsing(self) -> None:
        event = tool_event(
            text=json.dumps({"name": "Read", "input": {"file_path": "/remote/project/b.md"}})
        )
        normalized = self.validate(transcript(event))
        self.assertEqual(
            normalized.calls[0],
            TranscriptCall(
                tool="Read",
                detail="/remote/project/b.md",
                occurred_at=10.5,
                fidelity="full",
            ),
        )
        self.assertEqual(normalized.fidelity, "full")

    def test_tool_content_degrades_visibly_and_preserves_order(self) -> None:
        first = tool_event()
        second = tool_event(complete=False, text="not JSON")
        second["ctx_event_id"] = "event-8"
        normalized = self.validate(transcript(first, second))

        self.assertEqual(
            normalized.calls,
            (
                TranscriptCall(
                    tool="view_image",
                    detail="/remote/project/figure.png",
                    occurred_at=10.5,
                    fidelity="full",
                ),
                TranscriptCall(
                    tool=None,
                    detail="event-8",
                    occurred_at=10.5,
                    fidelity="partial",
                ),
            ),
        )
        self.assertEqual(normalized.fidelity, "partial")


if __name__ == "__main__":
    unittest.main()
