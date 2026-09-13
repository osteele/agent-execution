from __future__ import annotations

import json
import unittest
from typing import cast

from agent_execution.worker_transcript import TranscriptCall, validate_ctx_transcript


def tool_event(
    *,
    occurred_at: str = "1970-01-01T00:00:10.500Z",
    complete: bool = True,
    text: str | None = None,
) -> dict[str, object]:
    return {
        "provider": "codex",
        "provider_session_id": "thread-7",
        "ctx_event_id": "event-7",
        "event_type": "tool_call",
        "occurred_at": occurred_at,
        "content": {"complete": complete},
        "activity": {"facts": [{"kind": "session_cwd", "value": "/remote/project"}]},
        "text": text
        if text is not None
        else json.dumps(
            {
                "name": "view_image",
                "arguments": {"path": "/remote/project/figure.png"},
            }
        ),
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
        "events": list(events or (tool_event(),)),
    }


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
        activity = cast(dict[str, object], self.first_event(raw)["activity"])
        activity["facts"] = [{"kind": "session_cwd", "value": "/other/project"}]
        with self.assertRaisesRegex(ValueError, "does not corroborate"):
            self.validate(raw)

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
