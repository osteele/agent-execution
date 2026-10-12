"""Strict normalization of Weft's public host inventory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, cast

WEFT_HOST_LIST_COMMAND = ("weft", "host", "list", "--json")
DeclarationState = Literal["absent", "empty", "populated"]


@dataclass(frozen=True)
class WeftCapabilities:
    host: str
    state: DeclarationState
    values: tuple[str, ...] | None
    ssh_target: str | None = None
    ssh_identity_file: str | None = None
    endpoint_valid: bool = True
    command: tuple[str, ...] = WEFT_HOST_LIST_COMMAND

    def to_dict(self) -> dict[str, object]:
        return {
            "host": self.host,
            "state": self.state,
            "capabilities": None if self.values is None else list(self.values),
            "ssh_target": self.ssh_target,
            "ssh_identity_file": self.ssh_identity_file,
            "command": list(self.command),
        }


def parse_weft_host_capabilities(
    text: str,
    *,
    command: tuple[str, ...] = WEFT_HOST_LIST_COMMAND,
) -> tuple[WeftCapabilities, ...]:
    """Parse every host capability declaration from Weft's public inventory."""
    try:
        raw = cast(object, json.loads(text))
    except json.JSONDecodeError as error:
        raise ValueError("weft host list --json did not return JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("weft host list --json must return an object")
    envelope = cast(dict[str, object], raw)
    version = envelope.get("version")
    if (
        envelope.get("kind") != "host_list"
        or not isinstance(version, int)
        or isinstance(version, bool)
        or version != 1
    ):
        raise ValueError(
            "unsupported weft host list envelope: "
            f"kind={envelope.get('kind')!r}, version={envelope.get('version')!r}"
        )
    hosts = envelope.get("hosts")
    if not isinstance(hosts, list):
        raise ValueError("weft host list envelope must contain a hosts array")
    declarations: list[WeftCapabilities] = []
    for index, item in enumerate(hosts):
        if not isinstance(item, dict):
            raise ValueError(f"weft host list entry {index} must be an object")
        entry = cast(dict[str, object], item)
        name = entry.get("name")
        if not isinstance(name, str):
            raise ValueError(f"weft host list entry {index} must contain a string name")
        target = entry.get("ssh_target")
        identity_file = entry.get("ssh_identity_file")
        endpoint_valid = (target is None or isinstance(target, str)) and (
            identity_file is None or isinstance(identity_file, str)
        )
        if not isinstance(target, str):
            target = None
        if not isinstance(identity_file, str):
            identity_file = None
        if "capabilities" not in entry:
            declarations.append(
                WeftCapabilities(
                    name, "absent", None, target, identity_file, endpoint_valid, command
                )
            )
            continue
        capabilities = entry["capabilities"]
        if not isinstance(capabilities, list) or not all(
            isinstance(value, str) for value in capabilities
        ):
            raise ValueError(f"weft host {name!r} capabilities must be an array of strings")
        values = tuple(cast(list[str], capabilities))
        declarations.append(
            WeftCapabilities(
                name,
                "empty" if not values else "populated",
                values,
                target,
                identity_file,
                endpoint_valid,
                command,
            )
        )
    return tuple(declarations)
