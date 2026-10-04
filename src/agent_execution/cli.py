"""Command-line interface for shared agent execution state."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from agent_execution import provider_status


def _print_json(value: object) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")


def _provider_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    provider = subparsers.add_parser("provider", help="Observe and query shared provider status")
    commands = provider.add_subparsers(dest="provider_command", required=True)

    status = commands.add_parser("status", help="Read the local materialized provider snapshot")
    status.add_argument("--host", help="Filter to one observation host label")
    status.add_argument(
        "--refresh", action="store_true", help="Synchronize with the central R2 registry first"
    )
    status.add_argument("--json", action="store_true", help="Emit the versioned JSON document")

    probe = commands.add_parser(
        "probe", help="Probe OMP subscription providers and publish observations"
    )
    probe.add_argument("--host", help="Observation host label; defaults to the system hostname")
    probe.add_argument(
        "--sync", action="store_true", help="Synchronize observations with the central registry"
    )
    probe.add_argument("--json", action="store_true", help="Emit the versioned JSON document")

    sync = commands.add_parser(
        "sync", help="Push pending observations and refresh the local snapshot"
    )
    sync.add_argument(
        "--push-only",
        action="store_true",
        help="Publish the outbox without downloading observations",
    )
    sync.add_argument(
        "--pull-only",
        action="store_true",
        help="Download observations without publishing the outbox",
    )
    sync.add_argument(
        "--json", action="store_true", help="Emit the resulting versioned JSON document"
    )

    observe = commands.add_parser("observe", help="Publish one explicit provider observation")
    observe.add_argument("--route", required=True)
    observe.add_argument(
        "--kind",
        required=True,
        choices=["availability", "authentication", "inventory", "quota", "transport"],
    )
    observe.add_argument("--state", required=True, choices=["available", "unavailable", "unknown"])
    observe.add_argument("--host")
    observe.add_argument("--billing-pool")
    observe.add_argument("--credential-identity")
    observe.add_argument("--ttl", type=int, default=provider_status.DEFAULT_EVENT_TTL_SECONDS)
    observe.add_argument("--source-tool", default="agent-execution")
    observe.add_argument("--source-method", default="explicit-observation")
    observe.add_argument("--sync", action="store_true")
    observe.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-execution")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _provider_parser(subparsers)
    return parser


def _render_status(value: dict[str, object]) -> None:
    providers = value.get("providers")
    if not isinstance(providers, list) or not providers:
        print("No provider observations.")
    else:
        for raw in providers:
            if not isinstance(raw, dict):
                continue
            stale = " stale" if raw.get("stale") else ""
            print(
                f"{raw.get('host', '?'):16} {raw.get('route', '?'):24} "
                f"{raw.get('state', 'unknown')}{stale}"
            )
    pending = value.get("pending_observations")
    if isinstance(pending, int) and pending:
        print(f"warning: {pending} observation(s) await central sync", file=sys.stderr)
    diagnostics = value.get("diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            print(f"warning: {diagnostic}", file=sys.stderr)


def _run_provider(args: argparse.Namespace) -> int:
    diagnostics: list[str] = []
    if args.provider_command == "status":
        if args.refresh:
            diagnostics = provider_status.sync()
        value = provider_status.snapshot(host=args.host, diagnostics=diagnostics)
    elif args.provider_command == "probe":
        try:
            events = provider_status.probe_omp(host=args.host)
        except (RuntimeError, ValueError) as error:
            print(f"provider probe failed: {error}", file=sys.stderr)
            return 1
        observed_host = args.host
        if observed_host is None and events:
            subject = events[0].get("subject")
            if isinstance(subject, dict) and isinstance(subject.get("host"), str):
                observed_host = subject["host"]
        if args.sync:
            diagnostics = provider_status.sync()
        value = provider_status.snapshot(host=observed_host, diagnostics=diagnostics)
    elif args.provider_command == "sync":
        if args.push_only and args.pull_only:
            print("--push-only and --pull-only are mutually exclusive", file=sys.stderr)
            return 2
        diagnostics = provider_status.sync(push=not args.pull_only, pull=not args.push_only)
        value = provider_status.snapshot(diagnostics=diagnostics)
    elif args.provider_command == "observe":
        provider_status.observe(
            args.route,
            kind=args.kind,
            state=args.state,
            source_tool=args.source_tool,
            source_method=args.source_method,
            host=args.host,
            billing_pool=args.billing_pool,
            credential_identity=args.credential_identity,
            ttl_seconds=args.ttl,
        )
        if args.sync:
            diagnostics = provider_status.sync()
        value = provider_status.snapshot(host=args.host, diagnostics=diagnostics)
    else:  # pragma: no cover - argparse enforces the command vocabulary
        raise AssertionError(args.provider_command)

    if args.json:
        _print_json(value)
    else:
        _render_status(value)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "provider":
        return _run_provider(args)
    raise AssertionError(args.command)  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
