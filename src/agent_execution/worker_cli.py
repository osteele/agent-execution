"""Dedicated, database-incapable command surface for shared execution workers."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from agent_execution.credentials import observe_credential_basis
from agent_execution.identity import worker_evidence_path
from agent_execution.omp_execution import require_omp_sdk
from agent_execution.processes import install_termination_guard
from agent_execution.worker import (
    SUPPORTED_WORKER_PROVIDERS,
    execute_worker,
    installed_worker_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-execution-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("identity", help="Print this installed worker's version identity")
    commands.add_parser("install-omp", help="Explicitly provision the pinned optional OMP SDK")
    probe = commands.add_parser(
        "probe", help="Observe dependencies and credentials without generating"
    )
    probe.add_argument("--provider", required=True, choices=sorted(SUPPORTED_WORKER_PROVIDERS))
    probe.add_argument("--harness-model")
    probe.add_argument("--timeout", type=float, default=60.0)

    execute = commands.add_parser("execute", help="Run one resolved harness and export evidence")
    execute.add_argument("--provider", required=True)
    execute.add_argument("--model-call-id", required=True)
    # Kept off the harness command line on purpose; see execute_worker.
    execute.add_argument("--harness-model")
    execute.add_argument("--timeout", type=float, help="Optional hard harness completion limit")
    execute.add_argument(
        "--max-cost-usd", type=float, help="Hard incremental execution cost ceiling"
    )
    execute.add_argument("--ctx-timeout", type=float, default=120.0)
    execute.add_argument("--expect-protocol", type=int)
    execute.add_argument("--expect-source-sha256")
    execute.add_argument("--prompt-payload")
    execute.add_argument("--expect-prompt-sha256")
    execute.add_argument("worker_command", nargs=argparse.REMAINDER)
    return parser


def probe_worker(
    provider: str, *, harness_model: str | None = None, timeout: float = 60.0
) -> dict[str, object]:
    """Report observed facts; credential status does not promise generation or quota."""
    if provider not in SUPPORTED_WORKER_PROVIDERS:
        raise ValueError(f"unsupported worker provider: {provider}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("probe timeout must be finite and positive")
    if not harness_model or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", harness_model):
        raise ValueError("OMP probe requires an exact provider/model selector")
    dependencies: dict[str, object]
    try:
        root, bun = require_omp_sdk()
        dependencies = {"sdk_root": str(root), "bun": bun, "error": None}
    except ValueError as error:
        dependencies = {"sdk_root": None, "bun": shutil.which("bun"), "error": str(error)}
    credential = observe_credential_basis(
        "omp",
        state_root=None,
        cwd=Path.cwd(),
        profile=harness_model,
        refresh=True,
        timeout=timeout,
    )
    return {
        "schema_version": "agent-execution.worker-probe/v1",
        "worker_identity": installed_worker_identity().to_dict(),
        "provider": provider,
        "harness_model": harness_model,
        "dependencies": dependencies,
        "credential_basis": credential.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    install_termination_guard()
    try:
        if args.command == "identity":
            print(json.dumps(installed_worker_identity().to_dict(), sort_keys=True))
            return 0
        if args.command == "install-omp":
            from agent_execution.omp_install import install_omp_runtime

            install_omp_runtime()
            return 0
        if args.command == "probe":
            print(
                json.dumps(
                    probe_worker(
                        args.provider, harness_model=args.harness_model, timeout=args.timeout
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if (args.timeout is not None and args.timeout <= 0) or args.ctx_timeout <= 0:
            raise ValueError("harness and ctx timeouts must be positive")
        # One line, on stderr, before any harness work: the job log otherwise
        # ends in `uv`'s alphabetical `+ package==version` enumeration, whose
        # last entries are `wandb`, `xxhash`, `yarl`. A list ending at "y"
        # reads exactly like a stream cut mid-work, so two sessions read a job
        # that had finished preparing hours earlier as one still preparing —
        # and the "Installed 92 packages in 551ms" summary that would have said
        # otherwise was four lines above the tail, buried under its own output.
        # This separates "hung in prep" from "hung in the command" at the
        # moment anyone looks. stderr, because stdout carries the result
        # envelope.
        print(
            f"agent-execution-worker: starting {args.provider} for model call {args.model_call_id}",
            file=sys.stderr,
            flush=True,
        )
        command = list(args.worker_command)
        if command and command[0] == "--":
            command = command[1:]
        result = execute_worker(
            provider=args.provider,
            model_call_id=args.model_call_id,
            harness_model=args.harness_model,
            command=command,
            timeout=args.timeout,
            ctx_timeout=args.ctx_timeout,
            max_cost_usd=args.max_cost_usd,
            expect_protocol=args.expect_protocol,
            expect_source_sha256=args.expect_source_sha256,
            prompt_payload=args.prompt_payload,
            expect_prompt_sha256=args.expect_prompt_sha256,
        )
        print(
            json.dumps(
                result.summary(artifact_path=worker_evidence_path(args.model_call_id)),
                sort_keys=True,
            )
        )
        # The result envelope, not the process exit code, carries the execution
        # outcome so Weft can always retrieve a preflight or evidence failure.
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"agent-execution-worker: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
