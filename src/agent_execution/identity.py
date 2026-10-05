"""Exact executable-byte identity, identical in a source tree and an installed wheel.

This is a security identity, not a numerical-comparability projection. Python and
SDK sources plus the pinned SDK dependency declarations are hashed as raw bytes.
Bytecode, wheel metadata and provisioned SDK dependencies are not source inputs.
The SDK version and runtime installation are checked separately before execution.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def worker_evidence_path(model_call_id: str) -> str:
    """Return the protocol-2 evidence location in the model-inaccessible namespace."""
    if not isinstance(model_call_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{0,127}", model_call_id
    ):
        raise ValueError(
            "model call identity must be 1–128 lowercase ASCII letters, digits, _ or -"
        )
    return f".agent-execution/results/{model_call_id}.json"


def source_worker_executable(source_digest: str) -> str:
    """Public retained worker name, portable across hosts and installation roots.

    Installers bind this name directly to a verified immutable environment, never
    through their active-release pointer. The source handshake still authenticates
    the worker; a command name alone is not identity evidence.
    """
    if not isinstance(source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise ValueError("worker source must be a lowercase SHA-256 digest")
    return f"agent-execution-worker-{source_digest}"


def source_sha256(package_root: Path | None = None) -> str:
    """Hash the shared package, without a repository or generated install marker.

    ``package_root`` is the ``agent_execution`` directory, not its project root.
    Reading every call deliberately observes edits to a source installation;
    caching would allow the identity to describe bytes no longer on disk.
    """
    root = Path(__file__).resolve().parent if package_root is None else package_root.resolve()
    paths = sorted([*root.glob("*.py"), *root.glob("*.ts")], key=lambda path: path.name)
    paths.extend(root / "omp-runtime" / name for name in ("bun.lock", "package.json"))
    if not paths or not (root / "worker.py").is_file():
        raise ValueError(f"shared execution package source is missing at {root}")
    digest = hashlib.sha256(b"agent-execution.source/v1\0")
    for path in sorted(paths, key=lambda path: path.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()
