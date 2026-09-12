"""Exact executable-byte identity, identical in a source tree and an installed wheel.

This is a security identity, not a numerical-comparability projection. Python and
SDK sources plus the pinned SDK dependency declarations are hashed as raw bytes.
Bytecode, wheel metadata and provisioned SDK dependencies are not source inputs.
The SDK version and runtime installation are checked separately before execution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


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
