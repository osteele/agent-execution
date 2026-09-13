"""Provision the pinned OMP SDK only when explicitly requested.

A versioned, locked runtime is separate from the Python tool environment so
Weft snapshots and installed commands resolve the same dependency. Publishing
only a complete directory keeps an interrupted install from becoming usable.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from agent_execution.omp_execution import omp_sdk_root


def _check_sdk(root: Path, bun: str) -> None:
    result = subprocess.run(
        [bun, "-e", 'await import("@oh-my-pi/pi-coding-agent");'],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError(f"OMP SDK import failed: {(result.stderr or result.stdout).strip()}")


def install_omp_runtime() -> None:
    """Install the source revision's lockfile without replacing a live runtime."""
    bun = shutil.which("bun")
    if bun is None:
        raise ValueError("OMP runtime installation requires Bun >=1.3.14 on PATH")
    source = Path(__file__).with_name("omp-runtime")
    files = {name: (source / name).read_bytes() for name in ("package.json", "bun.lock")}
    root = omp_sdk_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if root.exists():
            if any(
                not (root / name).is_file() or (root / name).read_bytes() != data
                for name, data in files.items()
            ):
                raise ValueError(
                    f"OMP runtime at {root} does not match the pinned lockfile; "
                    "refusing to replace it"
                )
            _check_sdk(root, bun)
            return
        stage = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
        try:
            for name, data in files.items():
                (stage / name).write_bytes(data)
            result = subprocess.run(
                [bun, "install", "--frozen-lockfile", "--ignore-scripts"],
                cwd=stage,
                text=True,
                capture_output=True,
                check=False,
                timeout=300,
            )
            if result.returncode:
                raise ValueError(
                    f"OMP runtime installation failed: {(result.stderr or result.stdout).strip()}"
                )
            _check_sdk(stage, bun)
            os.rename(stage, root)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


if __name__ == "__main__":
    install_omp_runtime()
