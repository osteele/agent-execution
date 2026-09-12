"""Hold a child process until its parent durably records the process identity."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) < 2:
        raise ValueError("spawn gate requires a release FD and command")
    release_fd = int(arguments[0])
    command = arguments[1:]
    try:
        released = os.read(release_fd, 1)
    finally:
        os.close(release_fd)
    if released != b"1":
        raise RuntimeError("spawn gate closed before execution was admitted")
    os.execvp(command[0], command)


if __name__ == "__main__":
    raise SystemExit(main())
