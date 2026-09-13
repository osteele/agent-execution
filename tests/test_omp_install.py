"""A failed SDK install must never become an executable worker runtime."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_execution.omp_install import install_omp_runtime


class OmpRuntimeInstallTests(unittest.TestCase):
    def test_failed_sdk_import_does_not_publish_partial_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tools = directory / "bin"
            tools.mkdir()
            bun = tools / "bun"
            bun.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "if sys.argv[1] == '-e':\n"
                "    print('native dependency missing', file=sys.stderr)\n"
                "    raise SystemExit(1)\n"
            )
            bun.chmod(0o755)
            root = directory / "runtime"
            with (
                mock.patch.dict(
                    os.environ,
                    {"PATH": str(tools), "AGENT_EXECUTION_OMP_SDK_ROOT": str(root)},
                ),
                self.assertRaisesRegex(ValueError, "native dependency missing"),
            ):
                install_omp_runtime()
            self.assertFalse(root.exists())
            self.assertEqual(list(directory.glob(".runtime-*")), [])

    def test_mismatched_existing_runtime_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "runtime"
            root.mkdir()
            (root / "package.json").write_text('{"private":true}\n')
            (root / "bun.lock").write_text("existing lockfile\n")
            with (
                mock.patch.dict(os.environ, {"AGENT_EXECUTION_OMP_SDK_ROOT": str(root)}),
                mock.patch("agent_execution.omp_install.shutil.which", return_value=sys.executable),
                self.assertRaisesRegex(ValueError, "refusing to replace"),
            ):
                install_omp_runtime()
            self.assertEqual((root / "bun.lock").read_text(), "existing lockfile\n")
