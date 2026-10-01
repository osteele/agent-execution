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


class OmpSdkPinTests(unittest.TestCase):
    def test_every_copy_of_the_sdk_pin_names_one_version(self) -> None:
        """The runtime script refuses any SDK but its own, so a pin bumped in one
        place and not the others makes every probe silent rather than failing."""
        import json
        import re

        from agent_execution import omp_execution

        package = Path(omp_execution.__file__).resolve().parent
        script = (package / "omp_sdk.ts").read_text(encoding="utf-8")
        match = re.search(r'^export const SDK_VERSION = "([^"]+)";$', script, re.MULTILINE)
        assert match is not None, "omp_sdk.ts declares no SDK_VERSION"
        manifest = json.loads((package / "omp-runtime" / "package.json").read_text())
        pinned = manifest["dependencies"]["@oh-my-pi/pi-coding-agent"]
        self.assertEqual(
            {omp_execution.OMP_SDK_VERSION, match.group(1), pinned},
            {omp_execution.OMP_SDK_VERSION},
        )
