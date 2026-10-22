"""用命令替身核对 Docker 参数、挂载与路径转换，不启动 Docker 或显示窗口。"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(os.geteuid() == 0, "启动器要求普通用户")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.capture = self.folder / "docker.json"
        stubs = {
            "docker": "#!/bin/sh\nexit 0\n",
            "sudo": (
                "#!/usr/bin/env python3\nimport json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['RL_LAUNCH_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
            ),
            "xauth": (
                "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
                "if sys.argv[1] == 'nlist': print('ffff stub-cookie')\n"
                "elif sys.argv[1] == '-f': Path(sys.argv[2]).write_text(sys.stdin.read())\n"
            ),
        }
        for name, content in stubs.items():
            path = self.folder / name
            path.write_text(content)
            path.chmod(0o755)
        self.environment = {
            **os.environ,
            "PATH": f"{self.folder}:{os.environ['PATH']}",
            "RL_LAUNCH_CAPTURE": str(self.capture),
        }

    def launch(self, *args, gui=False):
        environment = self.environment.copy()
        environment.pop("DISPLAY", None)
        if gui:
            environment["DISPLAY"] = ":stub"
        subprocess.run(["bash", str(ROOT / "run.sh"), *args], cwd=self.folder,
                       env=environment, check=True, capture_output=True, text=True)
        return json.loads(self.capture.read_text())

    def test_headless_projects_mount_source_and_writable_outputs(self):
        for project, folder, entry in (
            ("go2", "projects/go2_learning", "go2_learning.main"),
            ("cartpole-rsl", "examples/rsl_rl_version", "rsl_rl_version.main"),
            ("cartpole-ppo", "examples/cartpole_ppo", "/workspace/cartpole_ppo/main.py"),
        ):
            with self.subTest(project=project):
                arguments = ("--control", "train") if project == "cartpole-ppo" else ("train",)
                argv = self.launch(project, *arguments, "--viz", "none", "--max_updates", "2")
                package = Path(folder).name
                self.assertIn(f"{ROOT / folder}:/workspace/{package}:ro", argv)
                self.assertIn(f"{ROOT / folder}/logs:/workspace/{package}/logs", argv)
                self.assertIn(entry, argv)
                self.assertIn("--pull=never", argv)
                self.assertEqual(argv[argv.index("--network") + 1], "none")
                self.assertNotIn("DISPLAY=:stub", argv)
                if project == "cartpole-ppo":
                    self.assertIn(f"{ROOT / folder}/checkpoints:/workspace/cartpole_ppo/checkpoints", argv)

    def test_go2_gui_provides_display_assets_and_source(self):
        argv = self.launch("go2", "play", gui=True)
        self.assertIn("DISPLAY=:stub", argv)
        self.assertIn("GO2_ASSET_ROOT=/workspace/assets", argv)
        self.assertIn(f"{ROOT}/assets:/workspace/assets:ro", argv)
        self.assertIn("newton_gl", argv)
        self.assertIn("--control", argv)
        self.assertIn("play", argv)

    def test_explicit_host_checkpoint_translates_in_both_argument_forms(self):
        log_root = ROOT / "projects/go2_learning/logs"
        with tempfile.TemporaryDirectory(prefix="launcher-test-", dir=log_root) as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.touch()
            mapped = f"/workspace/go2_learning/logs/{checkpoint.parent.name}/model.pt"
            argv = self.launch("go2", "train", "--resume", str(checkpoint))
            self.assertEqual(argv[argv.index("--resume") + 1], mapped)
            argv = self.launch("go2", "play", "--viz=none", f"--checkpoint={checkpoint}")
            self.assertIn(f"--checkpoint={mapped}", argv)

    def test_official_and_comparison_entries_mount_relocated_sources(self):
        for project in ("cartpole-official", "cartpole-compare"):
            with self.subTest(project=project):
                argv = self.launch(project, gui=True)
                self.assertIn(f"{ROOT}/third_party/IsaacLab/source:/workspace/isaaclab/source", argv)
                self.assertIn(f"{ROOT}/examples/cartpole_ppo:/workspace/isaaclab/scripts/my_cartpole", argv)
                self.assertIn(f"{ROOT}/examples/cartpole_compare.py:/workspace/isaaclab/scripts/compare_cartpole_policies.py:ro", argv)


if __name__ == "__main__":
    unittest.main()
