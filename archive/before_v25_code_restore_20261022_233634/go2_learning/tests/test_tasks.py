"""任务分离检查：配置隔离、环境分派、旧模型与任务日志兼容。"""

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from go2_learning.tasks import TASKS, get_task
from go2_learning.training.checkpoints import (
    resolve_checkpoint, resolve_resume_checkpoint, validate_checkpoint_task,
)
from go2_learning.tests.test_reset_settling import load_env_class, Scene, Sim
from go2_learning.training.history import inherited_history, make_segment


class TaskTests(unittest.TestCase):
    def test_task_instances_do_not_share_mutable_configuration(self):
        first, second = get_task(), get_task()
        first.cfg.rewards.alive = 12345.
        first.cfg.agent["actor"]["hidden_dims"][0] = 12345
        first.cfg.commands.forward_range = (1., 2.)
        self.assertNotEqual(first.cfg.rewards.alive, second.cfg.rewards.alive)
        self.assertNotEqual(first.cfg.agent["actor"], second.cfg.agent["actor"])
        self.assertEqual(second.cfg.commands.forward_range, (0., 0.))

    def test_generic_env_calls_selected_task_for_reward_and_failures(self):
        task = get_task()
        task.cfg.environment.episode_seconds = .2
        task.cfg.environment.reset_settle_seconds = 0.
        called = []

        def rewards(state, actions, failed, dt):
            called.append(actions.shape[0])
            return {"custom_task_reward": torch.full_like(actions[:, 0], 7.)}

        task.reward_components = rewards
        task.failures = lambda state: torch.zeros_like(state["height"], dtype=torch.bool)
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        scene.robot.data.projected_gravity_b.torch[:] = torch.tensor([1., 0., 0.])
        _, reward, done, extras = env.step(env.standing_actions)
        torch.testing.assert_close(reward, torch.full((3,), 7.))
        self.assertEqual(done.tolist(), [0, 0, 0])  # 通用环境没有再硬编码站立倾斜失败。
        self.assertEqual(env.max_episode_length, 6)
        self.assertEqual(called, [3])
        self.assertEqual(extras["log"]["Reward/custom_task_reward"], 7.)

    def test_standing_failure_thresholds_match_old_boundaries(self):
        task = get_task()
        state = {"height": torch.tensor([.18, .179, .3, .3]),
                 "gravity": torch.tensor([[0., 0., -.5], [0., 0., -1.], [0., 0., -.499], [0., 0., -1.]])}
        self.assertEqual(task.failures(state).tolist(), [False, True, True, False])

    def test_playback_can_ignore_height_while_preserving_tilt_and_timeout(self):
        for name in ("standing", "running"):
            task = get_task(name)
            scene = Scene()
            env = load_env_class()(Sim(scene), scene, task=task)
            env.episode_length_buf[:] = env.max_episode_length
            env.episode_length_buf[2] -= 1
            state = {"height": torch.tensor([.01, .3, .01]),
                     "gravity": torch.tensor([[0., 0., -1.], [1., 0., 0.], [0., 0., -1.]])}
            failed, timeout = env._compute_termination(state)
            self.assertEqual(failed.tolist(), [True, True, True])
            self.assertFalse(timeout.any())
            env.height_termination_enabled = False
            failed, timeout = env._compute_termination(state)
            self.assertEqual(failed.tolist(), [False, True, False])
            self.assertEqual(timeout.tolist(), [True, False, False])

    def test_latest_search_separates_tasks_and_supports_flat_legacy_logs(self):
        task = get_task()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [
                (root / "old_run", {"task_version": task.version}, 10),
                (root / "standing/new_run", {"task_name": "standing", "task_version": task.version}, 20),
                (root / "future/new_run", {"task_name": "future", "task_version": task.version}, 30),
                (root / "standing/wrong_version", {"task_name": "standing", "task_version": "other"}, 40),
            ]
            for folder, cfg, timestamp in candidates:
                folder.mkdir(parents=True)
                path = folder / "latest.pt"
                path.touch()
                os.utime(path, (timestamp, timestamp))
                path.with_name("run.json").write_text(json.dumps(cfg))
            self.assertEqual(resolve_checkpoint("latest", "standing", root), candidates[1][0] / "latest.pt")
            (candidates[1][0] / "latest.pt").unlink()
            self.assertEqual(resolve_checkpoint("latest", "standing", root), candidates[0][0] / "latest.pt")
            with self.assertRaises(ValueError):
                validate_checkpoint_task(candidates[2][1], task)

    def test_playback_restores_saved_environment_and_termination_configuration(self):
        task = get_task()
        metadata = task.run_config(True)
        metadata["env_cfg"]["episode_seconds"] = 9.
        metadata["termination_cfg"]["minimum_height"] = .12
        restored = get_task()
        restored.load_playback_config(metadata)
        self.assertEqual(restored.cfg.environment.episode_seconds, 9.)
        self.assertEqual(restored.cfg.termination.minimum_height, .12)
        self.assertEqual(restored.run_config(True), metadata)

    def test_resume_seed_in_running_directory_and_prefer_later_running_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "running/standing_seed/latest.pt"
            seed.parent.mkdir(parents=True)
            seed.touch()
            seed.with_name("run.json").write_text(json.dumps(get_task("standing").run_config(False)))
            self.assertEqual(resolve_resume_checkpoint("latest", "running", root), seed)
            self.assertEqual(resolve_resume_checkpoint(str(seed.parent), "running", root), seed.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint("latest", "running", root)  # 播放不把站立种子当移动策略。
            native = root / "running/moving/latest.pt"
            native.parent.mkdir(parents=True)
            native.touch()
            native.with_name("run.json").write_text(json.dumps(get_task("running").run_config(False)))
            os.utime(native, (10, 10))
            self.assertEqual(resolve_resume_checkpoint("latest", "running", root), native)
            self.assertEqual(resolve_resume_checkpoint("latest", "running", root, "standing"), seed)

    def test_standing_latest_ignores_its_copy_in_running_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scope, timestamp in (("standing", 10), ("running", 20)):
                path = root / scope / "same_run/latest.pt"
                path.parent.mkdir(parents=True)
                path.touch()
                path.with_name("run.json").write_text(json.dumps(get_task("standing").run_config(False)))
                os.utime(path, (timestamp, timestamp))
            self.assertEqual(resolve_checkpoint("latest", "standing", root), root / "standing/same_run/latest.pt")
            old = "/old/RL_Learning/go2_learning/logs/same_run/latest.pt"
            self.assertEqual(resolve_checkpoint(old, "standing", root), (root / "standing/same_run/latest.pt").resolve())

    def test_explicit_checkpoint_remaps_old_host_path_after_reorganization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "standing/first/latest.pt"
            current.parent.mkdir(parents=True)
            current.touch()
            old = "/old/location/RL_Learning/go2_learning/logs/standing/first/latest.pt"
            self.assertEqual(resolve_checkpoint(old, "standing", root), current.resolve())

    def test_history_remaps_nested_container_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "standing/first/latest.pt"
            source.parent.mkdir(parents=True)
            torch.save({"iter": 2}, source)
            source.with_name("run.json").write_text(json.dumps({"reward_cfg": {"alive": 1}}))
            current = {"reward_cfg": {"alive": 2}, "resumed_from": "/workspace/go2_learning/logs/standing/first/latest.pt"}
            history = inherited_history(root / "standing/second/latest.pt", current, 5, None, root)
            self.assertEqual([(s["start_round"], s["end_round"]) for s in history], [(1, 3), (4, 5)])
            metadata = get_task().run_config(True)
            segment = make_segment(metadata, "test", 1, 1, "checkpoint")
            self.assertEqual(segment["task_name"], "standing")
            self.assertEqual(segment["env_cfg"], metadata["env_cfg"])

    def test_cli_task_selection_and_randomization_overrides(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
        namespace = dict(argparse=argparse, TASKS=TASKS, get_task=get_task,
                         add_launcher_args=lambda parser: None, __doc__="")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), namespace)
        for mode in ("train", "play", "view"):
            for flag in (None, "--randomization", "--no-randomization"):
                argv = ["go2", "--control", mode, "--task", "standing"] + ([flag] if flag else [])
                with patch.object(sys, "argv", argv):
                    args = namespace["parse_args"]()
                self.assertEqual(args.task, "standing")
                expected = mode == "train" if flag is None else flag == "--randomization"
                self.assertEqual(args.randomization, expected)


if __name__ == "__main__":
    unittest.main()
