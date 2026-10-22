"""奖励可修改的权重续训、检查点历史、旧来源链和分支轮数检查。"""

import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import torch

from go2_learning.tasks.standing.agent_cfg import make_agent_cfg
from go2_learning.tasks.standing.integral_cfg import IntegralCfg
from go2_learning.tasks.standing.command_cfg import CommandCfg
from go2_learning.tasks.standing import TASK_VERSION
from go2_learning.tasks.standing.randomization_cfg import RandomizationCfg
from go2_learning.tasks.standing.rewards import RewardCfg
from go2_learning.tasks import checkpoint_task_name, get_task
from go2_learning.training.checkpoints import validate_checkpoint_task, validate_resume_task
from go2_learning.training.history import (
    TrainingHistoryMixin, clip_history, inherited_history, make_segment, validate_network_config,
)


class SaveBase:
    def save(self, path, infos=None):
        torch.save({"iter": self.current_learning_iteration, "infos": infos}, path)


class RecordingRunner(TrainingHistoryMixin, SaveBase):
    pass


class TrainingHistoryTests(unittest.TestCase):
    def test_clip_preserves_branch_history_without_future_rounds(self):
        history = [make_segment({"reward_cfg": {"posture_weight": 10}}, "a", 1, 300, "checkpoint"),
                   make_segment({"reward_cfg": {"posture_weight": .5}}, "b", 301, 600, "checkpoint")]
        clipped = clip_history(history, 450)
        self.assertEqual([(s["start_round"], s["end_round"]) for s in clipped], [(1, 300), (301, 450)])
        self.assertEqual(clipped[-1]["completed_updates"], 150)
        clipped[0]["reward_cfg"]["posture_weight"] = 99
        self.assertEqual(history[0]["reward_cfg"]["posture_weight"], 10)
        self.assertEqual(history[-1]["end_round"], 600)

    def test_legacy_chain_uses_checkpoint_iterations_and_original_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a", root / "b"
            a.mkdir(); b.mkdir()
            torch.save({"iter": 2, "infos": None}, a / "latest.pt")
            (a / "run.json").write_text(json.dumps({"reward_cfg": {"posture_weight": 10}}))
            config = {"reward_cfg": {"posture_weight": .5},
                      "resumed_from": "/workspace/go2_learning/logs/a/latest.pt"}
            (b / "run.json").write_text(json.dumps(config))
            torch.save({"iter": 4, "infos": None}, b / "model_4.pt")
            history = inherited_history(b / "model_4.pt", config, 5, None, root)
            self.assertEqual([(s["start_round"], s["end_round"], s["reward_cfg"]["posture_weight"])
                              for s in history], [(1, 3, 10), (4, 5, .5)])

    def test_checkpoint_history_is_independent_of_later_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"reward_cfg": {"posture_weight": .5}, "requested_updates": 4,
                      "resumed_from": "old/latest.pt"}
            parent = [make_segment({"reward_cfg": {"posture_weight": 10}}, "old", 1, 3, "checkpoint")]
            runner = RecordingRunner()
            runner.current_learning_iteration = 3
            runner.configure_history(root, config, parent)
            config["reward_cfg"]["posture_weight"] = 99
            runner.current_learning_iteration = 4
            runner.save(str(root / "model_4.pt"))
            early = torch.load(root / "model_4.pt", weights_only=False)
            runner.current_learning_iteration = 6
            runner.save(str(root / "latest.pt"))
            document = json.loads((root / "history.json").read_text())
            self.assertEqual(document["completed_rounds"], 7)
            self.assertEqual(document["segments"][-1]["reward_cfg"]["posture_weight"], .5)
            metadata = json.loads((root / "run.json").read_text())
            self.assertEqual((metadata["start_round"], metadata["end_round"], metadata["completed_updates"]), (4, 7, 4))
            self.assertEqual(metadata["planned_end_round"], 7)
            inherited = inherited_history(root / "model_4.pt", metadata, 5, early["infos"], root)
            self.assertEqual([(s["start_round"], s["end_round"]) for s in inherited], [(1, 3), (4, 5)])

    def test_missing_ancestor_is_not_assigned_an_invented_reward(self):
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings(record=True) as caught:
            root = Path(directory)
            history = inherited_history(root / "latest.pt", {"reward_cfg": {"posture_weight": 10},
                                        "resumed_from": "/missing/a/latest.pt"}, 500, None, root)
            self.assertTrue(caught)
            self.assertIsNone(history[0]["reward_cfg"])
            self.assertEqual(history[0]["end_round"], 500)

    def test_rearranged_logs_preserve_standing_history_and_save_running_phase_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "standing/old/latest.pt"
            parent.parent.mkdir(parents=True)
            old_cfg = get_task("standing").run_config(False)
            old_cfg["reward_cfg"]["posture_weight"] = 10.
            parent.with_name("run.json").write_text(json.dumps(old_cfg))
            torch.save({"iter": 2}, parent)
            copied = root / "running/old/latest.pt"
            copied.parent.mkdir(parents=True)
            copied.write_bytes(parent.read_bytes())
            copied.with_name("run.json").write_text(json.dumps(old_cfg))
            original = parent.read_bytes()
            config = get_task("running").run_config(False)
            config.update(requested_updates=4, resumed_from="/old/host/logs/old/latest.pt",
                          resumed_from_task="standing", log_relative_dir="running/new")
            # 旧扁平路径同时存在 standing/running 副本时，按真实来源任务定位。
            history = inherited_history(root / "running/new/latest.pt", config, 3, None, root)
            self.assertEqual(history[0]["task_name"], "standing")
            self.assertEqual(history[0]["reward_cfg"]["posture_weight"], 10.)
            config["resumed_from_relative"] = "standing/old/latest.pt"
            config["resumed_from"] = "/stale/path/latest.pt"
            self.assertEqual(inherited_history(root / "running/new/latest.pt", config, 3, None, root), history)
            folder = root / "running/new"
            folder.mkdir()
            runner = RecordingRunner()
            runner.current_learning_iteration = 3
            runner.configure_history(folder, config, history)
            runner.current_learning_iteration = 6
            runner.save(str(folder / "latest.pt"))
            metadata = json.loads((folder / "run.json").read_text())
            segments = json.loads((folder / "history.json").read_text())["segments"]
            self.assertEqual(metadata["log_relative_dir"], "running/new")
            self.assertEqual(metadata["resumed_from_relative"], "standing/old/latest.pt")
            self.assertEqual((metadata["start_round"], metadata["end_round"]), (4, 7))
            self.assertEqual([(s["task_name"], s["start_round"], s["end_round"])
                              for s in segments], [("standing", 1, 3), ("running", 4, 7)])
            self.assertEqual(parent.read_bytes(), original)

    def test_network_checks_ignore_initial_std_but_reject_shape_or_activation_changes(self):
        current = make_agent_cfg()
        saved = deepcopy(current)
        saved["actor"]["distribution_cfg"]["init_std"] = 8.
        validate_network_config(current, saved)
        saved["actor"]["activation"] = "relu"
        with self.assertRaises(ValueError):
            validate_network_config(current, saved)

    def test_actual_main_resume_uses_current_configs_but_play_uses_saved_configs(self):
        source = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        body = []
        for node in main.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "torch.manual_seed":
                break
            body.append(node)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            path.touch()
            saved_agent = make_agent_cfg()
            saved_agent["algorithm"]["learning_rate"] = .9
            saved_rewards = asdict(RewardCfg())
            saved_rewards["posture_weight"] = 10.
            saved = {"task_version": TASK_VERSION, "agent_cfg": saved_agent, "reward_cfg": saved_rewards,
                     "command_cfg": asdict(CommandCfg()), "integral_cfg": asdict(IntegralCfg()),
                     "num_envs": 999, "seed": 999, "randomization_cfg": asdict(RandomizationCfg())}
            path.with_name("run.json").write_text(json.dumps(saved))
            for control in ("train", "play", "old_positive_play", "legacy_play"):
                if control == "old_positive_play":
                    saved["reward_cfg"].pop("action_limit_outside_floor")
                    saved["reward_cfg"].pop("roll_velocity_weight")
                    saved["reward_cfg"].pop("pitch_velocity_weight")
                    saved["reward_cfg"].pop("undesired_contact_weight")
                    saved["reward_cfg"].pop("contact_force_threshold")
                    for field in ("horizontal_velocity_weight", "yaw_velocity_weight",
                                  "horizontal_velocity_sigma", "yaw_velocity_sigma", "foot_support_weight",
                                  "standing_linear_threshold", "standing_yaw_threshold"):
                        saved["reward_cfg"].pop(field)
                    path.with_name("run.json").write_text(json.dumps(saved))
                if control == "legacy_play":
                    saved["reward_cfg"].pop("tracking_reward_type")
                    saved["reward_cfg"].pop("only_positive_rewards")
                    path.with_name("run.json").write_text(json.dumps(saved))
                args = SimpleNamespace(task="standing", control="train" if control == "train" else "play",
                                       resume=str(path) if control == "train" else None,
                                       resume_task=None, checkpoint=str(path), randomization=False, num_envs=7, seed=8)
                namespace = dict(parse_args=lambda: args, resolve_checkpoint=lambda value, task_name: path, json=json,
                                 resolve_resume_checkpoint=lambda *args: path, LOG_ROOT=path.parent,
                                 checkpoint_task_name=checkpoint_task_name,
                                 get_task=get_task, validate_checkpoint_task=validate_checkpoint_task,
                                 validate_resume_task=validate_resume_task,
                                 TASK_VERSION=TASK_VERSION, CommandCfg=CommandCfg, RewardCfg=RewardCfg,
                                 IntegralCfg=IntegralCfg, RandomizationCfg=RandomizationCfg,
                                 make_agent_cfg=make_agent_cfg, validate_network_config=validate_network_config)
                exec(compile(ast.Module(body=body, type_ignores=[]), "main.py", "exec"), namespace)
                if control == "train":
                    self.assertEqual(namespace["reward_cfg"].posture_weight, RewardCfg().posture_weight)
                    self.assertEqual(namespace["train_cfg"]["algorithm"]["learning_rate"], make_agent_cfg()["algorithm"]["learning_rate"])
                    self.assertIsNone(namespace["randomization_cfg"])
                    self.assertEqual((args.num_envs, args.seed), (7, 8))
                else:
                    self.assertEqual(namespace["reward_cfg"].posture_weight, 10.)
                    self.assertEqual(namespace["train_cfg"]["algorithm"]["learning_rate"], .9)
                    cfg = namespace["reward_cfg"]
                    self.assertEqual(cfg.tracking_reward_type, "quadratic" if control == "legacy_play" else "exp")
                    self.assertEqual(cfg.only_positive_rewards, control != "legacy_play")
                    self.assertEqual(cfg.action_limit_outside_floor, control == "play")
                    self.assertEqual(cfg.roll_velocity_weight, RewardCfg().roll_velocity_weight if control == "play" else 0.)
                    self.assertEqual(cfg.pitch_velocity_weight, RewardCfg().pitch_velocity_weight if control == "play" else 0.)
                    self.assertEqual(cfg.undesired_contact_weight, RewardCfg().undesired_contact_weight if control == "play" else 0.)
                    self.assertEqual(cfg.horizontal_velocity_weight, RewardCfg().horizontal_velocity_weight if control == "play" else 0.)
                    self.assertEqual(cfg.yaw_velocity_weight, RewardCfg().yaw_velocity_weight if control == "play" else 0.)
                    self.assertEqual(cfg.foot_support_weight, RewardCfg().foot_support_weight if control == "play" else 0.)


if __name__ == "__main__":
    unittest.main()
