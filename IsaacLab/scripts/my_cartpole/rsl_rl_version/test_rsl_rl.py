# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contract tests with real RSL-RL and the existing fake-physics fixtures."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_double_cartpole import CartpoleEnv, FakeScene, FakeSim, make_cfg, y_quaternion
from rsl_rl_version.agent_cfg import make_agent_cfg
from rsl_rl_version.vec_env import CartpoleVecEnv


def make_env(num_envs: int = 3) -> CartpoleEnv:
    cfg = make_cfg(num_envs)
    scene = FakeScene(cfg)
    return CartpoleEnv(FakeSim(scene), scene, cfg)


class RslRlComparisonTests(unittest.TestCase):
    def test_same_physics_inputs_and_rewards_as_original(self):
        original = make_env()
        original.reset()
        adapted = CartpoleVecEnv(make_env())
        actions = torch.tensor([[2.0], [-0.5], [0.3]])
        unchanged = actions.clone()
        for _ in range(5):
            expected, reward, _, _, _ = original.step(actions)
            observed, adapted_reward, dones, _ = adapted.step(actions)
            torch.testing.assert_close(observed["policy"], expected)
            torch.testing.assert_close(adapted_reward, reward)
            self.assertFalse(dones.any())
        torch.testing.assert_close(actions, unchanged)
        self.assertEqual(adapted.env.forces[0, 0, 0].item(), 20.0)

    def test_selective_reset_and_timeout_flags(self):
        adapted = CartpoleVecEnv(make_env())
        previous = adapted.get_observations()
        snapshot = previous["policy"].clone()
        adapted.env.pole.data.root_link_quat_w.torch[0] = y_quaternion(torch.tensor([0.8]))[0]
        adapted.env.episode_steps[1] = adapted.max_episode_length - 1
        obs, _, dones, extras = adapted.step(torch.ones(3, 1) * 0.5)
        self.assertEqual(dones.tolist(), [1, 1, 0])
        self.assertEqual(extras["time_outs"].tolist(), [False, True, False])
        self.assertEqual(adapted.episode_length_buf.tolist(), [0, 0, 1])
        torch.testing.assert_close(obs["policy"][:2], torch.zeros(2, 6))
        self.assertGreater(obs["policy"][2, 0].item(), 0.0)
        torch.testing.assert_close(previous["policy"], snapshot)
        adapted.episode_length_buf = torch.tensor([1, 2, 3])
        self.assertEqual(adapted.env.episode_steps.tolist(), [1, 2, 3])

    def test_rsl_timeout_bootstrap_matches_documented_behavior(self):
        adapted = CartpoleVecEnv(make_env())
        cfg = make_agent_cfg()
        cfg["num_steps_per_env"] = 2
        runner = OnPolicyRunner(adapted, cfg, device="cpu")
        obs = adapted.get_observations()
        adapted.env.episode_steps[1] = adapted.max_episode_length - 1
        adapted.env.pole.data.root_link_quat_w.torch[0] = y_quaternion(torch.tensor([0.8]))[0]
        with torch.inference_mode():
            actions = runner.alg.act(obs)
            values = runner.alg.transition.values.clone().squeeze(-1)
            next_obs, rewards, dones, extras = adapted.step(actions)
            runner.alg.process_env_step(next_obs, rewards, dones, extras)
        expected = rewards + cfg["algorithm"]["gamma"] * values * extras["time_outs"]
        torch.testing.assert_close(runner.alg.storage.rewards[0, :, 0], expected)
        self.assertEqual(dones.tolist(), [1, 1, 0])

    def test_real_runner_update_save_load_and_inference(self):
        torch.manual_seed(42)
        original = make_env(4)
        original.max_episode_steps = 3  # Exercise reset boundaries inside a rollout.
        adapted = CartpoleVecEnv(original)
        cfg = make_agent_cfg()
        cfg["num_steps_per_env"] = 8
        with tempfile.TemporaryDirectory() as directory:
            runner = OnPolicyRunner(adapted, cfg, log_dir=directory, device="cpu")
            before = [p.detach().clone() for p in runner.alg.actor.parameters()]
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
            self.assertEqual(runner.logger.tot_timesteps, 64)
            after = list(runner.alg.actor.parameters())
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
            self.assertTrue(all(torch.isfinite(p).all() for p in after))
            checkpoint = Path(directory) / "latest.pt"
            runner.save(str(checkpoint), infos={"task": "my_double_cartpole"})
            self.assertTrue((Path(directory) / "model_1.pt").exists())
            self.assertTrue(list(Path(directory).glob("events.out.tfevents.*")))
            restored = OnPolicyRunner(CartpoleVecEnv(make_env(1)), make_agent_cfg(), device="cpu")
            infos = restored.load(str(checkpoint), map_location="cpu")
            self.assertEqual(infos["task"], "my_double_cartpole")
            obs = TensorDict({"policy": torch.ones(1, 6) * 0.1}, batch_size=[1])
            with torch.inference_mode():
                expected = runner.get_inference_policy()(obs)
                actual = restored.get_inference_policy()(obs)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
