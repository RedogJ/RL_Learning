# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks for decision timing and PPO, without launching Isaac Sim."""

import copy
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from networks import PolicyNetwork, ValueNetwork
from ppo import PPOTrainer, compute_gae
from rollout import collect_rollout

# Replace only the USD state reader; exercise the actual environment implementation.
with patch.dict(sys.modules, {"scene": SimpleNamespace(
    read_cartpole_state=lambda scene, cfg: scene.state.unbind(dim=-1)
)}):
    from env import CartpoleEnv


class FakeAsset:
    def __init__(self, scene):
        self.scene = scene
        pose = torch.zeros(scene.state.shape[0], 7)
        pose[:, 3] = 1.0
        self.data = SimpleNamespace(
            default_root_pose=SimpleNamespace(torch=pose),
            default_root_vel=SimpleNamespace(torch=torch.zeros(scene.state.shape[0], 6)),
        )
        self.instantaneous_wrench_composer = self

    def set_forces_and_torques_index(self, forces, is_global):
        self.scene.forces = forces.clone()

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        pass

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        pass


class FakeScene:
    def __init__(self, num_envs):
        self.state = torch.zeros(num_envs, 4)
        self.forces = torch.zeros(num_envs, 1, 3)
        self.env_origins = torch.zeros(num_envs, 3)
        self.assets = {name: FakeAsset(self) for name in ("cart", "ball")}

    def __getitem__(self, name):
        return self.assets[name]

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass

    def reset(self, env_ids):
        self.state[env_ids] = 0.0


class FakeSim:
    device = "cpu"

    def __init__(self, scene):
        self.scene = scene
        self.on_step = None

    def get_physics_dt(self):
        return 1 / 120

    def step(self, render):
        if self.on_step is not None:
            self.on_step()
        else:
            # Small deterministic dynamics for testing rollout/update plumbing only.
            state = self.scene.state
            force = self.scene.forces[:, 0, 0]
            state[:, 1] += force * 0.01
            state[:, 0] += state[:, 1] / 120
            state[:, 3] += 0.1 + force * 0.002
            state[:, 2] += state[:, 3] / 120


def make_env(num_envs=3, history_steps=5):
    scene = FakeScene(num_envs)
    sim = FakeSim(scene)
    env = CartpoleEnv(sim, scene, SimpleNamespace(num_envs=num_envs), history_steps)
    env.reset()
    return env


class DeepONetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_query_time_and_cached_history(self):
        policy = PolicyNetwork(history_steps=3, control_steps=5)
        history = torch.randn(4, 3, 4)
        cached = policy.encode_history(history)
        self.assertEqual(policy(history).shape, (4, 5, 1))
        torch.testing.assert_close(policy(history), policy.evaluate_control(cached, policy.control_times))
        query_times = torch.tensor([0.0, 0.003, 0.02])
        self.assertEqual(policy.evaluate_control(cached, query_times).shape, (4, 3, 1))
        # The smooth mean can be queried off the physical-step grid.
        center = policy.evaluate_control(cached, [0.01])
        neighbor = policy.evaluate_control(cached, [0.010001])
        self.assertLess((center - neighbor).abs().max().item(), 0.001)
        with self.assertRaises(ValueError):
            policy(torch.zeros(4, 4))

    def test_history_shift_and_selective_reset(self):
        env = make_env(2, history_steps=3)
        env.sim.on_step = lambda: env.scene.state.add_(1.0)
        env.step(torch.zeros(2, 1))
        env.step(torch.zeros(2, 1))
        expected = torch.tensor([0.0, 1.0, 2.0]).repeat(4, 1).T
        torch.testing.assert_close(env.get_history()[1], expected)
        env.reset(torch.tensor([0]))
        torch.testing.assert_close(env.get_history()[0], torch.zeros(3, 4))
        torch.testing.assert_close(env.get_history()[1], expected)
        # A caller changing its snapshot cannot corrupt the internal buffer.
        snapshot = env.get_history()
        snapshot.zero_()
        torch.testing.assert_close(env.get_history()[1], expected)

    def test_partial_chunk_masks_reward_and_terminal_history(self):
        env = make_env(3)
        env.episode_steps[1] = env.max_episode_steps - 3
        counter = 0

        def advance():
            nonlocal counter
            counter += 1
            env.scene.state[:, 0] = counter * 0.1
            env.scene.state[0, 2] = 0.6  # Env 0 fails on the first physical step.

        env.sim.on_step = advance
        observation, rewards, terminated, truncated, info = env.step_sequence(
            torch.zeros(3, 5, 1), gamma=0.5
        )
        self.assertEqual(info["durations"].tolist(), [1, 3, 5])
        self.assertEqual(info["action_masks"].tolist(), [
            [True, False, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, True],
        ])
        self.assertEqual(terminated.tolist(), [True, False, False])
        self.assertEqual(truncated.tolist(), [False, True, False])
        torch.testing.assert_close(observation[:, 0], torch.tensor([0.1, 0.3, 0.5]))
        torch.testing.assert_close(env.get_history()[:, -1], observation)
        expected = torch.tensor([
            (1 - 0.6**2 - 0.1 * 0.1**2) / 120,
            sum(0.5**i * (1 - 0.1 * ((i + 1) * 0.1)**2) / 120 for i in range(3)),
            sum(0.5**i * (1 - 0.1 * ((i + 1) * 0.1)**2) / 120 for i in range(5)),
        ])
        torch.testing.assert_close(rewards, expected)
        self.assertEqual(env.episode_steps.tolist(), [1, env.max_episode_steps, 5])

    def test_duration_discount_and_episode_boundaries(self):
        rollout = {
            "rewards": torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
            "values": torch.tensor([[1., 1., 1.], [2., 2., 2.]]),
            "next_values": torch.tensor([[2., 8., 4.], [3., 9., 5.]]),
            "durations": torch.tensor([[2, 1, 2], [1, 2, 1]]),
            "terminated": torch.tensor([[False, True, False], [False, False, False]]),
            "truncated": torch.tensor([[False, False, True], [False, True, False]]),
        }
        advantages, returns = compute_gae(rollout, gamma=0.5, gae_lambda=0.8)
        expected = torch.tensor([[1.06, 1., 3.], [3.5, 5.25, 6.5]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(returns, expected + rollout["values"])

    def test_rollout_bootstrap_before_reset_and_masked_ppo(self):
        env = make_env(4, history_steps=3)
        # Env 0 reaches its time limit after two of five actions.
        env.episode_steps[0] = env.max_episode_steps - 2
        policy = PolicyNetwork(history_steps=3)
        critic = ValueNetwork(history_steps=3)
        rollout, history = collect_rollout(env, policy, critic, env.get_history(), num_steps=3)
        self.assertEqual(rollout["durations"][0, 0].item(), 2)
        torch.testing.assert_close(rollout["observations"][1, 0], torch.zeros(3, 4))
        # Terminal history contains two observations before resetting the buffer.
        terminal_history = torch.zeros(1, 3, 4)
        second_state = rollout["actions"][0, 0, :2, 0].clamp(-1, 1) * env.force_limit
        speed = 0.0
        position = 0.0
        omega = 0.0
        theta = 0.0
        for index, force in enumerate(second_state.tolist(), start=1):
            speed += force * 0.01
            position += speed / 120
            omega += 0.1 + force * 0.002
            theta += omega / 120
            terminal_history[0, index] = torch.tensor([position, speed, theta, omega])
        with torch.no_grad():
            torch.testing.assert_close(rollout["next_values"][0, 0], critic(terminal_history)[0])

        advantages, returns = compute_gae(rollout)
        altered = {name: value.clone() for name, value in rollout.items()}
        altered["actions"][~altered["action_masks"]] = 100.0
        comparison_policy = copy.deepcopy(policy)
        comparison_critic = copy.deepcopy(critic)
        before = {name: value.clone() for name, value in policy.named_parameters()}
        trainer = PPOTrainer(policy, critic)
        comparison_trainer = PPOTrainer(comparison_policy, comparison_critic)
        losses = trainer.update(rollout, advantages, returns)
        altered_losses = comparison_trainer.update(altered, advantages, returns)
        self.assertTrue(all(math.isfinite(value) for value in losses.values()))
        self.assertEqual(losses, altered_losses)
        for actual, expected in zip(policy.parameters(), comparison_policy.parameters()):
            torch.testing.assert_close(actual, expected)
        for prefix in ("branch", "trunk"):
            self.assertTrue(any(
                not torch.equal(parameter, before[name])
                for name, parameter in policy.named_parameters() if name.startswith(prefix)
            ))
        self.assertEqual(history.shape, (4, 3, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
