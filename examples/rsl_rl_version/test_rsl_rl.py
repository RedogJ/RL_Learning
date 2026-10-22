# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""独立 CPU 测试：标准 RSL-RL、环境接口、保存加载与双杆几何。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from rsl_rl_version.agent_cfg import make_agent_cfg
from rsl_rl_version.env import CartpoleEnv
from rsl_rl_version.test_support import (
    FakeJoint, FakeScene, FakeSim, make_cfg, scene_functions, scene_module, y_quaternion,
)


def make_env(num_envs: int = 3) -> CartpoleEnv:
    cfg = make_cfg(num_envs)
    scene = FakeScene(cfg)
    return CartpoleEnv(FakeSim(scene), scene, cfg)


class RslRlTests(unittest.TestCase):
    def test_action_limit_observation_and_reward(self):
        env = make_env()
        actions = torch.tensor([[2.0], [-0.5], [0.3]])
        unchanged = actions.clone()
        obs, rewards, dones, _ = env.step(actions)
        torch.testing.assert_close(actions, unchanged)
        torch.testing.assert_close(env.scene.forces[:, 0, 0], torch.tensor([20.0, -10.0, 6.0]))
        # 假物理是 1 kg 小车的一步半隐式积分；可解析核对观测和奖励。
        expected_x = torch.tensor([20.0, -10.0, 6.0]) / 120**2
        torch.testing.assert_close(obs["policy"][:, 0], expected_x)
        torch.testing.assert_close(obs["policy"][:, 1], torch.tensor([20.0, -10.0, 6.0]) / 120)
        expected_reward = (torch.tensor([0.999, 0.99975, 0.99991]) - 0.1 * expected_x.square()) / 120
        torch.testing.assert_close(rewards, expected_reward)
        self.assertFalse(dones.any())

    def test_angles_upper_pole_reward_and_failure(self):
        env = make_env(2)
        env.scene["ball"].data.root_link_quat_w.torch[:] = y_quaternion(torch.tensor([0.2, -0.3]))
        env.scene["ball2"].data.root_link_quat_w.torch[:] = y_quaternion(torch.tensor([-0.1, 0.6]))
        env.scene["ball"].data.root_ang_vel_w.torch[:, 1] = torch.tensor([1.0, 2.0])
        expected = torch.tensor([[0., 0., 0.2, 1., -0.1, 0.], [0., 0., -0.3, 2., 0.6, 0.]])
        torch.testing.assert_close(env.get_observations()["policy"], expected)
        _, rewards, dones, _ = env.step(torch.zeros(2, 1))
        torch.testing.assert_close(rewards, torch.tensor([0.95, 0.55]) / 120)
        self.assertEqual(dones.tolist(), [0, 1])

    def test_selective_reset_and_timeout_flags(self):
        env = make_env()
        previous = env.get_observations()
        snapshot = previous["policy"].clone()
        env.scene["ball"].data.root_link_quat_w.torch[0] = y_quaternion(torch.tensor([0.8]))[0]
        env.episode_length_buf[1] = env.max_episode_length - 1
        obs, _, dones, extras = env.step(torch.ones(3, 1) * 0.5)
        self.assertEqual(dones.tolist(), [1, 1, 0])
        self.assertEqual(extras["time_outs"].tolist(), [False, True, False])
        self.assertEqual(env.episode_length_buf.tolist(), [0, 0, 1])
        torch.testing.assert_close(obs["policy"][:2], torch.zeros(2, 6))
        self.assertGreater(obs["policy"][2, 0].item(), 0.0)
        torch.testing.assert_close(previous["policy"], snapshot)
        for name in ("cart", "ball", "ball2"):
            asset = env.scene[name]
            expected = asset.data.default_root_pose.torch[:2, :3] + env.scene.env_origins[:2]
            torch.testing.assert_close(asset.data.root_link_pos_w.torch[:2], expected)

    def test_rsl_timeout_bootstrap(self):
        env = make_env()
        cfg = make_agent_cfg()
        cfg["num_steps_per_env"] = 2
        runner = OnPolicyRunner(env, cfg, device="cpu")
        obs = env.get_observations()
        env.episode_length_buf[1] = env.max_episode_length - 1
        env.scene["ball"].data.root_link_quat_w.torch[0] = y_quaternion(torch.tensor([0.8]))[0]
        with torch.inference_mode():
            actions = runner.alg.act(obs)
            values = runner.alg.transition.values.clone().squeeze(-1)
            next_obs, rewards, dones, extras = env.step(actions)
            runner.alg.process_env_step(next_obs, rewards, dones, extras)
        expected = rewards + cfg["algorithm"]["gamma"] * values * extras["time_outs"]
        torch.testing.assert_close(runner.alg.storage.rewards[0, :, 0], expected)
        self.assertEqual(dones.tolist(), [1, 1, 0])

    def test_real_runner_update_save_load_and_inference(self):
        torch.manual_seed(42)
        env = make_env(4)
        env.max_episode_length = 3
        cfg = make_agent_cfg()
        cfg["num_steps_per_env"] = 8
        with tempfile.TemporaryDirectory() as directory:
            runner = OnPolicyRunner(env, cfg, log_dir=directory, device="cpu")
            before = [p.detach().clone() for p in runner.alg.actor.parameters()]
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
            self.assertEqual(runner.logger.tot_timesteps, 64)
            after = list(runner.alg.actor.parameters())
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))
            self.assertTrue(all(torch.isfinite(p).all() for p in after))
            checkpoint = Path(directory) / "latest.pt"
            runner.save(str(checkpoint))
            self.assertTrue((Path(directory) / "model_1.pt").exists())
            self.assertTrue(list(Path(directory).glob("events.out.tfevents.*")))
            restored = OnPolicyRunner(make_env(1), make_agent_cfg(), device="cpu")
            restored.load(str(checkpoint), map_location="cpu")
            obs = TensorDict({"policy": torch.ones(1, 6) * 0.1}, batch_size=[1])
            with torch.inference_mode():
                expected = runner.get_inference_policy()(obs)
                actual = restored.get_inference_policy()(obs)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.shape, (1, 1))

    def test_geometry_scales_with_both_lengths(self):
        cfg = make_cfg()
        cfg.cart.init_state.pos = (1.0, -2.0, 2.0)
        cfg.cart.spawn.size = (0.4, 0.3, 0.4)
        cfg.ball.spawn.size = (0.08, 0.08, 0.8)
        cfg.ball2.spawn.size = (0.08, 0.08, 0.6)
        scene_module.configure_cartpole_scene_cfg(cfg)
        self.assertEqual(cfg.ball.spawn.parent_body, "Cart")
        self.assertEqual(cfg.ball2.spawn.parent_body, "Ball")
        self.assertAlmostEqual(cfg.ball.spawn.parent_anchor_z, 0.2)
        self.assertAlmostEqual(cfg.ball2.spawn.parent_anchor_z, 0.4)
        self.assertAlmostEqual(cfg.ball.init_state.pos[2], 2.6)
        self.assertAlmostEqual(cfg.ball2.init_state.pos[2], 3.3)
        # World-space anchors must coincide at 2.2 m and 3.0 m respectively.
        self.assertAlmostEqual(cfg.ball.init_state.pos[2] - 0.4, 2.2)
        self.assertAlmostEqual(cfg.ball2.init_state.pos[2] - 0.3, 3.0)
        self.assertEqual(cfg.ball2.init_state.pos[:2], (1.0, -2.0))

    def test_cloned_joints_use_own_environment(self):
        env_paths = ["/World/envs/env_0", "/World/envs/env_1"]
        joints = {
            f"{env_path}/{body}/{joint}": FakeJoint()
            for env_path in env_paths
            for body, joint in (("Cart", "SlideJoint"), ("Ball", "HingeJoint"), ("Ball2", "HingeJoint"))
        }
        positions = {f"{path}/Cart": (index * 2.0, 0.0, 2.6) for index, path in enumerate(env_paths)}
        stage = SimpleNamespace(joints=joints, GetPrimAtPath=lambda path: path)
        get_joint = SimpleNamespace(Get=lambda current_stage, path: current_stage.joints.get(path))
        bindings = {
            "sim_utils": SimpleNamespace(get_current_stage=lambda: stage),
            "UsdPhysics": SimpleNamespace(PrismaticJoint=get_joint, RevoluteJoint=get_joint),
            "UsdGeom": SimpleNamespace(XformCache=lambda: SimpleNamespace(
                GetLocalToWorldTransform=lambda prim: SimpleNamespace(Transform=lambda point: positions[prim])
            )),
            "Gf": SimpleNamespace(Vec3f=lambda *args: args, Vec3d=lambda *args: args),
            "Sdf": SimpleNamespace(Path=lambda path: path),
        }
        with patch.dict(scene_functions, bindings):
            scene_module.configure_cartpole_joints(SimpleNamespace(env_prim_paths=env_paths))
        for index, path in enumerate(env_paths):
            slide = joints[f"{path}/Cart/SlideJoint"].values
            self.assertEqual(slide["anchor0"], (index * 2.0, 0.0, 2.6))
            for parent, child in (("Cart", "Ball"), ("Ball", "Ball2")):
                joint = joints[f"{path}/{child}/HingeJoint"].values
                self.assertEqual(joint["body0"], [f"{path}/{parent}"])
                self.assertEqual(joint["body1"], [f"{path}/{child}"])
                self.assertFalse(joint["collision"])



if __name__ == "__main__":
    unittest.main()
