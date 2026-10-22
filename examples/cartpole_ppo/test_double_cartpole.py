# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check double-pole state, resets and PPO on CPU, without launching Kit/PhysX."""

import ast
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from networks import PolicyNetwork, ValueNetwork
from ppo import PPOTrainer, compute_gae
from rollout import collect_rollout

PROJECT_DIR = Path(__file__).resolve().parent
math_spec = importlib.util.spec_from_file_location(
    "double_cartpole_math", PROJECT_DIR.parents[1] / "third_party/IsaacLab/source/isaaclab/isaaclab/utils/math.py"
)
math_utils = importlib.util.module_from_spec(math_spec)
math_spec.loader.exec_module(math_utils)

# Load the actual scene functions without constructing USD assets. Runtime physics
# still requires the Docker smoke test; these checks exercise the data contracts.
scene_path = PROJECT_DIR / "scene.py"
scene_ast = ast.parse(scene_path.read_text(), filename=str(scene_path))
functions = [node for node in scene_ast.body if isinstance(node, ast.FunctionDef)]
for function in functions:
    function.decorator_list = []
isolated_ast = ast.Module(
    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
    type_ignores=[],
)
scene_functions = {"torch": torch, "quat_apply": math_utils.quat_apply}
exec(compile(ast.fix_missing_locations(isolated_ast), str(scene_path), "exec"), scene_functions)
scene_module = SimpleNamespace(**scene_functions)
with patch.dict(sys.modules, {"scene": scene_module}):
    from env import CartpoleEnv


def make_cfg(num_envs=2):
    def asset_cfg(size, pos):
        return SimpleNamespace(spawn=SimpleNamespace(size=size), init_state=SimpleNamespace(pos=pos))

    cfg = SimpleNamespace(
        num_envs=num_envs,
        cart=asset_cfg((0.4, 0.3, 0.2), (0.0, 0.0, 2.6)),
        ball=asset_cfg((0.08, 0.08, 1.0), (0.0, 0.0, 0.0)),
        ball2=asset_cfg((0.08, 0.08, 1.0), (0.0, 0.0, 0.0)),
    )
    scene_module.configure_cartpole_scene_cfg(cfg)
    return cfg


def y_quaternion(angles):
    result = torch.zeros(len(angles), 4)
    result[:, 1] = torch.sin(angles / 2)
    result[:, 3] = torch.cos(angles / 2)
    return result


class FakeAsset:
    def __init__(self, scene, cfg):
        self.scene = scene
        pose = torch.zeros(len(scene.env_origins), 7)
        pose[:, :3] = torch.tensor(cfg.init_state.pos)
        pose[:, 6] = 1.0  # xyzw
        world_pose = pose.clone()
        world_pose[:, :3] += scene.env_origins
        self.data = SimpleNamespace(
            default_root_pose=SimpleNamespace(torch=pose),
            default_root_vel=SimpleNamespace(torch=torch.zeros(len(pose), 6)),
            root_link_pos_w=SimpleNamespace(torch=world_pose[:, :3]),
            root_link_quat_w=SimpleNamespace(torch=world_pose[:, 3:]),
            root_link_lin_vel_w=SimpleNamespace(torch=torch.zeros(len(pose), 3)),
            root_ang_vel_w=SimpleNamespace(torch=torch.zeros(len(pose), 3)),
        )
        self.instantaneous_wrench_composer = self

    def set_forces_and_torques_index(self, forces, is_global):
        self.scene.forces = forces.clone()

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.data.root_link_pos_w.torch[env_ids] = root_pose[:, :3]
        self.data.root_link_quat_w.torch[env_ids] = root_pose[:, 3:]

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        self.data.root_link_lin_vel_w.torch[env_ids] = root_velocity[:, :3]
        self.data.root_ang_vel_w.torch[env_ids] = root_velocity[:, 3:]


class FakeScene:
    def __init__(self, cfg):
        self.env_origins = torch.zeros(cfg.num_envs, 3)
        self.env_origins[:, 0] = torch.arange(cfg.num_envs) * 2.0
        self.forces = torch.zeros(cfg.num_envs, 1, 3)
        self.assets = {name: FakeAsset(self, getattr(cfg, name)) for name in ("cart", "ball", "ball2")}

    def __getitem__(self, name):
        return self.assets[name]

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass

    def reset(self, env_ids):
        pass


class FakeSim:
    device = "cpu"

    def __init__(self, scene):
        self.scene = scene

    def get_physics_dt(self):
        return 1 / 120

    def step(self, render):
        # This fixture checks data flow, not double-pendulum dynamics.
        cart = self.scene["cart"].data
        cart.root_link_lin_vel_w.torch[:, 0] += self.scene.forces[:, 0, 0] / 120
        cart.root_link_pos_w.torch[:, 0] += cart.root_link_lin_vel_w.torch[:, 0] / 120


class FakeJoint:
    def __init__(self):
        self.values = {}

    def CreateBody0Rel(self):
        return SimpleNamespace(SetTargets=lambda targets: self.values.update(body0=targets))

    def CreateBody1Rel(self):
        return SimpleNamespace(SetTargets=lambda targets: self.values.update(body1=targets))

    def CreateLocalPos0Attr(self, value):
        self.values["anchor0"] = value

    def CreateCollisionEnabledAttr(self, value):
        self.values["collision"] = value


class DoubleCartpoleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = make_cfg()
        self.scene = FakeScene(self.cfg)
        self.env = CartpoleEnv(FakeSim(self.scene), self.scene, self.cfg)
        self.env.reset()

    def test_geometry_scales_with_both_lengths(self):
        self.cfg.cart.init_state.pos = (1.0, -2.0, 2.0)
        self.cfg.cart.spawn.size = (0.4, 0.3, 0.4)
        self.cfg.ball.spawn.size = (0.08, 0.08, 0.8)
        self.cfg.ball2.spawn.size = (0.08, 0.08, 0.6)
        scene_module.configure_cartpole_scene_cfg(self.cfg)
        self.assertEqual(self.cfg.ball.spawn.parent_body, "Cart")
        self.assertEqual(self.cfg.ball2.spawn.parent_body, "Ball")
        self.assertAlmostEqual(self.cfg.ball.spawn.parent_anchor_z, 0.2)
        self.assertAlmostEqual(self.cfg.ball2.spawn.parent_anchor_z, 0.4)
        self.assertAlmostEqual(self.cfg.ball.init_state.pos[2], 2.6)
        self.assertAlmostEqual(self.cfg.ball2.init_state.pos[2], 3.3)
        # World-space anchors must coincide at 2.2 m and 3.0 m respectively.
        self.assertAlmostEqual(self.cfg.ball.init_state.pos[2] - 0.4, 2.2)
        self.assertAlmostEqual(self.cfg.ball2.init_state.pos[2] - 0.3, 3.0)
        self.assertEqual(self.cfg.ball2.init_state.pos[:2], (1.0, -2.0))

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

    def test_absolute_angles_and_environment_offsets(self):
        self.scene["cart"].data.root_link_pos_w.torch[:, 0] += torch.tensor([0.3, -0.2])
        self.scene["ball"].data.root_link_quat_w.torch[:] = y_quaternion(torch.tensor([0.2, -0.3]))
        self.scene["ball2"].data.root_link_quat_w.torch[:] = y_quaternion(torch.tensor([-0.1, 0.4]))
        self.scene["ball"].data.root_ang_vel_w.torch[:, 1] = torch.tensor([1.0, 2.0])
        self.scene["ball2"].data.root_ang_vel_w.torch[:, 1] = torch.tensor([-1.0, -2.0])
        expected = torch.tensor([[0.3, 0.0, 0.2, 1.0, -0.1, -1.0], [-0.2, 0.0, -0.3, 2.0, 0.4, -2.0]])
        torch.testing.assert_close(self.env.get_observation(), expected)

    def test_reward_and_failure_include_upper_pole(self):
        states = torch.tensor([[1., 0., 0.2, 0., 0.3, 0.], [0., 0., 0., 0., 0.6, 0.]])
        actions = torch.tensor([[0.5], [0.]])
        torch.testing.assert_close(
            self.env.compute_reward(states, actions), torch.tensor([0.76975 / 120, 0.64 / 120])
        )
        self.assertEqual(self.env.compute_terminated(states).tolist(), [False, True])
        states[0, 2] = -0.6
        self.assertTrue(self.env.compute_terminated(states)[0].item())
        states[0, 2] = 0.0
        states[0, 0] = 1.9
        self.assertTrue(self.env.compute_terminated(states)[0].item())

    def test_selective_reset_restores_all_three_bodies(self):
        for asset in self.scene.assets.values():
            asset.data.root_link_pos_w.torch += 0.2
            asset.data.root_link_quat_w.torch[:] = y_quaternion(torch.tensor([0.2, 0.2]))
            asset.data.root_ang_vel_w.torch[:] = 1.0
        self.env.episode_steps[:] = 20
        self.env.forces[:] = 10.0
        before = self.env.get_observation().clone()
        after = self.env.reset(torch.tensor([1]))
        torch.testing.assert_close(after[0], before[0])
        torch.testing.assert_close(after[1], torch.zeros(6))
        self.assertEqual(self.env.episode_steps.tolist(), [20, 0])
        torch.testing.assert_close(self.env.forces[1], torch.zeros(1, 3))
        for name in ("cart", "ball", "ball2"):
            torch.testing.assert_close(
                self.scene[name].data.root_link_pos_w.torch[1],
                torch.tensor(getattr(self.cfg, name).init_state.pos) + self.scene.env_origins[1],
            )

    def test_six_dimensional_rollout_and_ppo_update(self):
        self.env.max_episode_steps = 2
        policy, critic = PolicyNetwork(), ValueNetwork()
        rollout, observation = collect_rollout(self.env, policy, critic, self.env.get_observation(), num_steps=4)
        self.assertEqual(rollout["observations"].shape, (4, 2, 6))
        self.assertEqual(rollout["actions"].shape, (4, 2, 1))
        self.assertEqual(observation.shape, (2, 6))
        self.assertEqual(rollout["truncated"].sum().item(), 4)
        before = policy.net[0].weight.detach().clone()
        advantages, returns = compute_gae(rollout)
        losses = PPOTrainer(policy, critic).update(rollout, advantages, returns)
        self.assertTrue(all(math.isfinite(value) for value in losses.values()))
        self.assertFalse(torch.equal(before, policy.net[0].weight))


if __name__ == "__main__":
    unittest.main(verbosity=2)
