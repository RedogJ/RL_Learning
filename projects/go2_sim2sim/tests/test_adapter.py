"""实际 MJCF / 固定检查点的接入检查，以及与 Isaac 观测公式的数值比较。"""

from types import SimpleNamespace as NS
import unittest

import glfw
import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from go2_learning.control.attitude_integral import AttitudeIntegral
from go2_learning.tasks.reward_terms import roll_pitch_from_gravity
from go2_sim2sim.config import DEFAULT_CHECKPOINT, PHYSICS_HZ, TORQUE_LIMIT
from go2_sim2sim.env import Go2Sim
from go2_sim2sim.main import Keyboard
from go2_sim2sim.observations import Observations
from go2_sim2sim.policy import Policy


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy = Policy(DEFAULT_CHECKPOINT)
        cls.env = Go2Sim(cls.policy, PHYSICS_HZ)

    def setUp(self):
        self.env.reset()

    def test_actor_matches_checkpoint_matrix_computation(self):
        state = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)["actor_state_dict"]
        x = torch.randn(64, 52, generator=torch.Generator().manual_seed(42))
        expected = x
        for i in (0, 2, 4):
            expected = F.linear(expected, state[f"mlp.{i}.weight"], state[f"mlp.{i}.bias"])
            if i != 4:
                expected = expected.tanh()
        torch.testing.assert_close(self.policy.net(x), expected, atol=1e-6, rtol=0.)

    def test_observation_layout_matches_isaac_torch_concatenation(self):
        rng = np.random.default_rng(42)
        obs = Observations(self.policy)
        state = {"linear_velocity": rng.normal(size=3), "angular_velocity": rng.normal(size=3),
                 "gravity": np.array([.1, -.2, -.97]), "q": rng.normal(size=12),
                 "dq": rng.normal(size=12), "height": .31}
        actions, commands = rng.normal(size=12), np.array([1.5, 0., 0.])
        reference = torch.cat(tuple(torch.tensor(x, dtype=torch.float32) for x in (
            state["linear_velocity"], state["angular_velocity"], state["gravity"],
            state["q"] - self.policy.default_q, .1 * state["dq"], actions, commands,
            [state["height"] - obs.target_height], np.zeros(3),
        )))
        np.testing.assert_allclose(obs.build(state, actions, commands), reference.numpy(), atol=1e-7)
        self.assertEqual(reference.shape, (52,))

    def test_integral_updates_match_original_implementation_and_clipping(self):
        numpy_obs = Observations(self.policy)
        cfg = NS(**self.policy.cfg["integral_cfg"])
        reference = AttitudeIntegral(1, "cpu", self.policy.control_dt, cfg)
        gravity = torch.tensor([[.1, -.2, -.97]])
        roll, pitch = roll_pitch_from_gravity(gravity)
        for _ in range(300):
            numpy_obs.update({"gravity": gravity[0].numpy(), "height": .35})
            reference.update(roll, pitch, torch.tensor([.35 - numpy_obs.target_height]))
        np.testing.assert_allclose(numpy_obs.integrals, reference.values[0].numpy(), atol=2e-6)
        numpy_obs.reset()
        np.testing.assert_array_equal(numpy_obs.integrals, np.zeros(3))

    def test_root_velocity_is_com_velocity_expressed_in_link_frame(self):
        env = self.env
        env.data.qpos[3:7] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        env.data.qvel[:3] = [1., 2., 3.]
        env.data.qvel[3:6] = [0., 0., 2.]
        mujoco.mj_forward(env.model, env.data)
        rotation = env.data.xmat[env.base_id].reshape(3, 3)
        omega = rotation @ np.array([0., 0., 2.])
        world_com_velocity = np.array([1., 2., 3.]) + np.cross(omega, rotation @ env.model.body_ipos[env.base_id])
        state = env.state()
        np.testing.assert_allclose(state["linear_velocity"], rotation.T @ world_com_velocity, atol=1e-10)
        np.testing.assert_allclose(state["angular_velocity"], [0., 0., 2.], atol=1e-10)

    def test_named_mapping_and_full_range_actions(self):
        env, policy = self.env, self.policy
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in env.joint_ids]
        self.assertEqual(names, policy.joint_names)
        np.testing.assert_array_equal(env.model.actuator_trnid[env.actuator_ids, 0], env.joint_ids)
        np.testing.assert_allclose(policy.target(np.full(12, -2.)), policy.limits[:, 0])
        np.testing.assert_allclose(policy.target(np.full(12, 2.)), policy.limits[:, 1])
        np.testing.assert_allclose(policy.target(policy.standing_action), policy.default_q)

    def test_settling_integrals_and_torque_limits(self):
        env = self.env
        for _ in range(9):
            env.step(np.ones(12) * 100)
            np.testing.assert_array_equal(env.last_action, self.policy.standing_action)
            np.testing.assert_array_equal(env.observations.integrals, np.zeros(3))
        self.assertTrue(env.active)
        env.step(np.ones(12) * 100)
        self.assertLessEqual(np.abs(env.last_torque).max(), TORQUE_LIMIT)
        self.assertFalse(np.all(env.observations.integrals == 0.))
        # 官方固定版本的 MJCF 被动参数应保留，不能再次被加载器清零。
        np.testing.assert_allclose(env.model.dof_damping[env.dof_ids], np.full(12, .1))
        np.testing.assert_allclose(env.model.dof_frictionloss[env.dof_ids], np.full(12, .2))
        np.testing.assert_allclose(env.model.dof_armature[env.dof_ids], np.full(12, .01))

    def test_keyboard_hold_release_focus_and_combinations(self):
        keyboard = Keyboard([0., 0., 0.], self.policy.cfg["command_cfg"])
        sampler = NS(held=set(), focused=True, bind_current_window=lambda: None, close=lambda: None)
        sampler.read = lambda: (sampler.held, sampler.focused)
        keyboard.sampler = sampler
        sampler.held = {glfw.KEY_UP}
        keyboard.press(glfw.KEY_UP)
        np.testing.assert_array_equal(keyboard.read()[0], [1.5, 0., 0.])
        # 不依赖 release 回调，实际键位清除后下一个控制周期即停车。
        sampler.held.clear()
        np.testing.assert_array_equal(keyboard.read()[0], np.zeros(3))
        sampler.held = {glfw.KEY_UP, glfw.KEY_LEFT}
        np.testing.assert_array_equal(keyboard.read()[0], [1.5, 0., .6])
        sampler.focused = False
        np.testing.assert_array_equal(keyboard.read()[0], np.zeros(3))
        sampler.focused = True
        sampler.held = {glfw.KEY_DOWN}
        np.testing.assert_array_equal(keyboard.read()[0], [-1.5, 0., 0.])
        sampler.held.add(glfw.KEY_SPACE)
        np.testing.assert_array_equal(keyboard.read()[0], np.zeros(3))
        keyboard.press(glfw.KEY_R)
        self.assertTrue(keyboard.read()[1])
        self.assertFalse(keyboard.read()[1])
        keyboard.close()
        self.assertIsNone(keyboard.sampler)

    def test_stand_forward_stop_backward_stop_in_one_episode(self):
        env = self.env
        stop_speed = None
        for step in range(420):
            t = step * self.policy.control_dt
            env.command[:] = [0. if t < 2 or 5 <= t < 8 or t >= 11 else 1.5 if t < 5 else -1.5, 0., 0.]
            action = self.policy(env.observation()) if env.active else self.policy.standing_action
            state, done = env.step(action)
            self.assertIsNone(done)
            if step == 239:
                stop_speed = np.linalg.norm(state["linear_velocity"][:2])
        self.assertLess(stop_speed, .1)
        self.assertLess(np.linalg.norm(state["linear_velocity"][:2]), .1)


if __name__ == "__main__":
    unittest.main()
