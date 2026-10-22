"""用 CPU 仿真替身检查预运行与并行重置时序，不依赖 Isaac Sim。"""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from go2_learning.control.attitude_integral import AttitudeIntegral
from go2_learning.tasks.standing.integral_cfg import IntegralCfg
from go2_learning.control.commands import VelocityCommands
from go2_learning.tasks.standing.command_cfg import CommandCfg
from go2_learning.simulation.contacts import LEGS, Go2Contacts, REQUIRED_BODY_NAMES
from go2_learning.simulation.randomization import DomainRandomizer
from go2_learning.tasks.standing.randomization_cfg import RandomizationCfg
from go2_learning.tasks.standing.rewards import RewardCfg, action_limit_penalty, roll_pitch_from_gravity, reward_components
from go2_learning.tasks import get_task


def load_env_class():
    # 使用真实环境方法；仅省去宿主机未安装的 RSL-RL / TensorDict 导入。
    tree = ast.parse((Path(__file__).resolve().parents[1] / "simulation/env.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2Env")
    cls.bases = []
    namespace = dict(
        torch=torch, math=math, AttitudeIntegral=AttitudeIntegral, IntegralCfg=IntegralCfg,
        CommandCfg=CommandCfg, VelocityCommands=VelocityCommands, RewardCfg=RewardCfg,
        roll_pitch_from_gravity=roll_pitch_from_gravity,
        action_limit_penalty=action_limit_penalty,
        reward_components=reward_components, TensorDict=lambda values, **kwargs: values,
        Go2Contacts=Go2Contacts, LEGS=LEGS,
        get_task=get_task,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "env.py", "exec"), namespace)
    return namespace["Go2Env"]


class Robot:
    num_joints = 12
    body_names = list(REQUIRED_BODY_NAMES)
    joint_names = [f"{leg}_{kind}_joint" for kind in ("hip", "thigh", "calf")
                   for leg in ("FL", "FR", "RL", "RR")]

    def __init__(self, count):
        pose = torch.tensor([0., 0., .4, 0., 0., 0., 1.]).repeat(count, 1)
        q = torch.tensor([0.] * 4 + [.8] * 4 + [-1.6] * 4).repeat(count, 1)
        limits = torch.tensor([[-1., 1.]] * 4 + [[-.5, 4.5]] * 4
                              + [[-2.72, -.838]] * 4).repeat(count, 1, 1)
        self.data = NS(**{name: NS(torch=value) for name, value in {
            "default_root_pose": pose, "root_link_pose_w": pose.clone(),
            "default_root_vel": torch.zeros(count, 6), "root_com_vel_w": torch.zeros(count, 6),
            "default_joint_pos": q, "joint_pos": q.clone(),
            "default_joint_vel": torch.zeros_like(q), "joint_vel": torch.zeros_like(q),
            "joint_pos_limits": limits, "soft_joint_pos_limits": limits,
            "joint_stiffness": torch.full_like(q, 25.), "joint_damping": torch.full_like(q, .5),
            "joint_effort_limits": torch.full_like(q, 23.7),
            "root_lin_vel_b": torch.zeros(count, 3), "root_ang_vel_b": torch.zeros(count, 3),
            "projected_gravity_b": torch.tensor([0., 0., -1.]).repeat(count, 1),
            "body_com_lin_vel_w": torch.zeros(count, len(self.body_names), 3),
            "body_link_pos_w": torch.zeros(count, len(self.body_names), 3),
        }.items()})
        self.data.root_link_pos_w = NS(torch=self.data.root_link_pose_w.torch[:, :3])
        self.data.root_link_quat_w = NS(torch=self.data.root_link_pose_w.torch[:, 3:])
        self.data.body_com_pos_w = NS(torch=self.data.root_link_pos_w.torch[:, None, :])
        self.data.body_link_quat_w = NS(torch=self.data.root_link_pose_w.torch[:, None, 3:])
        self.targets = q.clone()
        self.limit_writes = []
        self.actuators = NS(target_command=NS(set_position_index=self.set_targets))
        self.instantaneous_wrench_composer = NS(add_forces_and_torques_index=self.add_forces)
        self.push_calls = 0

    def find_joints(self, pattern):
        kind = next(name for name in ("hip", "thigh", "calf") if name in pattern)
        ids = [i for i, name in enumerate(self.joint_names) if name.endswith(f"_{kind}_joint")]
        return ids, [self.joint_names[i] for i in ids]

    def find_bodies(self, name):
        return [0], ["base"]

    def add_forces(self, **kwargs):
        self.push_calls += 1

    def set_targets(self, value, env_ids=None):
        self.targets[:] = value if env_ids is None else self.targets
        if env_ids is not None:
            self.targets[env_ids] = value

    def write_joint_position_limit_to_sim_index(self, *, limits, joint_ids, warn_limit_violation=True):
        self.data.joint_pos_limits.torch[:, joint_ids] = limits
        self.limit_writes.append((list(joint_ids), limits.clone()))

    def write_joint_stiffness_to_sim_index(self, *, stiffness, joint_ids):
        self.data.joint_stiffness.torch[:, joint_ids] = stiffness

    def write_joint_damping_to_sim_index(self, *, damping, joint_ids):
        self.data.joint_damping.torch[:, joint_ids] = damping

    def write_joint_effort_limit_to_sim_index(self, *, limits, joint_ids):
        self.data.joint_effort_limits.torch[:, joint_ids] = limits

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.data.root_link_pose_w.torch[env_ids] = root_pose

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        self.data.root_com_vel_w.torch[env_ids] = root_velocity

    def write_joint_position_to_sim_index(self, position, env_ids):
        self.data.joint_pos.torch[env_ids] = position

    def write_joint_velocity_to_sim_index(self, velocity, env_ids):
        self.data.joint_vel.torch[env_ids] = velocity


class Scene:
    num_envs = 3

    def __init__(self):
        self.robot = Robot(self.num_envs)
        self.contact_forces = NS(
            body_names=list(REQUIRED_BODY_NAMES), cfg=NS(history_length=4),
            data=NS(net_normal_forces_w_history=NS(torch=torch.zeros(self.num_envs, 4, 19, 3))),
            forces=torch.zeros(self.num_envs, 19, 3),
        )
        self.env_origins = torch.tensor([[0., 0., 0.], [3., 0., 1.], [6., 0., 2.]])
        self.writes = self.updates = 0

    def __getitem__(self, name):
        return {"robot": self.robot, "contact_forces": self.contact_forces}[name]

    def reset(self, env_ids):
        self.contact_forces.data.net_normal_forces_w_history.torch[env_ids] = 0
        self.contact_forces.forces[env_ids] = 0

    def write_data_to_sim(self):
        self.writes += 1

    def update(self, dt):
        self.updates += 1
        history = self.contact_forces.data.net_normal_forces_w_history.torch
        history[:, 1:] = history[:, :-1].clone()
        history[:, 0] = self.contact_forces.forces


class Sim:
    device = "cpu"

    def __init__(self, scene):
        self.scene, self.steps = scene, 0

    def get_physics_dt(self):
        return 1 / 120

    def step(self, render):
        self.steps += 1
        data = self.scene.robot.data
        # 用可识别的落地状态检查缓存恢复，保留非零速度以检查完整状态复制。
        data.root_link_pose_w.torch[:, 2] = self.scene.env_origins[:, 2] + .32
        data.root_com_vel_w.torch[:, 2] = -.01
        data.joint_vel.torch[:] = .02


class ResetSettlingTests(unittest.TestCase):
    def test_playback_fast_path_preserves_actions_observations_and_resets(self):
        for name in ('standing', 'running', 'upright'):
            envs=[]
            for _ in range(2):
                scene=Scene()
                env=load_env_class()(Sim(scene),scene,task=get_task(name))
                env.finish_initial_settling()
                envs.append(env)
            full, fast=envs
            generator=torch.Generator().manual_seed(7)
            for step in range(14):
                actions=torch.randn(3,12,generator=generator)
                if step==2:
                    for env in envs:env.reset(torch.tensor([1]))
                if step==12:
                    for env in envs:env.episode_length_buf[0]=env.max_episode_length-1
                a=full.step(actions,auto_reset=False)
                b=fast.step(actions,auto_reset=False,evaluate=False)
                torch.testing.assert_close(a[0]['policy'],b[0]['policy'])
                torch.testing.assert_close(a[2],b[2])
                torch.testing.assert_close(a[3]['time_outs'],b[3]['time_outs'])
                torch.testing.assert_close(full.actions,fast.actions)
                torch.testing.assert_close(full.robot.targets,fast.robot.targets)
                torch.testing.assert_close(full.attitude_integral.values,fast.attitude_integral.values)
                self.assertEqual(full.sim.steps,fast.sim.steps)
                self.assertTrue((b[1]==0).all())

    def test_sparse_render_keeps_all_physics_steps(self):
        env,sim,scene=self.make_env()
        flags=[]
        original=sim.step
        def step(render):
            flags.append(render)
            original(render)
        sim.step=step
        env.render_every_physics_steps=env.decimation
        before=sim.steps
        env.step(torch.zeros(3,12),evaluate=False)
        self.assertEqual(sim.steps-before,env.decimation)
        self.assertEqual(flags,[False]*(env.decimation-1)+[True])

    def make_env(self, randomized=False, finish=True, reward_cfg=None):
        scene = Scene()
        sim = Sim(scene)
        randomizer = (DomainRandomizer(scene.robot, 3, "cpu", 1 / 120,
                                      RandomizationCfg(), 42) if randomized else None)
        env = load_env_class()(sim, scene, randomizer, reward_cfg=reward_cfg)
        if finish:
            env.finish_initial_settling()
        return env, sim, scene

    def test_index_reuse_preserves_active_rewards_and_metrics(self):
        for task_name in ("standing", "running"):
            scene = Scene()
            env = load_env_class()(Sim(scene), scene, task=get_task(task_name))
            state = env._read_state()
            failed = torch.tensor([True, False, False])
            for active in (torch.tensor([True, False, True]),
                           torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)):
                expected = torch.zeros(3)
                if active.any():
                    terms = env.task.reward_components(
                        {key: value[active] for key, value in state.items()},
                        env.actions[active], failed[active], env.dt,
                    )
                    expected[active] = sum(terms.values())
                actual = env._compute_reward(state, failed, active)
                torch.testing.assert_close(actual, expected)
                log = env._make_extras(state, failed & active, torch.zeros(3, dtype=torch.bool), active)["log"]
                if active.any():
                    for name, values in terms.items():
                        torch.testing.assert_close(log[f"Reward/{name}"], values.mean())
                    torch.testing.assert_close(log["Task/height"], state["height"][active].mean())
                else:
                    self.assertNotIn("Reward/active_samples", log)

    def test_initial_settling_before_first_control_transition(self):
        env, sim, scene = self.make_env(finish=False)
        self.assertEqual(sim.steps, 0)
        self.assertEqual(env.settle_steps_remaining.tolist(), [9, 9, 9])
        env.finish_initial_settling()
        self.assertEqual(sim.steps, 36)
        # 无随机推力/鼠标外力时，每个控制周期只写一次关节目标。
        self.assertEqual(scene.writes, 9)
        self.assertEqual(scene.updates, 36)
        self.assertEqual(env.episode_length_buf.tolist(), [0, 0, 0])
        self.assertEqual(torch.count_nonzero(env.attitude_integral.values), 0)
        torch.testing.assert_close(env.get_observations()["policy"][:, :3], torch.zeros(3, 3))
        torch.testing.assert_close(env._read_state()["height"], torch.full((3,), .32))
        torch.testing.assert_close(scene.robot.targets, env.default_joint_pos)
        _, rewards, dones, _ = env.step(env.standing_actions)
        self.assertEqual(sim.steps, 40)
        self.assertEqual(env.episode_length_buf.tolist(), [1, 1, 1])
        self.assertEqual(rewards.shape, (3,))
        self.assertEqual(dones.tolist(), [0, 0, 0])

    def test_partial_reset_does_not_advance_or_reset_other_environments(self):
        env, sim, scene = self.make_env()
        data = scene.robot.data
        data.root_link_pose_w.torch[:, 2] += .5
        data.root_com_vel_w.torch.fill_(2.)
        data.joint_pos.torch.add_(.1)
        data.joint_vel.torch.fill_(3.)
        env.episode_length_buf[:] = 20
        env.attitude_integral.values.fill_(.1)
        untouched = data.root_link_pose_w.torch[[0, 2]].clone()
        env.reset(torch.tensor([1]))
        self.assertEqual(sim.steps, 36)
        torch.testing.assert_close(data.root_link_pose_w.torch[[0, 2]], untouched)
        self.assertAlmostEqual(data.root_link_pose_w.torch[1, 2].item(), 1.4, places=5)
        torch.testing.assert_close(data.root_com_vel_w.torch[1], torch.zeros(6))
        torch.testing.assert_close(data.joint_pos.torch[1], env.default_joint_pos[1])
        torch.testing.assert_close(data.joint_vel.torch[1], torch.zeros(12))
        self.assertEqual(env.settle_steps_remaining.tolist(), [0, 9, 0])
        self.assertEqual(env.episode_length_buf.tolist(), [20, 0, 20])
        torch.testing.assert_close(env.attitude_integral.values[1], torch.zeros(3))
        torch.testing.assert_close(env.attitude_integral.values[[0, 2]], torch.full((2, 3), .1))

    def test_randomized_resets_resample_angles_and_pause_push_timer(self):
        env, sim, scene = self.make_env(randomized=True)
        self.assertTrue((env.randomization.wait_steps >= 240).all())
        env.randomization.wait_steps[1] = 0
        env.randomization.remaining_steps[1] = 10
        env.randomization.forces[1] = 40.
        env.reset(torch.tensor([1]))
        q1 = scene.robot.data.joint_pos.torch[1].clone()
        env.reset(torch.tensor([1]))
        q2 = scene.robot.data.joint_pos.torch[1].clone()
        self.assertFalse(torch.equal(q1, q2))
        self.assertTrue(((q2 - env.default_joint_pos[1]).abs() <= .10001).all())
        self.assertEqual(sim.steps, 36)
        self.assertEqual(env.randomization.remaining_steps[1], 0)
        self.assertEqual(torch.count_nonzero(env.randomization.forces[1]), 0)
        self.assertGreaterEqual(env.randomization.wait_steps[1], 240)
        wait = env.randomization.wait_steps[1].clone()
        for _ in range(9):
            _, rewards, dones, extras = env.step(env.standing_actions)
            self.assertFalse(extras["policy_active"][1])
            self.assertEqual(rewards[1], 0)
            self.assertEqual(dones[1], 0)
            self.assertEqual(env.randomization.wait_steps[1], wait)
        env.step(env.standing_actions)
        self.assertEqual(env.randomization.wait_steps[1], wait - 4)

    def test_every_episode_waits_without_stopping_other_environments(self):
        env, sim, scene = self.make_env()
        for _ in range(2):
            env.reset(torch.tensor([1]))
            actions = torch.full((3, 12), .2)
            previous_steps = sim.steps
            other_length = env.episode_length_buf[0].item()
            # 等待期间即使倾斜超过终止阈值也不产生失败样本。
            scene.robot.data.projected_gravity_b.torch[1] = torch.tensor([1., 0., 0.])
            for i in range(9):
                _, rewards, dones, extras = env.step(actions)
                self.assertEqual(sim.steps, previous_steps + 4 * (i + 1))
                self.assertEqual(env.episode_length_buf[0], other_length + i + 1)
                self.assertEqual(env.episode_length_buf[1], 0)
                self.assertEqual(rewards[1], 0)
                self.assertEqual(dones[1], 0)
                self.assertFalse(extras["policy_active"][1])
                torch.testing.assert_close(env.attitude_integral.values[1], torch.zeros(3))
                torch.testing.assert_close(env.actions[1], env.standing_actions[1])
            scene.robot.data.projected_gravity_b.torch[1] = torch.tensor([0., 0., -1.])
            _, _, _, extras = env.step(actions)
            self.assertTrue(extras["policy_active"].all())
            self.assertEqual(env.episode_length_buf[1], 1)
            torch.testing.assert_close(env.actions[1], actions[1])

    def test_action_history_uses_control_steps_and_resets_only_selected_envs(self):
        env, _, _ = self.make_env()
        initial = env.standing_actions.clone()
        env._set_action(torch.full((3, 12), .5))
        torch.testing.assert_close(env.previous_actions, initial)
        torch.testing.assert_close(env._read_state()["previous_actions"], initial)
        env._set_action(torch.full((3, 12), 2.))
        torch.testing.assert_close(env.raw_actions, torch.full((3, 12), 2.))
        torch.testing.assert_close(env.previous_actions, torch.full((3, 12), .5))
        torch.testing.assert_close(env.actions, torch.ones(3, 12))
        env.reset(torch.tensor([1]))
        torch.testing.assert_close(env.actions[1], initial[1])
        torch.testing.assert_close(env.raw_actions[1], initial[1])
        torch.testing.assert_close(env.raw_actions[[0, 2]], torch.full((2, 12), 2.))
        torch.testing.assert_close(env.previous_actions[1], initial[1])
        torch.testing.assert_close(env.previous_actions[[0, 2]], torch.full((2, 12), .5))
        env._set_action(env.actions.clone())
        torch.testing.assert_close(env.actions, env.previous_actions)

    def test_contact_reward_and_logging_exclude_settling_and_reset_only_done_history(self):
        env, _, scene = self.make_env(reward_cfg=RewardCfg(undesired_contact_weight=10.))
        env.reset(torch.tensor([1]))
        sensor = scene.contact_forces
        names = sensor.body_names
        sensor.forces[:2, names.index("FL_calf"), 2] = 5.
        sensor.forces[:2, names.index("FR_calf"), 2] = 5.
        for leg in ("FL", "FR", "RL", "RR"):
            sensor.forces[2, names.index(f"{leg}_foot"), 2] = 30.
        _, rewards, dones, extras = env.step(env.standing_actions)
        self.assertEqual(rewards[1], 0.)  # 等待中的接触不进入 PPO 奖励。
        self.assertEqual(dones.tolist(), [0, 0, 0])  # 非足部接触不直接结束回合。
        self.assertEqual(extras["log"]["Contact/undesired_body_count"], 1.)  # 活动环境 [2, 0]。
        self.assertEqual(extras["log"]["Contact/undesired_env_fraction"], .5)
        self.assertEqual(extras["log"]["Contact/foot_body_count"], 2.)
        torch.testing.assert_close(env._reward_components["undesired_contact"],
                                   torch.tensor([-20 * env.dt, 0.]))
        torch.testing.assert_close(rewards[[0, 2]], sum(env._reward_components.values()))
        history = sensor.data.net_normal_forces_w_history.torch
        untouched = history[[0, 2]].clone()
        env.reset(torch.tensor([1]))
        self.assertEqual(torch.count_nonzero(history[1]), 0)
        torch.testing.assert_close(history[[0, 2]], untouched)

    def test_stationary_rewards_read_world_velocity_and_exclude_settling(self):
        env, _, scene = self.make_env()
        env.reset(torch.tensor([1]))
        data = scene.robot.data
        # 机身坐标数据保持零，世界坐标中 0 号平移、2 号转动，1 号正在等待。
        data.root_com_vel_w.torch[:, 0] = torch.tensor([.5, 100., 0.])
        data.root_com_vel_w.torch[:, 5] = torch.tensor([0., 100., 1.])
        _, rewards, _, extras = env.step(env.standing_actions)
        cfg, dt = env.reward_cfg, env.dt
        torch.testing.assert_close(env._reward_components["horizontal_velocity"],
            cfg.horizontal_velocity_weight * torch.exp(-torch.tensor([.5, 0.]).square() / cfg.horizontal_velocity_sigma**2) * dt)
        torch.testing.assert_close(env._reward_components["yaw_velocity"],
            cfg.yaw_velocity_weight * torch.exp(-(torch.tensor([0., 1.]) / cfg.yaw_velocity_sigma).square()) * dt)
        self.assertEqual(rewards[1], 0.)
        self.assertEqual(extras["log"]["Task/horizontal_speed_mps"], .25)
        self.assertEqual(extras["log"]["Task/yaw_velocity_abs_radps"], .5)

    def test_foot_reward_works_without_undesired_penalty_and_excludes_waiting(self):
        cfg = RewardCfg(undesired_contact_weight=0., foot_support_weight=20.)
        env, _, scene = self.make_env(reward_cfg=cfg)
        self.assertIsNotNone(env.contacts)
        env.reset(torch.tensor([1]))
        sensor = scene.contact_forces
        for leg in ("FL", "FR", "RL", "RR"):
            sensor.forces[:, sensor.body_names.index(f"{leg}_foot"), 2] = 30.
        env.commands.values[2, 2] = .5  # 原地转向指令不要求四足支撑。
        _, rewards, dones, extras = env.step(env.standing_actions)
        torch.testing.assert_close(env._reward_components["foot_support"], torch.tensor([20 * env.dt, 0.]))
        self.assertEqual(rewards[1], 0.)
        self.assertEqual(dones.tolist(), [0, 0, 0])
        self.assertEqual(extras["log"]["Contact/all_feet_support_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
