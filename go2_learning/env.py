"""RSL-RL VecEnv：52 维观测、12 维关节位置动作，带姿态和高度积分的站立任务。"""

import math

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .commands import CommandCfg, VelocityCommands
from .contacts import Go2Contacts
from .attitude_integral import AttitudeIntegral, IntegralCfg
from .rewards import RewardCfg, action_limit_penalty, reward_components, roll_pitch_from_gravity


def make_env(sim, num_envs=1, randomization_cfg=None, seed=42,
             command_cfg=None, random_commands=False, reward_cfg=None, integral_cfg=None):
    from isaaclab.scene import InteractiveScene

    from .scene import Go2SceneCfg, USD_PATH, GROUND_PATH

    if not USD_PATH.is_file():
        raise FileNotFoundError(f"找不到 Go2 模型：{USD_PATH}")
    if not GROUND_PATH.is_file():
        raise FileNotFoundError(f"找不到地面模型：{GROUND_PATH}")
    reward_cfg = reward_cfg or RewardCfg()
    scene_cfg = Go2SceneCfg(num_envs=num_envs, env_spacing=3.0, replicate_physics=False)
    if reward_cfg.undesired_contact_weight == 0.0 and reward_cfg.foot_support_weight == 0.0:
        # 关闭奖励时不额外创建传感器；旧模型回放也不引入新约束。
        scene_cfg.contact_forces = None
        scene_cfg.robot.spawn.activate_contact_sensors = False
    else:
        scene_cfg.contact_forces.update_period = sim.get_physics_dt()
    scene = InteractiveScene(scene_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.3))
    sim.reset()
    randomization = None
    if randomization_cfg is not None:
        from .randomization import DomainRandomizer

        randomization = DomainRandomizer(
            scene["robot"], num_envs, sim.device, sim.get_physics_dt(), randomization_cfg, seed,
        )
    return Go2Env(sim, scene, randomization, command_cfg, random_commands, reward_cfg, seed, integral_cfg)


class Go2Env(VecEnv):
    def __init__(self, sim, scene, randomization=None, command_cfg=None,
                 random_commands=False, reward_cfg=None, seed=42, integral_cfg=None):
        self.sim, self.scene = sim, scene
        self.robot = scene["robot"]
        self.device = sim.device
        self.num_envs = scene.num_envs
        self.num_actions = self.robot.num_joints
        if self.num_actions != 12:
            raise ValueError(f"预期 Go2 有 12 个关节，实际为 {self.robot.joint_names}")
        # 按名称选关节，避免依赖 USD 中 12 个关节的排列顺序。
        self.hip_joint_ids, hip_names = self.robot.find_joints(".*_hip_joint")
        if len(self.hip_joint_ids) != 4:
            raise ValueError(f"预期四个髋侧摆关节，实际匹配到 {hip_names}")
        self.cfg = {}
        self.physics_dt = sim.get_physics_dt()
        self.decimation = 4  # 物理 120 Hz，策略/环境 30 Hz。
        self.dt = self.physics_dt * self.decimation
        self.commands = VelocityCommands(
            self.num_envs, self.device, self.dt, command_cfg or CommandCfg(), random_commands, seed + 1,
        )
        self.reward_cfg = reward_cfg or RewardCfg()
        self.contacts = None
        if self.reward_cfg.undesired_contact_weight != 0.0 or self.reward_cfg.foot_support_weight != 0.0:
            self.contacts = Go2Contacts(scene["contact_forces"], self.reward_cfg.contact_force_threshold,
                                        self.decimation)
            print(f"接触惩罚：足部允许接触；非足部 {self.contacts.undesired_names}；"
                  f"阈值 {self.reward_cfg.contact_force_threshold:g} N，"
                  f"每个接触刚体扣 {self.reward_cfg.undesired_contact_weight:g} 分/秒")
            print(f"站立四足支撑奖励：最大 {self.reward_cfg.foot_support_weight:g} 分/秒；"
                  "按每个物理步的承重脚掌数计分，有移动或转向指令时关闭")
        self.attitude_integral = AttitudeIntegral(
            self.num_envs, self.device, self.dt, integral_cfg or IntegralCfg(),
        )
        self.max_episode_length = math.ceil(15.0 / self.dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.raw_actions = torch.zeros_like(self.actions)
        self.previous_actions = torch.zeros_like(self.actions)
        self.default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        # 按运行时关节顺序读取 USD 的完整机械限位，单位为 rad。
        self.action_joint_limits = self.robot.data.joint_pos_limits.torch.clone()
        lower, upper = self.action_joint_limits.unbind(dim=-1)
        if not torch.isfinite(self.action_joint_limits).all() or not (upper > lower).all():
            raise ValueError("完整范围动作映射要求每个关节都有有限且有效的上下限")
        self.action_center = (lower + upper) / 2
        self.action_half_range = (upper - lower) / 2
        # 零动作现在表示机械范围中点；默认站姿对应的动作通常非零。
        self.standing_actions = ((self.default_joint_pos - self.action_center)
                                 / self.action_half_range).clamp(-1.0, 1.0)
        self.randomization = randomization
        # 由运行入口接入外部交互；环境只规定调用时机。
        self.before_physics_step = None
        self.on_reset = None
        self.reset_settle_seconds = 0.3  # 仿真时间；预运行期间不采样、不计奖励或回合时间。
        self.settle_steps = math.ceil(self.reset_settle_seconds / self.dt)
        self.settle_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset()

    @torch.no_grad()
    def finish_initial_settling(self):
        """只在 Runner 创建前完成首回合等待，不产生训练样本。"""
        while (self.settle_steps_remaining > 0).any():
            self.step(self.standing_actions, auto_reset=False)

    @property
    def policy_active(self):
        """本控制步开始时已经结束等待的环境。"""
        return self.settle_steps_remaining == 0

    def get_observations(self):
        data = self.robot.data
        obs = torch.cat((
            data.root_lin_vel_b.torch,                       # 3：机身线速度
            data.root_ang_vel_b.torch,                       # 3：机身角速度
            data.projected_gravity_b.torch,                  # 3：机身坐标下的重力方向
            data.joint_pos.torch - self.default_joint_pos,   # 12：关节偏移
            0.1 * data.joint_vel.torch,                      # 12：缩放后的关节速度
            self.actions,                                   # 12：上一动作
            self.commands.values,                           # 3：目标 vx、vy、yaw_rate
            (data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
             - self.reward_cfg.target_height).unsqueeze(-1), # 1：瞬时高度误差，m
            self.attitude_integral.observations(),           # 3：归一化 roll/pitch/height 积分
        ), dim=-1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def step(self, actions, auto_reset=True):
        """执行一次控制步：动作 → 仿真 → 评估 → 重置 → 返回观测。"""
        active = self.policy_active
        self._set_action(actions)
        self._advance_simulation(active)
        self.settle_steps_remaining.sub_((self.settle_steps_remaining > 0).long())
        self.episode_length_buf += active.long()

        state = self._read_state()
        # 积分仅保留在观测和日志中，奖励不使用积分。
        self.attitude_integral.update(state["roll"], state["pitch"], state["height_error"], active)
        failed, timed_out = self._compute_termination(state)
        failed, timed_out = failed & active, timed_out & active
        rewards = self._compute_reward(state, failed, active)
        dones = failed | timed_out
        extras = self._make_extras(state, failed, timed_out, active)
        extras["policy_active"] = active
        self.commands.step(dones, active)

        # 奖励/结束标志属于重置前；自动重置的环境返回新一轮的初始观测。
        if auto_reset:
            self._reset_finished_envs(dones)
        observations = self.get_observations()
        return observations, rewards, dones.long(), extras

    def _set_action(self, actions):
        """[-1, 1] 分别对应每个关节机械下限/上限，0 对应中点。"""
        self.previous_actions.copy_(self.actions)
        self.raw_actions.copy_(torch.where(self.policy_active[:, None], actions.detach(), self.standing_actions))
        self.actions.copy_(self.raw_actions.clamp(-1.0, 1.0))
        target = self.action_center + self.action_half_range * self.actions
        self.robot.actuators.target_command.set_position_index(value=target)

    def _advance_simulation(self, active):
        """保持当前关节目标，推进 decimation 个物理步。"""
        for _ in range(self.decimation):
            if self.before_physics_step is not None:
                self.before_physics_step()
            if self.randomization is not None:
                self.randomization.apply(active)
            self.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.scene.update(self.physics_dt)

    def _read_state(self):
        """集中读取奖励和结束条件需要的本步状态。"""
        data = self.robot.data
        roll, pitch = roll_pitch_from_gravity(data.projected_gravity_b.torch)
        height = data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        world_velocity = data.root_com_vel_w.torch  # 机身质心：世界线速度 XYZ + 世界角速度 XYZ。
        state = {
            "height": height,
            "height_error": height - self.reward_cfg.target_height,
            "gravity": data.projected_gravity_b.torch,
            "linear_velocity": data.root_lin_vel_b.torch,
            "angular_velocity": data.root_ang_vel_b.torch,
            "linear_velocity_w": world_velocity[:, :3],
            "angular_velocity_w": world_velocity[:, 3:],
            "joint_offset": data.joint_pos.torch - self.default_joint_pos,
            "hip_joint_pos": data.joint_pos.torch[:, self.hip_joint_ids],
            "roll": roll,
            "pitch": pitch,
            "attitude_integral": self.attitude_integral.values,
            "previous_actions": self.previous_actions,
            "raw_actions": self.raw_actions,
            "commands": self.commands.values,
        }
        if self.contacts is not None:
            state.update(self.contacts.read())
        return state

    def _compute_reward(self, state, failed, active):
        """环境提供状态，奖励公式由 rewards.py 管理。"""
        rewards = torch.zeros(self.num_envs, device=self.device)
        self._reward_components = {}
        if active.any():
            active_state = {key: value[active] for key, value in state.items()}
            self._reward_components = reward_components(
                active_state, self.actions[active], failed[active],
                self.dt, self.reward_cfg,
            )
            rewards[active] = sum(self._reward_components.values())
        return rewards

    def _compute_termination(self, state):
        """区分跌倒和超时，跌倒优先；供 PPO 正确处理超时补偿。"""
        failed = (state["height"] < 0.18) | (state["gravity"][:, 2] > -0.5)
        timed_out = (self.episode_length_buf >= self.max_episode_length) & ~failed
        return failed, timed_out

    def _make_extras(self, state, failed, timed_out, active):
        """组织 RSL-RL 使用的超时标志和日志指标。"""
        if not active.any():
            return {"time_outs": timed_out, "log": {"Task/failures_per_step": failed.float().sum()}}
        state = {key: value[active] for key, value in state.items()}
        # 保留机身坐标的旧指令误差日志；新的静止目标直接使用世界坐标速度。
        commands = self.commands.values[active]
        linear_error = (state["linear_velocity"][:, :2] - commands[:, :2]).square().sum(dim=-1)
        yaw_error = (state["angular_velocity"][:, 2] - commands[:, 2]).square()
        return {
            "time_outs": timed_out,
            "log": {
                "Task/failures_per_step": failed.float().sum(),
                "Reward/active_samples": active.sum(),
                **{f"Reward/{name}": values.mean() for name, values in self._reward_components.items()},
                "Policy/raw_action_abs_max": self.raw_actions[active].abs().max(),
                "Policy/action_clip_fraction": (self.raw_actions[active].abs() > 1.0).float().mean(),
                "Policy/action_limit_violation": action_limit_penalty(self.raw_actions[active]).mean(),
                "Task/height": state["height"].mean(),
                "Task/height_abs_error_m": state["height_error"].abs().mean(),
                "Task/hip_abs_angle_rad": state["hip_joint_pos"].abs().mean(),
                "Task/joint_posture_rms_rad": state["joint_offset"].square().mean(dim=-1).sqrt().mean(),
                "Task/roll_abs_rad": state["roll"].abs().mean(),
                "Task/pitch_abs_rad": state["pitch"].abs().mean(),
                "Task/roll_velocity_abs_radps": state["angular_velocity"][:, 0].abs().mean(),
                "Task/pitch_velocity_abs_radps": state["angular_velocity"][:, 1].abs().mean(),
                "Task/horizontal_speed_mps": state["linear_velocity_w"][:, :2].norm(dim=-1).mean(),
                "Task/yaw_velocity_abs_radps": state["angular_velocity_w"][:, 2].abs().mean(),
                "Task/roll_integral_abs_rad_s": state["attitude_integral"][:, 0].abs().mean(),
                "Task/pitch_integral_abs_rad_s": state["attitude_integral"][:, 1].abs().mean(),
                "Task/height_integral_abs_m_s": state["attitude_integral"][:, 2].abs().mean(),
                "Task/linear_tracking_error_mps": linear_error.sqrt().mean(),
                "Task/yaw_tracking_error_radps": yaw_error.sqrt().mean(),
                "Task/command_vx": self.commands.values[active, 0].mean(),
                "Task/actual_vx": state["linear_velocity"][:, 0].mean(),
                "Task/command_yaw_rate": self.commands.values[active, 2].mean(),
                "Task/actual_yaw_rate": state["angular_velocity"][:, 2].mean(),
                **({
                    "Contact/undesired_body_count": state["undesired_contact_count"].mean(),
                    "Contact/undesired_env_fraction": (state["undesired_contact_count"] > 0).float().mean(),
                    "Contact/foot_body_count": state["foot_contact_count"].mean(),
                    "Contact/foot_support_fraction": state["foot_support_fraction"].mean(),
                    "Contact/all_feet_support_fraction": state["all_feet_support_fraction"].mean(),
                } if self.contacts is not None else {}),
            },
        }

    def _reset_finished_envs(self, dones):
        """仅重置本步结束的环境，其余环境继续运行。"""
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.reset(done_ids)

    def reset(self, env_ids=None):
        if self.on_reset is not None:
            self.on_reset()
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.attitude_integral.reset(env_ids)
        self.commands.reset(env_ids)
        # 每回合从配置中的初始状态重新开始，不恢复任何历史速度或落地快照。
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        if self.randomization is not None:
            joint_pos = self.randomization.reset(env_ids, joint_pos)
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=velocity, env_ids=env_ids,
        )
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel, env_ids=env_ids,
        )
        self.scene.reset(env_ids)
        self.robot.actuators.target_command.set_position_index(value=self.default_joint_pos[env_ids], env_ids=env_ids)
        self.actions[env_ids] = self.standing_actions[env_ids]
        self.raw_actions[env_ids] = self.standing_actions[env_ids]
        self.previous_actions[env_ids] = self.standing_actions[env_ids]
        self.episode_length_buf[env_ids] = 0
        self.settle_steps_remaining[env_ids] = self.settle_steps
