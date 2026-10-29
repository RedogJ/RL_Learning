"""通用 Go2 VecEnv：仿真、观测、动作和重置；奖励及失败规则交给所选任务。"""

import math

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from go2_learning.control.commands import VelocityCommands
from go2_learning.simulation.contacts import LEGS, Go2Contacts
from go2_learning.control.attitude_integral import AttitudeIntegral
from go2_learning.tasks.reward_terms import action_limit_penalty, roll_pitch_from_gravity
from go2_learning.tasks import get_task


def make_env(sim, num_envs=1, randomization_cfg=None, seed=42,
             command_cfg=None, random_commands=False, reward_cfg=None, integral_cfg=None, task=None,
             terrain="flat"):
    from isaaclab.scene import InteractiveScene

    from go2_learning.simulation.scene import Go2SceneCfg, USD_PATH, GROUND_PATH

    if not USD_PATH.is_file():
        raise FileNotFoundError(f"找不到 Go2 模型：{USD_PATH}")
    if terrain not in ("flat", "rough", "stairs"):
        raise ValueError(f"未知地形：{terrain}")
    if terrain != "flat" and (num_envs != 1 or random_commands):
        raise ValueError("崎岖/楼梯地形目前用于单环境 view/play 测试")
    if terrain == "flat" and not GROUND_PATH.is_file():
        raise FileNotFoundError(f"找不到地面模型：{GROUND_PATH}")
    task = task or get_task()
    if reward_cfg is not None:
        task.cfg.rewards = reward_cfg  # 兼容旧调用方式；入口直接传 task。
    scene_cfg = Go2SceneCfg(num_envs=num_envs, env_spacing=task.cfg.environment.env_spacing, replicate_physics=False)
    task.configure_scene(scene_cfg)
    if terrain != "flat":
        from go2_learning.simulation.terrain import make_rough_terrain_cfg, make_stairs_terrain_cfg

        # 替换碰撞平面，不能在平面上叠加起伏，否则凹处仍会踩到平面。
        scene_cfg.ground = (make_rough_terrain_cfg(seed) if terrain == "rough"
                            else make_stairs_terrain_cfg(seed))
        x, y, z = scene_cfg.robot.init_state.pos
        scene_cfg.robot.init_state.pos = (x, y, z + (0.10 if terrain == "rough" else 0.))
    if not task.needs_contact_sensor:
        # 关闭奖励时不额外创建传感器；旧模型回放也不引入新约束。
        scene_cfg.contact_forces = None
        scene_cfg.robot.spawn.activate_contact_sensors = False
    else:
        scene_cfg.contact_forces.update_period = sim.get_physics_dt()
        scene_cfg.contact_forces.history_length = task.cfg.environment.decimation
    scene = InteractiveScene(scene_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.3))
    if terrain == "stairs":
        sim.set_camera_view(eye=(5., 6., 3.), target=(2.8, 0., 0.35))
    sim.reset()
    if terrain != "flat":
        from go2_learning.ui.terrain_view import style_rough_terrain

        style_rough_terrain(sim, terrain=terrain)
    randomization = None
    if randomization_cfg is not None:
        from go2_learning.simulation.randomization import DomainRandomizer

        randomization = DomainRandomizer(
            scene["robot"], num_envs, sim.device, sim.get_physics_dt(), randomization_cfg, seed,
        )
    return Go2Env(sim, scene, randomization, command_cfg, random_commands, reward_cfg, seed, integral_cfg, task)


class Go2Env(VecEnv):
    def __init__(self, sim, scene, randomization=None, command_cfg=None,
                 random_commands=False, reward_cfg=None, seed=42, integral_cfg=None, task=None):
        self.sim, self.scene = sim, scene
        self.task = task or get_task()
        if reward_cfg is not None:
            self.task.cfg.rewards = reward_cfg
        if command_cfg is not None:
            self.task.cfg.commands = command_cfg
        if integral_cfg is not None:
            self.task.cfg.integral = integral_cfg
        env_cfg = self.task.cfg.environment
        self.robot = scene["robot"]
        self.device = sim.device
        self.num_envs = scene.num_envs
        self.num_actions = self.robot.num_joints
        if self.num_actions != 12:
            raise ValueError(f"预期 Go2 有 12 个关节，实际为 {self.robot.joint_names}")
        # 按名称选关节，避免依赖 USD 中 12 个关节的排列顺序。
        joint_names = list(self.robot.joint_names)

        def leg_order(kind):
            ids, names = self.robot.find_joints(f".*_{kind}_joint")
            if len(ids) != 4:
                raise ValueError(f"预期四个 {kind} 关节，实际匹配到 {names}")
            # 日志固定按 FL、FR、RL、RR；find_joints 的顺序取决于 USD，不能直接当腿序使用。
            try:
                return ids, [ids.index(joint_names.index(f"{leg}_{kind}_joint")) for leg in LEGS]
            except ValueError as error:
                raise ValueError(f"无法按 FL/FR/RL/RR 对应 {kind} 关节：{names}") from error

        self.hip_joint_ids, self.hip_log_order = leg_order("hip")
        self.thigh_joint_ids, self.thigh_log_order = leg_order("thigh")
        self.calf_joint_ids, self.calf_log_order = leg_order("calf")
        self.hip_ordered_ids = [self.hip_joint_ids[i] for i in self.hip_log_order]
        self.thigh_ordered_ids = [self.thigh_joint_ids[i] for i in self.thigh_log_order]
        self.calf_ordered_ids = [self.calf_joint_ids[i] for i in self.calf_log_order]
        self.diagnostics_interval = 1  # 训练入口默认设为每轮采样一次。
        self.detailed_diagnostics = True
        self._diagnostic_step = 0
        self._pending_policy_selection = None
        self._step_active_ids = None
        self.cfg = {}
        self.physics_dt = sim.get_physics_dt()
        self.decimation = env_cfg.decimation
        self.render_every_physics_steps = 1
        self.dt = self.physics_dt * self.decimation
        self.commands = VelocityCommands(
            self.num_envs, self.device, self.dt, self.task.cfg.commands, random_commands, seed + 1,
        )
        self.reward_cfg = self.task.cfg.rewards  # 兼容已有调试代码。
        self.contacts = None
        if self.task.needs_contact_sensor:
            self.contacts = Go2Contacts(scene["contact_forces"], self.task.contact_force_threshold,
                                        self.decimation,
                                        track_phase_times=getattr(self.reward_cfg, "foot_time_variance_weight", 0.0) != 0)
            print(f"接触检测：任务 {self.task.name}；足部与非足部按刚体名称区分；"
                  f"阈值 {self.task.contact_force_threshold:g} N")
        self.foot_body_ids = None
        if getattr(self.reward_cfg, "foot_slide_weight", 0.0) != 0:
            # 两侧刚体顺序可能不同，按名字匹配传感器足部与机器人速度。
            self.foot_body_ids = [self.robot.body_names.index(self.contacts.sensor.body_names[i])
                                  for i in self.contacts.foot_ids]
        if getattr(self.reward_cfg, "rear_knee_clearance_weight", 0.0) != 0:
            self.rear_knee_body_ids = [self.robot.body_names.index(f"{leg}_calf") for leg in LEGS[2:]]
        self.attitude_integral = AttitudeIntegral(
            self.num_envs, self.device, self.dt, self.task.cfg.integral,
        )
        self.max_episode_length = math.ceil(env_cfg.episode_seconds / self.dt)
        self.height_termination_enabled = True
        self.failure_termination_enabled = True
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.raw_actions = torch.zeros_like(self.actions)
        self.previous_raw_actions = torch.zeros_like(self.actions)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_target_delta = torch.zeros_like(self.actions)
        self.action_filter_alpha = getattr(self.reward_cfg, "action_filter_alpha", 0.0)
        if not 0.0 <= self.action_filter_alpha < 1.0:
            raise ValueError("action_filter_alpha 必须在 [0, 1) 范围内")
        self.default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        # 按运行时关节顺序读取 USD 的完整机械限位，单位为 rad。
        self.action_joint_limits = self.robot.data.joint_pos_limits.torch.clone()
        lower, upper = self.action_joint_limits.unbind(dim=-1)
        if not torch.isfinite(self.action_joint_limits).all() or not (upper > lower).all():
            raise ValueError("完整范围动作映射要求每个关节都有有限且有效的上下限")
        self.action_center = (lower + upper) / 2
        self.action_half_range = (upper - lower) / 2
        # 物理重置姿态独立于策略观测使用的默认关节参考，旧检查点仍使用原观测定义。
        self.reset_root_pose, self.reset_joint_pos, self.reset_joint_target = self.task.reset_state(
            self.robot.data.default_root_pose.torch.clone(), self.default_joint_pos.clone(), joint_names,
        )
        for label, positions in (("初始关节角", self.reset_joint_pos), ("等待期 PD 目标", self.reset_joint_target)):
            if (not torch.isfinite(positions).all()
                    or not ((positions >= lower) & (positions <= upper)).all()):
                raise ValueError(f"任务重置{label}超出当前模型的机械限位")
        from go2_learning.simulation.joint_lock import lock_joint_positions

        fixed_positions = self.task.fixed_joint_positions()
        # 原机械范围和默认观测参考已保存；物理锁定不改变旧网络的动作映射与观测定义。
        self.fixed_joint_ids = lock_joint_positions(self.robot, fixed_positions)
        self.policy_joint_ids = [i for i in range(self.num_actions) if i not in self.fixed_joint_ids]
        if self.fixed_joint_ids:
            fixed = self.reset_joint_pos.new_tensor(list(fixed_positions.values()))
            self.reset_joint_pos[:, self.fixed_joint_ids] = fixed
            self.reset_joint_target[:, self.fixed_joint_ids] = fixed
        # 零动作现在表示机械范围中点；默认站姿对应的动作通常非零。
        # 初始关节角与等待期 PD 目标可不同；直立时需要预载以抵抗重力。
        # 此目标只用于 view 与等待期，策略接管后仍使用原来的完整范围动作映射。
        self.standing_actions = ((self.reset_joint_target - self.action_center)
                                 / self.action_half_range).clamp(-1.0, 1.0)
        self.action_activity = None
        if getattr(self.reward_cfg, "action_activity_weight", 0.0) != 0:
            from go2_learning.control.action_activity import ActionActivity
            self.action_activity = ActionActivity(self.robot.joint_names, self.num_envs, self.device,
                                                  self.dt, getattr(self.reward_cfg, "action_activity_tau", 0.5))
        self.randomization = randomization
        # 由运行入口接入外部交互；环境只规定调用时机。
        self.before_physics_step = None
        self.on_reset = None
        self.reset_settle_seconds = env_cfg.reset_settle_seconds
        self.settle_steps = math.ceil(self.reset_settle_seconds / self.dt)
        self.settle_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        factory = getattr(self.task, "make_reset_orientation_sampler", None)
        self.reset_orientation_sampler = factory(self, seed + 2) if factory is not None else None
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
             - self.task.target_height).unsqueeze(-1),       # 1：瞬时高度误差，m
            self.attitude_integral.observations(),           # 3：归一化 roll/pitch/height 积分
        ), dim=-1)
        observations = {"policy": obs}
        critic_observations = getattr(self.task, "critic_observations", None)
        if critic_observations is not None:
            observations.update(critic_observations(self))
        return TensorDict(observations, batch_size=[self.num_envs])

    def cache_policy_selection(self, active, ids):
        """策略采样后仅供紧接着的一次 step 使用，不能跨重置或控制步缓存。"""
        self._pending_policy_selection = (active, ids)

    def step(self, actions, auto_reset=True, evaluate=True):
        """执行一次控制步：动作 → 仿真 → 评估 → 重置 → 返回观测。"""
        selection = self._pending_policy_selection
        self._pending_policy_selection = None
        active = self.policy_active if selection is None else selection[0]
        self._step_active_ids = None if selection is None else selection[1]
        self._set_action(actions)
        self._advance_simulation(active)
        if evaluate and self.contacts is not None and (getattr(self.reward_cfg, "foot_air_time_weight", 0.0) != 0
                                          or getattr(self.reward_cfg, "foot_contact_ratio_weight", 0.0) != 0
                                          or getattr(self.reward_cfg, "foot_force_balance_weight", 0.0) != 0
                                          or getattr(self.reward_cfg, "foot_time_variance_weight", 0.0) != 0
                                          or getattr(self, "contact_diagnostics", False)):
            self.contacts.update_air_time(
                self.physics_dt, active, getattr(self.reward_cfg, "foot_contact_ratio_tau", 0.5))
        was_waiting = self.settle_steps_remaining > 0
        self.settle_steps_remaining.sub_(was_waiting.long())
        if self.reset_orientation_sampler is not None:
            ready = (was_waiting & (self.settle_steps_remaining == 0)).nonzero().flatten()
            if ready.numel():
                self.reset_orientation_sampler.apply(ready)
        self.episode_length_buf += active.long()

        state = self._read_state() if evaluate else self._read_state(minimal=True)
        # 积分仅保留在观测和日志中，奖励不使用积分。
        self.attitude_integral.update(state["roll"], state["pitch"], state["height_error"], active)
        failed, timed_out = self._compute_termination(state)
        failed, timed_out = failed & active, timed_out & active
        rewards = (self._compute_reward(state, failed, active) if evaluate
                   else torch.zeros(self.num_envs, device=self.device))
        dones = failed | timed_out
        extras = (self._make_extras(state, failed, timed_out, active) if evaluate
                  else {"time_outs": timed_out, "failed": failed})
        extras["policy_active"] = active
        self.commands.step(dones, active)

        # 奖励/结束标志属于重置前；自动重置的环境返回新一轮的初始观测。
        if auto_reset:
            self._reset_finished_envs(dones)
        observations = self.get_observations()
        self._step_active_ids = None
        return observations, rewards, dones.long(), extras

    def _set_action(self, actions):
        """[-1, 1] 分别对应每个关节机械下限/上限，0 对应中点。"""
        if getattr(self.reward_cfg, "action_rate_derivative_weight", 0.0) != 0:
            # 必须在 previous_actions 被覆盖前保存上一个控制步的目标角变化量。
            self.previous_target_delta.copy_((self.actions - self.previous_actions) * self.action_half_range)
        self.previous_actions.copy_(self.actions)
        if (getattr(self.reward_cfg, "action_rate_weight", 0.0) != 0
                and getattr(self.reward_cfg, "reward_schema_version", 11) < 11):
            self.previous_raw_actions.copy_(self.raw_actions)
        self.raw_actions.copy_(torch.where(self.policy_active[:, None], actions.detach(), self.standing_actions))
        if self.fixed_joint_ids:
            # 先替换固定关节指令，再计算执行动作与超界日志/惩罚。
            self.raw_actions[:, self.fixed_joint_ids] = self.standing_actions[:, self.fixed_joint_ids]
        clipped = self.raw_actions.clamp(-1.0, 1.0)
        if self.action_filter_alpha:
            self.actions.copy_(self.action_filter_alpha * self.previous_actions
                               + (1.0 - self.action_filter_alpha) * clipped)
            # 等待期间立即回到重置目标，避免滤波残留。
            self.actions.copy_(torch.where(self.policy_active[:, None], self.actions, self.standing_actions))
        else:
            self.actions.copy_(clipped)
        if self.action_activity is not None:
            self.action_activity.update(self.actions, self.previous_actions, self.policy_active)
        target = self.action_center + self.action_half_range * self.actions
        self.robot.actuators.target_command.set_position_index(value=target)

    def _advance_simulation(self, active):
        """保持当前关节目标，推进 decimation 个物理步。"""
        # 没有逐物理步外力时，目标在整个控制周期内不变，写一次即可。
        write_each_step = self.randomization is not None or self.before_physics_step is not None
        if not write_each_step:
            self.scene.write_data_to_sim()
        for substep in range(self.decimation):
            if self.before_physics_step is not None:
                self.before_physics_step()
            if self.randomization is not None:
                self.randomization.apply(active)
            if write_each_step:
                self.scene.write_data_to_sim()
            self.sim.step(render=(substep + 1) % getattr(self, "render_every_physics_steps", 1) == 0)
            self.scene.update(self.physics_dt)

    def _read_state(self, minimal=False):
        """集中读取奖励和结束条件需要的本步状态。"""
        data = self.robot.data
        roll, pitch = roll_pitch_from_gravity(data.projected_gravity_b.torch)
        height = data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        if minimal:
            # 播放仍保留相同的积分观测、跌倒和超时判定，只跳过奖励/训练统计。
            state = {"height": height, "height_error": height - self.task.target_height,
                     "gravity": data.projected_gravity_b.torch, "roll": roll, "pitch": pitch}
            if getattr(self.task.cfg.termination, "reset_on_nonfoot_contact", False):
                state["undesired_contact_count"] = self.contacts.read()["undesired_contact_count"]
            return state
        world_velocity = data.root_com_vel_w.torch  # 机身质心：世界线速度 XYZ + 世界角速度 XYZ。
        # 三类腿关节统一按 FL、FR、RL、RR 排列，奖励可按前后两组直接切片。
        hip_joint_pos = data.joint_pos.torch[:, self.hip_ordered_ids]
        thigh_joint_pos = data.joint_pos.torch[:, self.thigh_ordered_ids]
        calf_joint_pos = data.joint_pos.torch[:, self.calf_ordered_ids]
        state = {
            "height": height,
            "height_error": height - self.task.target_height,
            "gravity": data.projected_gravity_b.torch,
            "linear_velocity": data.root_lin_vel_b.torch,
            "angular_velocity": data.root_ang_vel_b.torch,
            "linear_velocity_w": world_velocity[:, :3],
            "angular_velocity_w": world_velocity[:, 3:],
            "joint_offset": data.joint_pos.torch - self.default_joint_pos,
            "hip_joint_pos": hip_joint_pos,
            "thigh_joint_pos": thigh_joint_pos,
            "calf_joint_pos": calf_joint_pos,
            "roll": roll,
            "pitch": pitch,
            "attitude_integral": self.attitude_integral.values,
            "previous_actions": self.previous_actions,
            "raw_actions": self.raw_actions,
            **({"joint_target_delta": (self.actions - self.previous_actions) * self.action_half_range}
               if getattr(self.reward_cfg, "action_rate_weight", 0.0) != 0
               and getattr(self.reward_cfg, "reward_schema_version", 11) >= 11 else {}),
            **({"joint_target_delta_derivative": (
                (self.actions - self.previous_actions) * self.action_half_range - self.previous_target_delta
            ) / self.dt} if getattr(self.reward_cfg, "action_rate_derivative_weight", 0.0) != 0 else {}),
            **({"previous_raw_actions": self.previous_raw_actions}
               if getattr(self.reward_cfg, "action_rate_weight", 0.0) != 0
               and getattr(self.reward_cfg, "reward_schema_version", 11) < 11 else {}),
            "commands": self.commands.values,
        }
        if getattr(self.reward_cfg, "joint_velocity_weight", 0.0) != 0:
            state["joint_velocity"] = data.joint_vel.torch
        if self.action_activity is not None:
            state["action_activity_variance"] = self.action_activity.penalty()
        if self.contacts is not None:
            state.update(self.contacts.read())
        if self.foot_body_ids is not None:
            state["foot_velocity_w"] = data.body_com_lin_vel_w.torch[:, self.foot_body_ids]
        if getattr(self, "rear_knee_body_ids", None) is not None:
            state["rear_knee_height"] = data.body_link_pos_w.torch[:, self.rear_knee_body_ids, 2].min(dim=-1).values
        return state

    def _compute_reward(self, state, failed, active):
        """环境提供状态，仅对正式控制样本调用所选任务的奖励。"""
        rewards = torch.zeros(self.num_envs, device=self.device)
        self._reward_components = {}
        # 布尔索引会重复求有效索引；每步只求一次，奖励和日志共用。
        self._active_ids = (self._step_active_ids if self._step_active_ids is not None
                            else active.nonzero(as_tuple=False).flatten())
        ids = self._active_ids
        self._active_state = {}
        if ids.numel():
            active_state = {key: value.index_select(0, ids) for key, value in state.items()}
            self._active_state = active_state
            self._reward_components = self.task.reward_components(
                active_state, self.actions[ids], failed[ids],
                self.dt,
            )
            rewards[ids] = sum(self._reward_components.values())
        return rewards

    def _compute_termination(self, state):
        """区分跌倒和超时，跌倒优先；供 PPO 正确处理超时补偿。"""
        if not self.failure_termination_enabled:
            failed = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        elif self.height_termination_enabled:
            failed = self.task.failures(state)
        else:
            failed = self.task.tilt_failures(state)
            contact_failures = getattr(self.task, "contact_failures", None)
            if contact_failures is not None:
                failed = failed | contact_failures(state)
        timed_out = (self.episode_length_buf >= self.max_episode_length) & ~failed
        return failed, timed_out

    def _make_extras(self, state, failed, timed_out, active):
        """组织 RSL-RL 使用的超时标志和日志指标。"""
        sample = self._diagnostic_step % self.diagnostics_interval == 0
        self._diagnostic_step += 1
        if not sample:
            return {"time_outs": timed_out, "log": {
                "Task/failures_per_step": failed.float().sum(),
                "Reward/active_samples": active.sum(),
            }}
        ids = self._active_ids
        if not ids.numel():
            return {"time_outs": timed_out, "log": {"Task/failures_per_step": failed.float().sum()}}
        # 奖励计算后、重置前读取；复用同一步的筛选结果。
        state = self._active_state
        # 共用运动学诊断同时保留机身指令误差和世界坐标速度，任务自行选择奖励所需量。
        commands = state["commands"]
        raw_actions = state["raw_actions"][:, self.policy_joint_ids]
        linear_error = (state["linear_velocity"][:, :2] - commands[:, :2]).square().sum(dim=-1)
        yaw_error = (state["angular_velocity"][:, 2] - commands[:, 2]).square()
        return {
            "time_outs": timed_out,
            "log": {
                "Task/failures_per_step": failed.float().sum(),
                "Reward/active_samples": active.sum(),
                **{f"Reward/{name}": values.mean() for name, values in self._reward_components.items()},
                "Policy/raw_action_abs_max": raw_actions.abs().max(),
                "Policy/action_clip_fraction": (raw_actions.abs() > 1.0).float().mean(),
                "Policy/action_limit_violation": action_limit_penalty(raw_actions).mean(),
                "Task/height": state["height"].mean(),
                "Task/height_abs_error_m": state["height_error"].abs().mean(),
                "Task/hip_abs_angle_rad": state["hip_joint_pos"].abs().mean(),
                **({f"Task/{kind}_{leg}_rad": state[f"{kind}_joint_pos"][:, index].mean()
                    for kind in ("hip", "thigh", "calf") for index, leg in enumerate(LEGS)}
                   if self.detailed_diagnostics else {}),
                "Task/joint_posture_rms_rad": state["joint_offset"].square().mean(dim=-1).sqrt().mean(),
                "Task/roll_abs_rad": state["roll"].abs().mean(),
                "Task/pitch_abs_rad": state["pitch"].abs().mean(),
                "Task/roll_velocity_abs_radps": state["angular_velocity"][:, 0].abs().mean(),
                "Task/pitch_velocity_abs_radps": state["angular_velocity"][:, 1].abs().mean(),
                "Task/horizontal_speed_mps": state["linear_velocity_w"][:, :2].norm(dim=-1).mean(),
                "Task/yaw_velocity_abs_radps": state["angular_velocity_w"][:, 2].abs().mean(),
                **({f"Task/{name}_integral_abs_{unit}": state["attitude_integral"][:, index].abs().mean()
                    for index, name, unit in ((0, "roll", "rad_s"), (1, "pitch", "rad_s"), (2, "height", "m_s"))}
                   if self.detailed_diagnostics else {}),
                "Task/linear_tracking_error_mps": linear_error.sqrt().mean(),
                "Task/yaw_tracking_error_radps": yaw_error.sqrt().mean(),
                "Task/command_vx": commands[:, 0].mean(),
                "Task/actual_vx": state["linear_velocity"][:, 0].mean(),
                "Task/command_yaw_rate": commands[:, 2].mean(),
                "Task/actual_yaw_rate": state["angular_velocity"][:, 2].mean(),
                **self.task.metrics(state),
                **({"Task/rear_knee_height": state["rear_knee_height"].mean()}
                   if "rear_knee_height" in state else {}),
                **({
                    "Contact/foot_support_fraction": state["foot_support_fraction"].mean(),
                } if self.contacts is not None else {}),
                **({
                    "Contact/undesired_body_count": state["undesired_contact_count"].mean(),
                    "Contact/undesired_env_fraction": (state["undesired_contact_count"] > 0).float().mean(),
                    "Contact/hip_body_count": state["hip_contact_count"].mean(),
                    "Contact/foot_body_count": state["foot_contact_count"].mean(),
                    "Contact/rear_knee_contact_count": state["rear_knee_contact_count"].mean(),
                    "Contact/rear_knee_support_count": state["rear_knee_support_count"].mean(),
                    "Contact/rear_knee_support_fraction": state["rear_knee_support_fraction"].mean(),
                    "Contact/rear_knee_and_foot_support_fraction":
                        state["rear_knee_and_foot_support_fraction"].mean(),
                    "Contact/rear_leg_supported_fraction": state["rear_leg_supported_fraction"].mean(),
                    "Contact/rear_both_unsupported_fraction": state["rear_both_unsupported_fraction"].mean(),
                    "Contact/rear_both_airborne_fraction": state["rear_feet_airborne_fraction"].mean(),
                    "Contact/all_feet_support_fraction": state["all_feet_support_fraction"].mean(),
                } if self.contacts is not None and self.detailed_diagnostics else {}),
            },
        }

    def _reset_finished_envs(self, dones):
        """仅重置本步结束的环境，其余环境继续运行。"""
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.reset(done_ids)

    def reset(self, env_ids=None):
        self._pending_policy_selection = None
        if self.on_reset is not None:
            self.on_reset()
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if self.contacts is not None:
            self.contacts.reset(env_ids)
        if self.action_activity is not None:
            self.action_activity.reset(env_ids)
        self.attitude_integral.reset(env_ids)
        self.commands.reset(env_ids)
        # 每回合从配置中的初始状态重新开始，不恢复任何历史速度或落地快照。
        pose = self.reset_root_pose[env_ids].clone()
        velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        joint_pos = self.reset_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        if self.randomization is not None:
            joint_pos = self.randomization.reset(env_ids, joint_pos)
        if self.fixed_joint_ids:
            # 随机初始角不能把锁定的关节写到硬限位之外。
            joint_pos[:, self.fixed_joint_ids] = self.reset_joint_pos[env_ids][:, self.fixed_joint_ids]
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
        self.robot.actuators.target_command.set_position_index(value=self.reset_joint_target[env_ids], env_ids=env_ids)
        self.actions[env_ids] = self.standing_actions[env_ids]
        self.raw_actions[env_ids] = self.standing_actions[env_ids]
        self.previous_raw_actions[env_ids] = self.standing_actions[env_ids]
        self.previous_actions[env_ids] = self.standing_actions[env_ids]
        self.previous_target_delta[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.settle_steps_remaining[env_ids] = self.settle_steps
        if self.settle_steps == 0 and self.reset_orientation_sampler is not None:
            self.reset_orientation_sampler.apply(env_ids)
