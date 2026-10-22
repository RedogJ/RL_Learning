"""单环境 MuJoCo 控制循环：30 Hz 策略目标，物理子步重算 PD 力矩。"""

import math

import mujoco
import numpy as np

from .config import INITIAL_HEIGHT, KD, KP, TORQUE_LIMIT
from .observations import Observations
from .robot import load_model


class Go2Sim:
    def __init__(self, policy, physics_hz):
        self.policy = policy
        self.model, self.joint_ids, self.actuator_ids = load_model(policy, physics_hz)
        self.data = mujoco.MjData(self.model)
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.substeps = round(policy.control_dt / self.model.opt.timestep)
        if self.substeps < 1 or not math.isclose(self.substeps * self.model.opt.timestep, policy.control_dt):
            raise ValueError("物理频率必须是策略频率的整数倍")
        self.observations = Observations(policy)
        self.command = np.zeros(3)
        self.last_action = policy.standing_action.copy()
        self.reset_count = 0
        self.total_steps = 0
        self.episode_steps = 0
        self.last_raw_action = self.last_action.copy()
        self.last_torque = np.zeros(12)
        self.reset()

    @property
    def active(self):
        return self.settling_remaining == 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0., 0., INITIAL_HEIGHT]
        self.data.qpos[3:7] = [1., 0., 0., 0.]  # MuJoCo 四元数 wxyz。
        self.data.qpos[self.qpos_ids] = self.policy.default_q
        self.data.qvel[:] = 0.
        self.data.ctrl[:] = 0.
        self.data.xfrc_applied[:] = 0.
        self.last_action[:] = self.policy.standing_action
        self.last_raw_action[:] = self.last_action
        self.observations.reset()
        self.settling_remaining = math.ceil(self.policy.cfg["env_cfg"]["reset_settle_seconds"] / self.policy.control_dt)
        self.episode_steps = 0
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        velocity = np.zeros(6)
        # BODY 返回根刚体质心的世界速度；再用 link 的旋转矩阵转换到机身坐标。
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
        return {"linear_velocity": rotation.T @ velocity[3:],
                "angular_velocity": rotation.T @ velocity[:3],
                "gravity": rotation.T @ np.array([0., 0., -1.]),
                "q": self.data.qpos[self.qpos_ids].copy(),
                "dq": self.data.qvel[self.dof_ids].copy(),
                "height": float(self.data.xpos[self.base_id, 2])}

    def observation(self):
        return self.observations.build(self.state(), self.last_action, self.command)

    def step(self, raw_action):
        active = self.active
        self.last_raw_action = np.asarray(raw_action if active else self.policy.standing_action).copy()
        if not np.isfinite(self.last_raw_action).all():
            raise FloatingPointError("非有限动作")
        self.last_action[:] = np.clip(self.last_raw_action, -1., 1.)
        target = self.policy.target(self.last_action)
        for _ in range(self.substeps):
            torque = KP * (target - self.data.qpos[self.qpos_ids]) - KD * self.data.qvel[self.dof_ids]
            self.last_torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
            self.data.ctrl[self.actuator_ids] = self.last_torque
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo 状态非有限")
        state = self.state()
        if active:
            self.observations.update(state)
            self.episode_steps += 1
        else:
            self.settling_remaining -= 1
        self.total_steps += 1
        limit = -math.cos(math.radians(self.policy.cfg["termination_cfg"]["maximum_tilt_degrees"]))
        fallen = active and state["gravity"][2] > limit
        timeout = active and self.episode_steps * self.policy.control_dt >= self.policy.cfg["env_cfg"]["episode_seconds"]
        return state, "tilt" if fallen else "timeout" if timeout else None
