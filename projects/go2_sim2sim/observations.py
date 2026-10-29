"""复现 Isaac 工程的 52 维观测与逐控制周期积分时序。"""

import math

import numpy as np


class Observations:
    def __init__(self, policy):
        self.default_q = policy.default_q
        self.target_height = policy.cfg["reward_cfg"]["target_height"]
        cfg = policy.cfg["integral_cfg"]
        self.enabled = cfg.get("enabled", True)
        self.dt = policy.control_dt
        self.decay = math.exp(-self.dt / cfg["decay_seconds"])
        self.limits = np.array([cfg["limit"], cfg["limit"], cfg["height_limit"]])
        self.integrals = np.zeros(3)

    @staticmethod
    def angles(gravity):
        roll = np.arctan2(-gravity[1], -gravity[2])
        pitch = np.arctan2(gravity[0], np.linalg.norm(gravity[1:]))
        return roll, pitch

    def reset(self):
        self.integrals[:] = 0.

    def update(self, state):
        if not self.enabled:
            return
        roll, pitch = self.angles(state["gravity"])
        errors = np.array([roll, pitch, state["height"] - self.target_height])
        self.integrals[:] = np.clip(self.integrals * self.decay + errors * self.dt,
                                    -self.limits, self.limits)

    def build(self, state, previous_action, command):
        return np.concatenate((
            state["linear_velocity"], state["angular_velocity"], state["gravity"],
            state["q"] - self.default_q, 0.1 * state["dq"], previous_action, command,
            [state["height"] - self.target_height], self.integrals / self.limits,
        )).astype(np.float32)
