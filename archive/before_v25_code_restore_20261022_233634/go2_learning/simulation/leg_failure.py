"""回放中的单腿断电：关闭隐式 PD 驱动，保留刚体、碰撞与被动关节属性。"""

import math


class LegPowerCut:
    """每回合经过若干正式控制步后断电；只在故障和恢复时写 PhysX 属性。"""

    PROPERTIES = (
        ("joint_stiffness", "write_joint_stiffness_to_sim_index", "stiffness"),
        ("joint_damping", "write_joint_damping_to_sim_index", "damping"),
        ("joint_effort_limits", "write_joint_effort_limit_to_sim_index", "limits"),
    )

    def __init__(self, robot, leg, after_seconds, control_dt):
        if leg not in ("FL", "FR", "RL", "RR"):
            raise ValueError(f"未知腿名：{leg}")
        if not math.isfinite(after_seconds) or after_seconds < 0:
            raise ValueError("断电延迟必须非负且有限")
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("控制周期必须为正数且有限")
        self.robot, self.leg, self.dt = robot, leg, control_dt
        self.names = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf")]
        self.joint_ids = [list(robot.joint_names).index(name) for name in self.names]
        for _, method, _ in self.PROPERTIES:
            if not callable(getattr(robot, method, None)):
                raise RuntimeError(f"当前执行器不支持断电所需接口：{method}")
        self.original = {
            field: getattr(robot.data, field).torch[:, self.joint_ids].clone()
            for field, _, _ in self.PROPERTIES
        }
        self.delay_steps = math.ceil(after_seconds / control_dt - 1e-9)
        self.elapsed_steps = 0
        self.is_cut = False

    def before_control_step(self):
        """仅在策略正式执行时调用；Python 计步，不读取 GPU 状态或逐步写属性。"""
        if not self.is_cut and self.elapsed_steps >= self.delay_steps:
            for _, method, argument in self.PROPERTIES:
                getattr(self.robot, method)(**{argument: 0.0, "joint_ids": self.joint_ids})
            self.is_cut = True
            print(f"[单腿断电] {self.leg}：{', '.join(self.names)}；"
                  f"正式控制 {self.elapsed_steps * self.dt:.2f}s 后关闭驱动，关节可被动运动。", flush=True)
        self.elapsed_steps += 1

    def reset(self):
        """故障后恢复健康驱动，再由原环境执行落地等待；下一回合重新计时。"""
        if self.is_cut:
            for field, method, argument in self.PROPERTIES:
                getattr(self.robot, method)(**{argument: self.original[field], "joint_ids": self.joint_ids})
        self.is_cut = False
        self.elapsed_steps = 0

    def close(self):
        self.reset()
