"""同类关节的近期指令活动量；不进入策略观测。"""
import math
import torch


class ActionActivity:
    def __init__(self, joint_names, num_envs, device, dt, tau):
        names = list(joint_names)
        groups = [[names.index(f"{leg}_{kind}_joint") for leg in ("FL", "FR", "RL", "RR")]
                  for kind in ("hip", "thigh", "calf")]
        self.groups = torch.tensor(groups, device=device)
        self.values = torch.zeros((num_envs, len(names)), device=device)
        self.dt = dt
        self.alpha = -math.expm1(-dt / tau)

    def update(self, actions, previous_actions, active):
        # 使用裁剪后指令的变化速度绝对值；相反方向同样计入活动量。
        speed = (actions - previous_actions).abs() / self.dt
        updated = self.values + self.alpha * (speed - self.values)
        self.values.copy_(torch.where(active[:, None], updated, 0.0))

    def penalty(self):
        # (环境, 三种关节, 四条腿)，总体方差不使用无偏估计。
        return self.values[:, self.groups].var(dim=-1, unbiased=False).sum(dim=-1)

    def reset(self, env_ids):
        self.values[env_ids] = 0.0
