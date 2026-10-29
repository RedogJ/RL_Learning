"""近竖直姿态的独立采样；使用 xyzw 四元数。"""

import math

import torch


def multiply(a, b):
    """四元数乘法，支持批量输入。"""
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return torch.cat((
        aw * bv + bw * av + torch.linalg.cross(av, bv),
        aw * bw - (av * bv).sum(-1, keepdim=True),
    ), dim=-1)


def rotate(q, v):
    """用单位四元数旋转向量。"""
    cross = 2 * torch.linalg.cross(q[..., :3], v)
    return v + q[..., 3:] * cross + torch.linalg.cross(q[..., :3], cross)


class ResetOrientationSampler:
    def __init__(self, env, cone_degrees, seed):
        self.env = env
        self.cone = math.radians(cone_degrees)
        self.generator = torch.Generator(device=env.device).manual_seed(seed)
        self.foot_ids = [env.robot.body_names.index(f"{leg}_foot") for leg in ("RL", "RR")]
        self.nominal = env.reset_root_pose[:, 3:].clone()
        self.last_tilt_degrees = torch.zeros(env.num_envs, device=env.device)

    def sample(self, nominal):
        # cos(theta) 和方位角均匀：圆锥内方向按立体角均匀，而非 Euler 角均匀。
        u = torch.rand((len(nominal), 2), device=nominal.device, generator=self.generator)
        cos_theta = 1 - u[:, 0] * (1 - math.cos(self.cone))
        theta = cos_theta.clamp(-1, 1).acos()
        phi = 2 * math.pi * u[:, 1]
        half = theta * 0.5
        delta = torch.stack((
            half.sin() * phi.cos(), half.sin() * phi.sin(),
            torch.zeros_like(half), half.cos(),
        ), dim=-1)
        return multiply(delta, nominal), theta * 180 / math.pi

    @torch.no_grad()
    def apply(self, ids):
        env = self.env
        data = env.robot.data
        old_p = data.root_link_pos_w.torch[ids].clone()
        old_q = data.root_link_quat_w.torch[ids].clone()
        q, tilt = self.sample(self.nominal[ids])
        conjugate = old_q.clone()
        conjugate[:, :3].neg_()
        delta = multiply(q, conjugate)
        feet = data.body_link_pos_w.torch[ids][:, self.foot_ids].clone()
        pivot = feet.mean(1)
        p = pivot + rotate(delta, old_p - pivot)
        rotated_feet = pivot[:, None] + rotate(
            delta[:, None].expand(-1, 2, -1), feet - pivot[:, None],
        )
        # 保持最低后脚点原有高度；倾斜时允许另一脚稍微抬起。
        lift = feet[:, :, 2].min(-1).values - rotated_feet[:, :, 2].min(-1).values
        p[:, 2] += lift
        env.robot.write_root_pose_to_sim_index(root_pose=torch.cat((p, q), -1), env_ids=ids)
        env.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(ids), 6), device=env.device), env_ids=ids,
        )
        env.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros((len(ids), env.num_actions), device=env.device), env_ids=ids,
        )
        self.last_tilt_degrees[ids] = tilt
