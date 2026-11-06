"""身体位置、朝向、速度跟踪；输出各项奖励率。"""
import torch
import torch.nn.functional as F


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw - (av*bv).sum(-1, keepdim=True),
                      aw*bv + bw*av + torch.cross(av, bv, dim=-1)), dim=-1)


def rotation_vector(q_to, q_from):
    q = F.normalize(quat_mul(F.normalize(q_to, dim=-1), quat_conjugate(F.normalize(q_from, dim=-1))), dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, q[..., :1].clamp_min(0))
    return v * torch.where(length > 1e-7, angle / length.clamp_min(1e-7), 2.0)


def gravity_local(q):
    g = torch.zeros_like(q)
    g[..., 3] = -1
    return quat_mul(quat_mul(quat_conjugate(q), g), q)[..., 1:]


def heading_error_features(actual_quat, reference_quat):
    """参考 yaw - 机身 yaw；输出 sin/cos，绕 ±π 连续。"""
    def yaw(q):
        q = F.normalize(q, dim=-1, eps=1e-8)
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    error = yaw(reference_quat) - yaw(actual_quat)
    return torch.stack((error.sin(), error.cos()), dim=-1)


def estimate_torque_squared(previous_q, previous_v, current_q, current_v, target,
                            kp, kd, torque_limit, substeps):
    """可微的周期力矩平方均值近似：插值起止状态，在各物理子步起点查询 PD。

    世界模型没有子步轨迹，此值不是实际子步力矩的精确重建。
    """
    alpha = torch.arange(substeps, device=current_q.device, dtype=current_q.dtype) / substeps
    alpha = alpha[:, None, None]
    q = previous_q[None] + alpha * (current_q - previous_q)[None]
    v = previous_v[None] + alpha * (current_v - previous_v)[None]
    tau = (kp * (target[None] - q) - kd * v).clamp(-torque_limit, torque_limit)
    return tau.square().mean(0)


def compute_reward(cfg, actual, ref, action, previous_action, joint_pos, joint_limits, dt,
                   joint_vel=None, prev_joint_vel=None, target=None, torque_squared_mean=None,
                   kp=25.0, kd=0.5, torque_limit=23.7):
    pos = (actual['pos'] - ref['pos']).square().sum(-1).mean(-1)
    ori = rotation_vector(actual['quat'], ref['quat']).square().sum(-1).mean(-1)
    vel = (actual['linear_vel'] - ref['linear_vel']).square().sum(-1).mean(-1)
    ang = (actual['angular_vel'] - ref['angular_vel']).square().sum(-1).mean(-1)
    root_pos = (actual['pos'][:, 0] - ref['pos'][:, 0]).square().sum(-1)
    root_ori = rotation_vector(actual['quat'][:, 0], ref['quat'][:, 0]).square().sum(-1)
    limits = (joint_limits[:, 0] - joint_pos).clamp_min(0) + (joint_pos - joint_limits[:, 1]).clamp_min(0)
    terms = {
        'body_position': cfg.body_position_weight * torch.exp(-pos / cfg.body_position_sigma**2),
        'body_orientation': cfg.body_orientation_weight * torch.exp(-ori / cfg.body_orientation_sigma**2),
        'body_linear_velocity': cfg.body_linear_velocity_weight * torch.exp(-vel / cfg.body_linear_velocity_sigma**2),
        'body_angular_velocity': cfg.body_angular_velocity_weight * torch.exp(-ang / cfg.body_angular_velocity_sigma**2),
        'root_position': cfg.root_position_weight * torch.exp(-root_pos / cfg.root_position_sigma**2),
        'root_orientation': cfg.root_orientation_weight * torch.exp(-root_ori / cfg.root_orientation_sigma**2),
        'action_rate': -cfg.action_rate_weight * (action - previous_action).square().sum(-1),
        'joint_limit': -cfg.joint_limit_weight * limits.sum(-1),
    }
    if cfg.joint_position_weights is not None or cfg.joint_position_weight:
        joint_sq = (joint_pos - ref['joint_pos']).square()
        if cfg.joint_position_mode == 'per_joint':
            sigma = cfg.joint_position_sigma
            if cfg.joint_sigmas is not None:
                sigma = joint_pos.new_tensor(cfg.joint_sigmas)
                if len(cfg.joint_sigmas) == 3:
                    sigma = sigma.repeat(4)
            scores = torch.exp(-joint_sq / sigma**2)
            if cfg.joint_position_weights is not None:
                # 各关节独立给分，最终相加成为一个标量奖励；权重不做归一化。
                weights = joint_pos.new_tensor(cfg.joint_position_weights)
                terms['joint_position'] = (weights * scores).sum(-1)
            else:
                # 历史 checkpoint 保留原来的平均形式与奖励数值。
                terms['joint_position'] = cfg.joint_position_weight * scores.mean(-1)
        elif cfg.joint_sigmas is None:
            joint_error = joint_sq.mean(dim=-1)
            terms['joint_position'] = cfg.joint_position_weight * torch.exp(-joint_error / cfg.joint_position_sigma**2)
        else:
            # 关节顺序 FL、FR、RL、RR，每条腿 hip、thigh、calf；四条腿同类关节取均方误差，三类各算一个指数项再平均。
            by_type = joint_sq.view(-1, 4, 3).mean(dim=1)
            sigmas = joint_pos.new_tensor(cfg.joint_sigmas)
            terms['joint_position'] = cfg.joint_position_weight * torch.exp(-by_type / sigmas**2).mean(dim=-1)
    if cfg.torque_l2_weight:
        if cfg.torque_mode == 'substep_mean':
            if torque_squared_mean is None:
                raise ValueError('substep_mean 力矩奖励需要周期内力矩平方均值')
            torque_sq = torque_squared_mean
        else:
            if joint_vel is None or target is None:
                raise ValueError('endpoint 力矩奖励需要 joint_vel 和 target')
            torque_sq = (kp * (target - joint_pos) - kd * joint_vel).clamp(-torque_limit, torque_limit).square()
        terms['torque_l2'] = -cfg.torque_l2_weight * torque_sq.sum(-1)
    if cfg.joint_acc_l2_weight:
        if joint_vel is None or prev_joint_vel is None:
            raise ValueError('joint_acc_l2_weight 非零时需要传入 joint_vel 和 prev_joint_vel')
        terms['joint_acc_l2'] = -cfg.joint_acc_l2_weight * ((joint_vel - prev_joint_vel) / dt).square().sum(-1)
    return sum(terms.values()) * dt, terms
