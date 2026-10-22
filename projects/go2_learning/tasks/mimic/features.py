"""学生本体信息与扩散物理状态；固定字段次序，避免混入 expert reference。

扩散状态坐标使用瞬时机身系（不是论文 yaw-centric 系），为 Go2 upright 避免
机身 X 轴近乎竖直时的 yaw 退化。当前阶段仅支持本体速度/直立引导，不支持世界航点。
"""
import torch


def inverse_rotate(q, vectors):
    # Isaac Lab 3 xyzw；q=(N,4)，vectors=(N,B,3)。
    xyz = q[:, None, :3]
    uv = torch.cross(xyz.expand_as(vectors), vectors, dim=-1)
    return vectors - 2 * q[:, None, 3:4] * uv + 2 * torch.cross(xyz.expand_as(vectors), uv, dim=-1)


def extract(env, expert):
    obs = env.get_observations()['policy']
    d = env.robot.data
    height = d.root_link_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3]
    proprio = torch.cat((obs[:, :45], height, obs[:, 49:52]), -1)
    onehot = obs.new_zeros((env.num_envs, 2))
    onehot[:, 0 if expert == 'running' else 1] = 1.
    intent = torch.cat((onehot, env.commands.values,
                        height.new_full(height.shape, env.task.target_height)), -1)
    positions = d.body_link_pos_w.torch - d.root_link_pos_w.torch[:, None]
    velocities = d.body_com_vel_w.torch[:, :, :3] - d.root_com_vel_w.torch[:, None, :3]
    q = d.root_link_quat_w.torch
    local_pos, local_vel = inverse_rotate(q, positions), inverse_rotate(q, velocities)
    # 0:3 body linear velocity; 3:6 angular; 6:9 gravity; 9:21 q offset;
    # 21:33 dq*0.1; 33 root height; 34: body positions then body velocities.
    state = torch.cat((obs[:, :33], height, local_pos.flatten(1), local_vel.flatten(1)), -1)
    return proprio, intent, state
