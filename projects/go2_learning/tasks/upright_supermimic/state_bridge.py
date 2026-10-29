"""Isaac 状态到世界模型状态，以及可微 upright 奖励的桥接。"""
import math
import torch

POLICY_DIM = 52
# 附加：世界线速度3、脚掌高度4、支撑4、接触4、脚掌速度8、接触统计5、推力上下文8、后小腿高度2。
STATE_DIM = 90


def observe(env, policy_obs):
    data = env.robot.data
    contacts = env.contacts.read()
    foot_ids = [env.robot.body_names.index(env.contacts.sensor.body_names[i]) for i in env.contacts.foot_ids]
    foot_z = data.body_link_pos_w.torch[:, foot_ids, 2] - env.scene.env_origins[:, 2:3]
    foot_xy = data.body_com_lin_vel_w.torch[:, foot_ids, :2].reshape(env.num_envs, 8)
    counts = torch.stack([contacts[k] for k in ('undesired_contact_count', 'rear_knee_support_count',
                                                'rear_knee_contact_count', 'hip_contact_count',
                                                'rear_feet_airborne_fraction')], -1).float()
    if env.randomization is None:
        push = policy_obs.new_zeros((env.num_envs, 8))
    else:
        randomizer = env.randomization
        active = (randomizer.remaining_steps > 0).to(policy_obs.dtype).unsqueeze(-1)
        push = torch.cat((randomizer.wait_steps[:,None]*env.physics_dt,
                          randomizer.remaining_steps[:,None]*env.physics_dt,
                          randomizer.forces[:,0]*active,
                          randomizer.point_offsets_b[:,0]*active), -1)
    knees = [env.robot.body_names.index(f'{leg}_calf') for leg in ('RL','RR')]
    knee_z = data.body_link_pos_w.torch[:, knees, 2] - env.scene.env_origins[:, 2:3]
    return torch.cat((policy_obs, data.root_com_vel_w.torch[:, :3], foot_z,
                      contacts['foot_support_per_foot'], contacts['foot_contact'].float(), foot_xy, counts, push, knee_z), -1)


def predicted_state(x, action, env, previous_state=None):
    """将连续预测值映射回 upright 奖励所需字段；接触代理并非真实接触传感器。"""
    obs = x[:, :POLICY_DIM]
    joint = obs[:, 9:21] + env.default_joint_pos[0]
    previous = (x if previous_state is None else previous_state)[:, 33:45]
    filtered = env.action_filter_alpha * previous + (1-env.action_filter_alpha) * action.clamp(-1,1)
    vel_xy = x[:, 67:75].reshape(-1,4,2)
    foot_vel = torch.cat((vel_xy, torch.zeros_like(vel_xy[..., :1])), -1)
    return {
        'gravity':obs[:,6:9], 'height':obs[:,48]+env.task.target_height,
        'linear_velocity_w':x[:,52:55], 'angular_velocity':obs[:,3:6],
        'joint_velocity':obs[:,21:33]*10,
        'hip_joint_pos':joint[:,env.hip_ordered_ids], 'thigh_joint_pos':joint[:,env.thigh_ordered_ids],
        'calf_joint_pos':joint[:,env.calf_ordered_ids], 'foot_velocity_w':foot_vel,
        'foot_support_per_foot':x[:,59:63].clamp(0,1), 'foot_contact':x[:,63:67].clamp(0,1),
        'undesired_contact_count':x[:,75].clamp(0, 18),
        'rear_knee_support_count':x[:,76].clamp(0,2), 'rear_knee_contact_count':x[:,77].clamp(0,2),
        'hip_contact_count':x[:,78].clamp(0,4), 'rear_feet_airborne_fraction':x[:,79].clamp(0,1),
        'raw_actions':action, 'previous_raw_actions':previous,
        'joint_target_delta':(filtered-previous)*env.action_half_range[0],
        'rear_knee_height':x[:,88:90].amin(-1),
    }


def analytical_reward(x, action, env, previous_state=None):
    state = predicted_state(x, action, env, previous_state)
    failed = torch.zeros(x.shape[0],device=x.device,dtype=torch.bool)
    parts = env.task.reward_components(state, action, failed, env.dt)
    return sum(parts.values())


def soft_failure(x, env, softness=.04, height_softness=.02, contact_softness=.2):
    """连续近似原任务的角度、高度和非脚掌接触终止。"""
    alignment = -x[:,6]
    angle_limit = math.cos(math.radians(env.task.cfg.termination.maximum_tilt_degrees))
    p_angle = torch.sigmoid((angle_limit-alignment)/softness)
    height=x[:,48]+env.task.target_height
    p_height=torch.sigmoid((env.task.cfg.termination.minimum_height-height)/height_softness)
    if env.task.cfg.termination.reset_on_nonfoot_contact:
        p_contact=1-torch.exp(-x[:,75].clamp(min=0)/contact_softness)
    else:
        p_contact=torch.zeros_like(p_angle)
    return 1-(1-p_angle)*(1-p_height)*(1-p_contact)


def next_policy_action(previous, raw, alpha):
    return alpha*previous + (1-alpha)*raw.clamp(-1,1)


def enforce_action_state(x, old, action, alpha):
    """上一次执行动作由已知滤波器精确更新，不让网络学习这一确定性部分。"""
    filtered=next_policy_action(old[:,33:45],action,alpha)
    return torch.cat((x[:,:33],filtered,x[:,45:]),-1)


def enforce_context(x, old, dt, observed_next=None):
    """训练时使用真实下一推力上下文；想象时延续当前已知外力，不猜测未来随机抽样。"""
    if observed_next is not None:
        context = observed_next[:,80:88]
    else:
        remaining=(old[:,81:82]-dt).clamp_min(0)
        pushing=(old[:,81:82]>0).to(old.dtype)
        wait=(old[:,80:81]-dt*(1-pushing)).clamp_min(0)
        context=torch.cat((wait,remaining,old[:,82:88]*(remaining>0).to(old.dtype)),dim=-1)
    return torch.cat((x[:,:80],context,x[:,88:]),-1)
