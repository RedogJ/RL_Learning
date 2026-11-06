"""MuJoCo Warp 并行 Go2 模仿环境；张量均保留在 GPU。"""
from dataclasses import dataclass, field
import json
import time
from functools import wraps
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .environment import Go2Environment
from .tasks.mimic.config import MimicTaskCfg
from .tasks.mimic.task import MimicTask
from .tasks.mimic.rewards import gravity_local, rotation_vector, quat_mul, heading_error_features


def warp_stream_ordered(fn):
    """让 Torch 和 Warp 使用同一 Warp 流，并在入口/出口建立流依赖。"""
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        caller = torch.cuda.current_stream(self.device)
        self.torch_warp_stream.wait_stream(caller)
        def record(value, stream):
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(stream)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    record(item, stream)
            elif isinstance(value, dict):
                for item in value.values():
                    record(item, stream)
        record(args, self.torch_warp_stream)
        record(kwargs, self.torch_warp_stream)
        try:
            with torch.cuda.stream(self.torch_warp_stream):
                result = fn(self, *args, **kwargs)
        finally:
            caller.wait_stream(self.torch_warp_stream)
        record(result, caller)
        return result
    return wrapped


@wp.kernel
def pd_control(qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
               ctrl: wp.array2d(dtype=wp.float32), target: wp.array2d(dtype=wp.float32),
               qids: wp.array(dtype=wp.int64), vids: wp.array(dtype=wp.int64), aids: wp.array(dtype=wp.int64),
               kp: float, kd: float, limit: float, torque_squared_sum: wp.array2d(dtype=wp.float32)):
    world, joint = wp.tid()
    torque = kp * (target[world, joint] - qpos[world, qids[joint]]) - kd * qvel[world, vids[joint]]
    ctrl[world, aids[joint]] = wp.clamp(torque, -limit, limit)
    torque_squared_sum[world, joint] += ctrl[world, aids[joint]] * ctrl[world, aids[joint]]


@wp.kernel
def flag_nonfoot_ground_contact(nacon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
                                worldid: wp.array(dtype=wp.int32), foot_mask: wp.array(dtype=wp.int32),
                                ground: int, flag: wp.array(dtype=wp.int32)):
    # 任一非脚掌几何体与地面接触时，把对应世界的标志置 1；标志由调用方在每个控制步前清零。
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    if g[0] == ground:
        if g[1] >= 0:
            if foot_mask[g[1]] == 0:
                flag[worldid[i]] = 1
    elif g[1] == ground:
        if g[0] >= 0:
            if foot_mask[g[0]] == 0:
                flag[worldid[i]] = 1


@dataclass
class EnvCfg:
    num_envs: int = 64
    physics_hz: int = 600
    control_hz: int = 30
    episode_seconds: float = 5.0
    action_scale: float = 1.0  # rad：目标角 = 默认角 + action_scale * action，action ∈ [-1, 1]
    action_joint_scales: tuple[float, ...] | None = None  # None 保留旧的统一映射
    clamp_joint_targets: bool = False  # 新映射可限制目标角在机械范围内
    wm_foot_features: bool = False  # 世界模型额外使用脚掌距地高度和竖直速度
    kp: float = 25.0
    kd: float = 0.5
    torque_limit: float = 23.7
    task: MimicTaskCfg = field(default_factory=MimicTaskCfg)


class Reference:
    def __init__(self, path, device, cpu, body_ids, loop=True):
        self.loop = loop
        path = Path(path)
        frames = np.load(path, allow_pickle=False)
        if frames.ndim != 2 or frames.shape[1] != 61 or not np.isfinite(frames).all():
            raise ValueError('参考动作必须是有限的 (T, 61) 数组')
        config_path = path.with_name('config.json')
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        start, end = config.get('policy_cfg', {}).get('motion_range', [0, None])
        frames = frames[slice(start, end)]
        if len(frames) < 2:
            raise ValueError('参考动作有效范围至少需要两帧')
        info_path = path.with_name('clip.json')
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        self.fps = float(info.get('fps', 60))
        if self.fps <= 0:
            raise ValueError('参考动作帧率必须大于零')
        self.frames = torch.as_tensor(frames.copy(), device=device, dtype=torch.float32)
        self.joints = self.frames[:, 6:18].reshape(-1, 3, 4).transpose(1, 2).reshape(-1, 12).contiguous()
        self.position = self.frames[:, 48:51]
        self.quaternion = torch.nn.functional.normalize(self.frames[:, 51:55], dim=-1)
        # 从参考根部与关节位置生成同一 Go2 模型各刚体的世界位姿。
        poses, quats, qposes = [], [], []
        for frame in frames:
            cpu.data.qpos[:3] = frame[48:51]
            cpu.data.qpos[3:7] = frame[51:55]
            cpu.data.qpos[cpu.qpos_ids] = frame[6:18].reshape(3, 4).T.reshape(-1)
            qposes.append(cpu.data.qpos.copy())
            mujoco.mj_kinematics(cpu.model, cpu.data)
            poses.append(cpu.data.xpos[body_ids].copy())
            quats.append(cpu.data.xquat[body_ids].copy())
        self.body_pos = torch.tensor(np.asarray(poses), device=device, dtype=torch.float32)
        self.body_quat = torch.tensor(np.asarray(quats), device=device, dtype=torch.float32)
        # MuJoCo 的 free joint 角速度在局部坐标系中；使用官方位置差分确保约定正确。
        velocities = np.zeros((len(qposes), cpu.model.nv), dtype=np.float64)
        for i in range(len(qposes)):
            lo, hi = max(0, i-1), min(len(qposes)-1, i+1)
            mujoco.mj_differentiatePos(cpu.model, velocities[i], (hi-lo)/self.fps, qposes[lo], qposes[hi])
        self.velocity = torch.tensor(velocities, device=device, dtype=torch.float32)

    def sample(self, absolute_frame):
        """首尾接续：每周期累计根部水平位移和航向，包含根部与所有跟踪刚体。"""
        period = len(self.frames) - 1
        if not self.loop:
            frame = absolute_frame.clamp(0, period).long()
            return {'joints': self.joints[frame], 'pos': self.position[frame],
                    'quat': self.quaternion[frame], 'body_pos': self.body_pos[frame],
                    'body_quat': self.body_quat[frame]}
        frame = absolute_frame.remainder(period).long()
        cycle = torch.div(absolute_frame, period, rounding_mode='floor').to(self.position.dtype)
        first, last = self.quaternion[0], self.quaternion[-1]
        def yaw(q):
            w, x, y, z = q.unbind(-1)
            return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        delta = torch.atan2(torch.sin(yaw(last)-yaw(first)), torch.cos(yaw(last)-yaw(first)))
        angle = cycle*delta
        c, si = torch.cos(angle), torch.sin(angle)
        half = (cycle-1)*delta/2
        scale = cycle * torch.sinc(cycle*delta/(2*torch.pi)) / torch.sinc(delta/(2*torch.pi))
        dx, dy = (self.position[-1, :2]-self.position[0, :2]).unbind(-1)
        sx = scale*(torch.cos(half)*dx-torch.sin(half)*dy)
        sy = scale*(torch.sin(half)*dx+torch.cos(half)*dy)
        def transform(pos, has_body=False):
            cc, ss = c, si
            tx, ty = sx, sy
            if has_body:
                cc, ss, tx, ty = (v.unsqueeze(-1) for v in (cc, ss, tx, ty))
            rel = pos[..., :2]-self.position[0, :2]
            x = self.position[0, 0]+tx+cc*rel[..., 0]-ss*rel[..., 1]
            y = self.position[0, 1]+ty+ss*rel[..., 0]+cc*rel[..., 1]
            return torch.stack((x, y, pos[..., 2]), dim=-1)
        yaw_q = torch.stack((torch.cos(angle/2), torch.zeros_like(angle), torch.zeros_like(angle), torch.sin(angle/2)), dim=-1)
        return {'joints': self.joints[frame], 'pos': transform(self.position[frame]),
                'quat': quat_mul(yaw_q, self.quaternion[frame]),
                'body_pos': transform(self.body_pos[frame], True),
                'body_quat': quat_mul(yaw_q.unsqueeze(-2), self.body_quat[frame])}


def longitudinal_features(ref, frame, frame_limit, position, velocity_world, dt):
    """参考前向的 (参考-实体) 位置 m、速度 m/s；参考速度按下个控制步差分。"""
    current = ref.sample(frame)
    advance = max(1, int(round(dt * ref.fps)))
    next_frame = torch.minimum(frame + advance, frame_limit)
    future = ref.sample(next_frame)
    ref_velocity = (future['pos'] - current['pos']) / dt
    w, x, y, z = current['quat'].unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    forward = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
    pos_error = ((current['pos'][..., :2]-position[..., :2])*forward).sum(-1)
    vel_error = ((ref_velocity[..., :2]-velocity_world[..., :2])*forward).sum(-1)
    return torch.stack((pos_error, vel_error), dim=-1)


class WarpGo2Env:
    def __init__(self, cfg, motion):
        if cfg.num_envs <= 0 or cfg.episode_seconds <= 0:
            raise ValueError('环境数量与回合时长必须为正')
        self.cfg = cfg
        self.profile = None
        self.physics_graph = None
        self.task = MimicTask(cfg.task)
        self.device = torch.device('cuda:0')
        wp.init()
        wp.set_device('cuda:0')
        self.warp_stream = wp.get_stream('cuda:0')
        self.torch_warp_stream = torch.cuda.ExternalStream(self.warp_stream.cuda_stream, device=self.device)
        cpu = Go2Environment(cfg.physics_hz, cfg.control_hz)
        self.substeps = cpu.substeps
        self.dt = cpu.control_dt
        # Warp 不支持 MULTICCD 与非零碰撞 margin 的组合。
        cpu.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD) | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.model = mjw.put_model(cpu.model)
        # 为跌倒后的多点接触预留约束和接触容量。
        self.data = mjw.put_data(cpu.model, cpu.data, nworld=cfg.num_envs, njmax=256, nconmax=128)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.qids = torch.tensor(cpu.qpos_ids, device=self.device, dtype=torch.long)
        self.vids = torch.tensor(cpu.dof_ids, device=self.device, dtype=torch.long)
        self.aids = torch.tensor(cpu.actuator_ids, device=self.device, dtype=torch.long)
        # 非脚掌接触判定：地面 geom 与四个脚掌 geom（FL、FR、RL、RR）。
        self.ground_geom = mujoco.mj_name2id(cpu.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
        foot_geoms = [mujoco.mj_name2id(cpu.model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in ('FL', 'FR', 'RL', 'RR')]
        if self.ground_geom < 0 or min(foot_geoms) < 0:
            raise ValueError('模型缺少地面或脚掌 geom')
        self.foot_mask = torch.zeros(cpu.model.ngeom, dtype=torch.int32, device=self.device)
        self.foot_mask[foot_geoms] = 1
        self.foot_mask_wp = wp.from_torch(self.foot_mask, dtype=wp.int32)
        self.contact_flag = torch.zeros(cfg.num_envs, dtype=torch.int32, device=self.device)
        self.contact_flag_wp = wp.from_torch(self.contact_flag, dtype=wp.int32)
        self.contact_fall = torch.zeros(cfg.num_envs, dtype=torch.bool, device=self.device)
        names = ['base_link'] + [f'{leg}_{part}' for leg in ('FL', 'FR', 'RL', 'RR')
                                 for part in ('hip', 'thigh', 'calf', 'foot')]
        body_ids = [mujoco.mj_name2id(cpu.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
        if min(body_ids) < 0:
            raise ValueError('模型缺少跟踪刚体')
        self.bids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        from .tasks.mimic.kinematics import BodyKinematics
        self.kinematics = BodyKinematics(cpu.model, body_ids, cpu.joint_ids).to(self.device)
        self.xpos = wp.to_torch(self.data.xpos)
        self.xquat = wp.to_torch(self.data.xquat)
        self.default_q = torch.tensor(cpu.default_q, device=self.device, dtype=torch.float32)
        self.joint_limits = torch.tensor(cpu.model.jnt_range[cpu.joint_ids].copy(), device=self.device, dtype=torch.float32)
        scales = cfg.action_joint_scales
        if scales is not None and (len(scales) != 12 or any(not np.isfinite(x) or x <= 0 for x in scales)):
            raise ValueError('action_joint_scales 必须包含 12 个有限正数')
        self.action_scales = torch.tensor(scales if scales is not None else [cfg.action_scale] * 12, device=self.device)
        self.ref = Reference(motion, self.device, cpu, body_ids, loop=cfg.task.reference_loop)
        self.frame = torch.zeros(cfg.num_envs, device=self.device, dtype=torch.long)
        self.steps = torch.zeros_like(self.frame)
        self.start_frame = torch.zeros_like(self.frame)
        self.frame_limit = torch.full_like(self.frame, (len(self.ref.frames)-1)*cfg.task.reference_cycles)
        self.previous_action = torch.zeros((cfg.num_envs, 12), device=self.device)
        self.previous_joint_vel = torch.zeros((cfg.num_envs, 12), device=self.device)  # 上一控制步末的关节速度，用于加速度惩罚
        self.torque_squared_sum = torch.zeros((cfg.num_envs, 12), device=self.device)
        self.torque_squared_sum_wp = wp.from_torch(self.torque_squared_sum, dtype=wp.float32)
        self.history = torch.zeros((cfg.num_envs, cfg.task.history_length, 42), device=self.device)
        self.reset(torch.ones(cfg.num_envs, device=self.device, dtype=torch.bool))

    @warp_stream_ordered
    def reset(self, mask, rsi_probabilities=None, rsi_frames_per_bin=None, start_probability=0.):
        # 参考阶段与回合计时分开；历史以重置后的真实初始观测重复填充。
        ids = mask.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return
        reset_mask = wp.from_torch(mask.contiguous(), dtype=wp.bool)
        mjw.reset_data(self.model, self.data, reset_mask)
        period = len(self.ref.frames)-1
        if rsi_probabilities is not None:
            if rsi_frames_per_bin is None or rsi_frames_per_bin <= 0:
                raise ValueError('RSI 分箱帧数必须为正')
            selected = torch.multinomial(rsi_probabilities, ids.numel(), replacement=True)
            begin = selected * rsi_frames_per_bin
            lengths = torch.minimum(torch.full_like(begin, rsi_frames_per_bin), period-begin)
            phase = begin + torch.floor(torch.rand(ids.numel(), device=self.device)*lengths).long()
        else:
            phase = (torch.randint(period, (ids.numel(),), device=self.device)
                     if self.cfg.task.random_reference_start else torch.zeros_like(ids))
        if start_probability:
            phase = torch.where(torch.rand(ids.numel(), device=self.device) < start_probability, 0, phase)
        self.start_frame[ids] = phase
        self.frame[ids] = phase
        self.frame_limit[ids] = (phase + period*self.cfg.task.reference_cycles if self.cfg.task.reference_loop
                                 else torch.full_like(phase, period))
        self.steps[ids] = 0
        self.previous_action[ids] = 0
        self.qpos[ids, :3] = self.ref.position[phase]
        self.qpos[ids, 3:7] = self.ref.quaternion[phase]
        self.qpos[ids[:, None], self.qids] = self.ref.joints[phase]
        self.qvel[ids] = self.ref.velocity[phase]
        # 重置步没有有意义的上一步速度：取当前速度，使该步加速度为 0。
        self.previous_joint_vel[ids] = self.qvel[ids][:, self.vids]
        mjw.forward(self.model, self.data)
        initial = self.proprioception()
        self.history[ids] = initial[ids, None, :]

    def proprioception(self):
        q = torch.nn.functional.normalize(self.qpos[:, 3:7], dim=-1)
        parts = [self.qpos[:, self.qids] - self.default_q, self.qvel[:, self.vids],
                 self.qvel[:, 3:6], gravity_local(q), self.previous_action]
        if self.cfg.task.observation_noise:
            scales = (0.01, 0.5, 0.2, 0.05, 0.0)
            parts = [v + (torch.rand_like(v) * 2 - 1) * scale for v, scale in zip(parts, scales)]
        return torch.cat(parts, dim=-1)

    def observation(self):
        # 未来目标按控制频率采样；片段尾部重复最后一帧。
        offsets = torch.arange(1, self.cfg.task.future_length + 1, device=self.device)
        indices = torch.minimum(self.frame[:, None] + torch.floor(offsets * self.dt * self.ref.fps).long(), self.frame_limit[:, None])
        ref = self.ref.sample(indices)
        goals = torch.cat((ref['joints'], ref['pos'], ref['quat']), dim=-1)
        parts = [self.history.flatten(1), goals.flatten(1)]
        if self.cfg.task.heading_observation:
            current_ref = self.ref.sample(self.frame)
            parts.append(heading_error_features(self.qpos[:, 3:7], current_ref['quat']))
        if self.cfg.task.longitudinal_observation:
            parts.append(longitudinal_features(self.ref, self.frame, self.frame_limit,
                                               self.qpos[:, :3], self.qvel[:, :3], self.dt))
        return torch.cat(parts, dim=-1)

    def _flag_nonfoot_contacts(self):
        # 在每个物理子步后累积：任一非脚掌几何体接触地面则置位，避免只读到控制周期最后一个子步的接触。
        if self.cfg.task.terminate_on_nonfoot_contact:
            wp.launch(flag_nonfoot_ground_contact, dim=self.data.naconmax,
                      inputs=[self.data.nacon, self.data.contact.geom, self.data.contact.worldid,
                              self.foot_mask_wp, self.ground_geom, self.contact_flag_wp])

    def _physics_substeps(self, target):
        for _ in range(self.substeps):
            torque = self.cfg.kp * (target - self.qpos[:, self.qids]) - self.cfg.kd * self.qvel[:, self.vids]
            self.ctrl[:, self.aids] = torque.clamp(-self.cfg.torque_limit, self.cfg.torque_limit)
            self.torque_squared_sum.add_(self.ctrl[:, self.aids].square())
            mjw.step(self.model, self.data)
            self._flag_nonfoot_contacts()

    def enable_cuda_graph(self):
        """录制固定数量的物理子步；调用后重置所有环境。"""
        if self.physics_graph is not None:
            return
        self.graph_target = torch.empty((self.cfg.num_envs, 12), device=self.device)
        self.graph_target.copy_(self.default_q.expand_as(self.graph_target))
        self.graph_target_wp = wp.from_torch(self.graph_target, dtype=wp.float32)
        def physics():
            for _ in range(self.substeps):
                wp.launch(pd_control, dim=(self.cfg.num_envs, 12), inputs=[self.data.qpos, self.data.qvel, self.data.ctrl,
                          self.graph_target_wp, wp.from_torch(self.qids), wp.from_torch(self.vids), wp.from_torch(self.aids),
                          self.cfg.kp, self.cfg.kd, self.cfg.torque_limit, self.torque_squared_sum_wp])
                mjw.step(self.model, self.data)
                self._flag_nonfoot_contacts()
            # 控制周期末更新运动学，并入 Graph；step 在有 Graph 时不再单独调用。
            mjw.forward(self.model, self.data)
        torch.cuda.synchronize()
        physics()
        wp.synchronize_device()
        wp.capture_begin(device='cuda:0')
        physics()
        self.physics_graph = wp.capture_end(device='cuda:0')
        self.reset(torch.ones(self.cfg.num_envs, device=self.device, dtype=torch.bool))

    def _profile_mark(self, label):
        if self.profile is not None:
            torch.cuda.synchronize()
            now = time.perf_counter()
            if self._profile_last is not None:
                self.profile[label] = self.profile.get(label, 0.0) + now - self._profile_last
            self._profile_last = now

    @warp_stream_ordered
    def step(self, action):
        self._profile_last = None
        self._profile_mark('start')
        cfg = self.cfg
        if action.shape != (cfg.num_envs, 12):
            raise ValueError('动作形状应为 (num_envs, 12)')
        action = action.clamp(-1, 1)
        idx = self.frame.clamp(max=len(self.ref.frames)-1)
        target = self.default_q + self.action_scales * action
        if cfg.clamp_joint_targets:
            target = target.clamp(self.joint_limits[:, 0], self.joint_limits[:, 1])
        old_pos = self.xpos[:, self.bids].clone()
        old_quat = self.xquat[:, self.bids].clone()
        self._profile_mark('prepare')
        self.contact_flag.zero_()
        self.torque_squared_sum.zero_()
        if self.physics_graph is None:
            self._physics_substeps(target)
        else:
            self.graph_target.copy_(target)
            wp.capture_launch(self.physics_graph, stream=self.warp_stream)
        self._profile_mark('physics')
        self.steps += 1
        # 连续拼接参考周期，只在最终周期结束、超时或失败后重置。
        next_frame = self.start_frame + torch.floor(self.steps.float() * self.dt * self.ref.fps).long()
        ref_idx = torch.minimum(next_frame, self.frame_limit)
        # mj_step 后显式更新运动学，速度由控制周期内的位姿变化估计。
        # 使用 CUDA Graph 时，forward 已包含在 Graph 末尾。
        if self.physics_graph is None:
            mjw.forward(self.model, self.data)
        self._profile_mark('forward')
        pos, quat = self.xpos[:, self.bids], self.xquat[:, self.bids]
        prev_idx = self.frame
        target_ref = self.ref.sample(ref_idx)
        prev_ref = self.ref.sample(prev_idx)
        actual = {'pos': pos, 'quat': quat, 'linear_vel': (pos-old_pos)/self.dt,
                  'angular_vel': rotation_vector(quat, old_quat)/self.dt}
        ref = {'pos': target_ref['body_pos'], 'quat': target_ref['body_quat'],
               'joint_pos': target_ref['joints'],
               'linear_vel': (target_ref['body_pos']-prev_ref['body_pos'])/self.dt,
               'angular_vel': rotation_vector(target_ref['body_quat'], prev_ref['body_quat'])/self.dt}
        joint_vel = self.qvel[:, self.vids]
        reward, terms = self.task.reward(actual, ref, action, self.previous_action,
                                         self.qpos[:, self.qids], self.joint_limits, self.dt,
                                         joint_vel=joint_vel, prev_joint_vel=self.previous_joint_vel, target=target,
                                         torque_squared_mean=self.torque_squared_sum/self.substeps,
                                         kp=cfg.kp, kd=cfg.kd, torque_limit=cfg.torque_limit)
        self.previous_action.copy_(action)
        self.previous_joint_vel.copy_(joint_vel)
        self.frame.copy_(ref_idx)
        self.history = torch.cat((self.history[:, 1:], self.proprioception()[:, None]), dim=1)
        terminated = self.task.terminated(self.qpos[:, :3], self.qpos[:, 3:7],
                                          target_ref['pos'], target_ref['quat'])
        # 任一非脚掌几何体接触地面即失败；接触在 Graph 内逐物理子步累积，不会漏掉控制周期中间的触地。
        self.contact_fall = self.contact_flag > 0
        if cfg.task.terminate_on_nonfoot_contact:
            terminated = terminated | self.contact_fall
        # 失败惩罚是一次性的回报，不属于每秒奖励率，不乘 dt。
        reward = reward - self.cfg.task.reward.failure_penalty * terminated.float()
        truncated = (next_frame >= self.frame_limit) | (self.steps * self.dt >= cfg.episode_seconds)
        obs = self.observation()
        self._profile_mark('reward_observation')
        return obs, reward, terminated, truncated, terms
