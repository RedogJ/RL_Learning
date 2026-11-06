"""连续状态世界模型训练：真实采集、八步状态拟合、模型想象与 actor/critic。"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from .train import Replay
from .networks import MotionPolicy
from .adaptive_rsi import AdaptiveRSI
from .backflip_reward_cfg import make_backflip_reward_cfg
from .sideflip_reward_cfg import make_sideflip_reward_cfg
from .reward_cfg import RewardCfg, saved_reward_config
from .state_model import Dynamics, StateValue, TerminationHead, LAYOUT, build_state, obs_from_state, critic_remaining_time
from .rewards import gravity_local, rotation_vector, heading_error_features, estimate_torque_squared
from ...warp_env import EnvCfg, WarpGo2Env, longitudinal_features


def resolve_wm_updates(num_envs, collect_steps, batch, window, ratio, updates=None):
    """按名义窗口步数/新增 transition 数设置训练量；窗口重叠与终止掩码不在此扣除。"""
    if updates is not None:
        return updates
    return max(1, math.ceil(ratio * num_envs * collect_steps / (batch * window)))


def reference_critic_mask(frame, frame_limit):
    """保留参考动作内的当前状态，包括下一步进入终点的状态。"""
    return frame < frame_limit


def fit_critic(critic, optimizer, states, goals, contexts, targets, epochs, minibatches,
               *, histories=None, remaining_times=None):
    """同一批固定 λ-return 目标，打乱后拟合多遍；每个样本每遍恰好使用一次。"""
    states, goals, contexts, targets = (x.detach() for x in (states, goals, contexts, targets))
    histories = histories.detach() if histories is not None else None
    remaining_times = remaining_times.detach() if remaining_times is not None else None
    n = targets.numel()
    if epochs <= 0 or not 1 <= minibatches <= n:
        raise ValueError('Critic epochs 必须为正，minibatches 必须在 1 到样本数之间')
    with torch.no_grad():
        before = (critic(states, goals, contexts, histories, remaining_times) - targets).square().mean()
    losses = []
    for _ in range(epochs):
        for ids in torch.randperm(n, device=targets.device).tensor_split(minibatches):
            h = histories[ids] if histories is not None else None
            time = remaining_times[ids] if remaining_times is not None else None
            loss = (critic(states[ids], goals[ids], contexts[ids], h, time) - targets[ids]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach() * ids.numel())
    with torch.no_grad():
        after = (critic(states, goals, contexts, histories, remaining_times) - targets).square().mean()
    mean_loss = torch.stack(losses).sum() / (epochs * n)
    return mean_loss.item(), before.item(), after.item()


def goal(env, frame, frame_limit):
    offsets = torch.arange(1, env.cfg.task.future_length+1, device=env.device)
    idx = torch.minimum(frame[:, None] + torch.floor(offsets*env.dt*env.ref.fps).long(), frame_limit[:, None])
    r = env.ref.sample(idx)
    return torch.cat((r['joints'], r['pos'], r['quat']), -1).flatten(1)


def soft_termination(env, state, frame):
    if not env.cfg.task.terminate_on_tracking_error:
        return state.new_zeros(state.shape[0])
    r = env.ref.sample(frame)
    height = (state[:,2]-r['pos'][:,2]).abs()
    tilt = (gravity_local(state[:,3:7])[:,2]-gravity_local(r['quat'])[:,2]).abs()
    # 解析平滑近似，仅用于想象中的可微存活权重。
    a = torch.sigmoid(40*(height-env.cfg.task.max_height_error))
    b = torch.sigmoid(40*(tilt-env.cfg.task.max_tilt_error))
    return 1-(1-a)*(1-b)


def termination_probability(env, model, head, state, frame, frame_limit):
    if head is None:
        return soft_termination(env, state, frame)
    return head(model.normalizer(state), goal(env, frame, frame_limit)).sigmoid()


def load_actor(actor, weights):
    weights = weights.copy()
    name = 'decoder.0.weight'
    wanted = actor.state_dict()[name]
    old = weights[name]
    expanded = old.shape != wanted.shape
    if expanded:
        extra = wanted.shape[1] - old.shape[1]
        if wanted.shape[0] != old.shape[0] or extra not in (2, 4):
            raise ValueError('Actor 结构不兼容')
        weights[name] = torch.cat((old, old.new_zeros(old.shape[0], extra)), dim=1)
    actor.load_state_dict(weights)
    return expanded


def policy_input(env, state, history, frame, frame_limit):
    parts = [history.flatten(1), goal(env,frame,frame_limit)]
    if env.cfg.task.heading_observation:
        parts.append(heading_error_features(state[:, 3:7], env.ref.sample(frame)['quat']))
    if env.cfg.task.longitudinal_observation:
        parts.append(longitudinal_features(env.ref, frame, frame_limit, state[:, :3], state[:, 19:22], env.dt))
    return torch.cat(parts, -1)


@torch.no_grad()
def evaluate_policy(env, actor, seed):
    """固定随机参考起点、无观测和动作噪声，按回合统计真实表现。"""
    with torch.random.fork_rng(devices=[env.device.index]):
        torch.manual_seed(seed)
        env.reset(torch.ones(env.cfg.num_envs, device=env.device, dtype=torch.bool))
        active=torch.ones(env.cfg.num_envs, device=env.device, dtype=torch.bool)
        returns=torch.zeros(env.cfg.num_envs, device=env.device)
        durations=torch.zeros_like(returns)
        complete=torch.zeros_like(active)
        yaw_sum=torch.zeros((), device=env.device)
        pos_sum=torch.zeros((), device=env.device)
        samples=torch.zeros((), device=env.device)
        for _ in range(math.ceil(env.cfg.episode_seconds / env.dt) + 1):
            action=actor(env.observation())
            _,reward,terminated,truncated,_=env.step(action)
            ref=env.ref.sample(env.frame)
            feat=heading_error_features(env.qpos[:,3:7],ref['quat'])
            yaw_abs=torch.atan2(feat[:,0],feat[:,1]).abs()
            position_error=(env.qpos[:,:3]-ref['pos']).norm(dim=-1)
            returns += torch.where(active,reward,0)
            durations += active.float()*env.dt
            yaw_sum += torch.where(active,yaw_abs,0).sum()
            pos_sum += torch.where(active,position_error,0).sum()
            samples += active.sum()
            complete |= active & truncated & ~terminated
            done=terminated|truncated
            active &= ~done
            if not active.any(): break
            env.reset(done)
        if active.any(): raise RuntimeError('评估未能在回合时间上限内结束')
        return {'eval_episode_return_mean': returns.mean().item(),
                'eval_completion_rate': complete.float().mean().item(),
                'eval_episode_duration_mean_s': durations.mean().item(),
                'eval_yaw_error_mean_deg': torch.rad2deg(yaw_sum/samples.clamp_min(1)).item(),
                'eval_root_position_error_mean_m': (pos_sum/samples.clamp_min(1)).item(),
                'eval_num_envs': env.cfg.num_envs, 'eval_seed': seed}


def save_checkpoint(run, payload):
    """保留每次保存的轮次，latest 原子更新；优先硬链接避免重复占用空间。"""
    numbered = run / f"model_{payload['cycle']}.pt"
    temp = run / '.checkpoint.tmp'
    latest_temp = run / '.latest.tmp'
    torch.save(payload, temp)
    os.replace(temp, numbered)
    try:
        os.link(numbered, latest_temp)
    except OSError:
        shutil.copyfile(numbered, latest_temp)
    os.replace(latest_temp, run / 'latest.pt')
    print(f'已保存 model_{payload["cycle"]}.pt 和 latest.pt', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('walking','backflip','sideflip'), default='walking')
    p.add_argument('--rsi-start-probability', type=float, default=0.2, help='单次动作 reset 从开头开始的概率')
    p.add_argument('--longitudinal-observation', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--adaptive-rsi', action='store_true', help='使用按参考片段回报自适应的 RSI')
    p.add_argument('--rsi-bin-seconds', type=float, default=0.2, help='RSI 分箱长度，秒')
    p.add_argument('--heading-observation', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--action-scale', type=float, default=1.0, help='动作到关节目标偏移的尺度，单位 rad')
    p.add_argument('--cycles', type=int, default=100)
    p.add_argument('--collect-steps', type=int, default=32)
    p.add_argument('--buffer-steps', type=int, default=320)
    p.add_argument('--wm-batch', type=int, default=128)
    p.add_argument('--actor-batch', type=int, default=128, help='每轮 Actor 想象的起点数')
    p.add_argument('--actor-rounds', type=int, default=1, help='每个 cycle 内滚动更新 Actor 的轮数；每轮重新采样 actor-batch 条起点，用更新后的 Actor 重新想象')
    p.add_argument('--actor-lr', type=float, default=5e-5, help='Actor 学习率；续训时覆盖 checkpoint 中的旧学习率')
    p.add_argument('--actor-critic-tail-weight', type=float, default=1.0,
                   help='Actor 想象目标中 Critic 尾项的系数；1 对应原始目标，0 关闭尾项，不改变 Critic 训练目标')
    p.add_argument('--critic-epochs', type=int, default=5, help='每批固定想象 λ-return 目标的拟合遍数；论文为 5')
    p.add_argument('--critic-minibatches', type=int, default=4, help='每个 Critic epoch 的随机 minibatch 数；论文为 4')
    p.add_argument('--critic-input-warmup-cycles', type=int, default=20,
                   help='新 Critic 或旧输入迁移后，先冻结 Actor 校准 Critic 的 cycle 数')
    p.add_argument('--wm-updates', type=int, default=None, help='手动指定每 cycle 的 WM 更新次数；省略时按新增数据量计算')
    p.add_argument('--wm-data-ratio', type=float, default=3.75, help='WM 名义训练窗口步数/新增 transition 数；3.75 对应原 256 环境、32 步、128 batch、8 步窗口、30 次更新')
    p.add_argument('--warmup-cycles', type=int, default=20, help='先采集和拟合动力学，暂不更新 Actor/Critic')
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--eval-interval', type=int, default=50, help='每隔多少 cycle 固定条件评估；0 关闭')
    p.add_argument('--eval-envs', type=int, default=16)
    p.add_argument('--eval-seed', type=int, default=123)
    p.add_argument('--save-interval', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--motion', type=Path, default=Path('/workspace/go2_mujoco_mimic/data/kine2go/ai4_dog_canter/motion.npy'))
    p.add_argument('--run-name', default='state_world_model')
    p.add_argument('--resume', type=Path)
    p.add_argument('--use-current-reward', action='store_true', help='续训时采用当前奖励配置，并重新初始化 Critic 和 RSI 统计')
    p.add_argument('--backflip-joint-mapping', action='store_true', help='后空翻按关节设置动作尺度并限制机械范围；续训时重新训练世界模型')
    p.add_argument('--wm-foot-features', action='store_true', help='世界模型输入脚掌离地距离和竖直速度')
    p.add_argument('--wm-phase-priority', action='store_true', help='世界模型窗口一半均匀采样，一半按分段预测误差加权')
    p.add_argument('--wm-only', action='store_true', help='冻结 Actor 和 Critic，只采集并训练世界模型')
    p.add_argument('--learned-termination', action=argparse.BooleanOptionalAction, default=None,
                   help='用预测状态上的 BCE 学习失败头；省略时继承 checkpoint，旧 checkpoint 使用解析终止')
    p.add_argument('--profile', action='store_true', help='每个 cycle 同步计时各阶段并打印；有测量开销，仅供诊断')
    p.add_argument('--nonfoot-contact-fall', action='store_true', help='任一非脚掌几何体接触地面即判失败（fallen）；续训时可从关闭切换为开启')
    p.add_argument('--actor-checkpoint', type=Path, help='只初始化旧版训练的 Actor；不加载旧世界模型')
    args = p.parse_args()
    if not math.isfinite(args.wm_data_ratio) or args.wm_data_ratio <= 0:
        p.error('wm-data-ratio 必须是有限正数')
    if args.wm_updates is not None and args.wm_updates <= 0:
        p.error('wm-updates 必须为正')
    if not math.isfinite(args.rsi_bin_seconds) or args.rsi_bin_seconds <= 0: p.error('rsi-bin-seconds 必须是有限正数')
    if not 0 <= args.rsi_start_probability <= 1: p.error('rsi-start-probability 必须在 0 到 1 之间')
    if args.warmup_cycles < 0 or args.eval_interval < 0 or args.eval_envs <= 0: p.error('warmup-cycles / eval-interval 不可为负，eval-envs 必须为正')
    if not math.isfinite(args.actor_lr) or args.actor_lr <= 0:
        p.error('actor-lr 必须是有限正数')
    if not math.isfinite(args.actor_critic_tail_weight) or args.actor_critic_tail_weight < 0:
        p.error('actor-critic-tail-weight 必须是有限的非负数')
    if not math.isfinite(args.action_scale) or args.action_scale <= 0:
        p.error('action-scale 必须是有限正数')
    if min(args.num_envs,args.cycles,args.collect_steps,args.wm_batch,args.actor_batch,args.actor_rounds,args.critic_epochs,args.critic_minibatches,args.window,args.horizon,args.save_interval) <= 0 or args.buffer_steps < max(args.window,args.horizon):
        p.error('参数应为正数且缓冲容量不小于窗口与想象长度')
    if args.critic_minibatches > args.actor_batch * args.horizon:
        p.error('critic-minibatches 不可大于 actor-batch × horizon')
    if args.critic_input_warmup_cycles < 0:
        p.error('critic-input-warmup-cycles 不可为负')
    wm_updates_auto = args.wm_updates is None
    args.wm_updates = resolve_wm_updates(args.num_envs, args.collect_steps, args.wm_batch,
                                         args.window, args.wm_data_ratio, args.wm_updates)
    wm_new_transitions = args.num_envs * args.collect_steps
    wm_nominal_steps = args.wm_updates * args.wm_batch * args.window
    wm_nominal_ratio = wm_nominal_steps / wm_new_transitions
    print(f'[WM 训练量] 新增={wm_new_transitions} transitions/cycle；'
          f'更新={args.wm_updates} 次（{"自动" if wm_updates_auto else "手动"}）；'
          f'batch={args.wm_batch}，窗口={args.window}；名义训练/新增={wm_nominal_ratio:.3f}。'
          '窗口可重复采样，跨终止后的步不计 loss，实际有效比例另行记录。', flush=True)
    if not wm_updates_auto and wm_nominal_ratio < args.wm_data_ratio:
        print(f'[WM 提示] 手动 --wm-updates 覆盖自动计算；当前比例低于目标 {args.wm_data_ratio:g}。', flush=True)
    print(f'[Critic] 每批目标固定，{args.critic_epochs} epochs × {args.critic_minibatches} minibatches'
          f' = {args.critic_epochs * args.critic_minibatches} 次优化器更新/想象批次', flush=True)
    resume_state = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    saved_learned_termination = resume_state is not None and resume_state.get('termination_head') is not None
    if args.task == 'sideflip':
        if args.learned_termination is True or args.nonfoot_contact_fall:
            p.error('侧空翻采用解析高度/倾斜跟踪终止，请勿启用学习式终止或非脚掌接触终止')
        args.learned_termination = False
    elif args.learned_termination is None:
        args.learned_termination = saved_learned_termination
    termination_changed = resume_state is not None and args.learned_termination != saved_learned_termination
    torch.manual_seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = True
    prof={}
    def mark(label):
        if not args.profile: return
        torch.cuda.synchronize(); now=time.perf_counter()
        prof[label]=prof.get(label,0.)+now-mark.last; mark.last=now
    mark.last=time.perf_counter()
    def report():
        if not args.profile: return
        items={**prof, **{'env_'+k:v for k,v in env.profile.items()}}
        print('[profile] '+' '.join(f'{k}={v:.3f}s' for k,v in items.items()), flush=True)
        prof.clear(); env.profile.clear()
    cfg = EnvCfg(num_envs=args.num_envs, action_scale=args.action_scale)
    cfg.wm_foot_features = args.wm_foot_features
    if args.backflip_joint_mapping or args.task == 'sideflip':
        if args.task not in ('backflip', 'sideflip'): p.error('--backflip-joint-mapping 仅适用于空翻任务')
        cfg.action_joint_scales = (1.0, 1.5, 1.15, 1.0, 1.5, 1.15,
                                   1.0, 1.55, 1.0, 1.0, 1.55, 1.0)
        cfg.clamp_joint_targets = True
    cfg.task.random_reference_start = True
    cfg.task.terminate_on_nonfoot_contact = args.nonfoot_contact_fall
    if args.task in ('backflip', 'sideflip'):
        cfg.task.reward = make_sideflip_reward_cfg() if args.task == 'sideflip' else make_backflip_reward_cfg()
        cfg.task.terminate_on_tracking_error = True
        if args.task == 'sideflip':
            # 比论文的 0.25 m / 0.8 严格一档；误差始终相对于当前参考姿态。
            cfg.task.max_height_error = 0.20
            cfg.task.max_tilt_error = 0.6
        cfg.task.reference_loop = False
        cfg.task.reference_cycles = 1
        cfg.episode_seconds = 3.0 if args.task == 'sideflip' else 2.0
        # 竖直翻转时欧拉 yaw 有奇异性，当前不复用行走特征。
        cfg.task.heading_observation = False
        cfg.task.longitudinal_observation = False
        args.adaptive_rsi = True
        if str(args.motion) == '/workspace/go2_mujoco_mimic/data/kine2go/ai4_dog_canter/motion.npy':
            args.motion = (Path('/workspace/go2_mujoco_mimic/data/go2_flip_to/sideflip_left_30hz/motion.npy')
                           if args.task == 'sideflip' else Path('/workspace/go2_mujoco_mimic/data/solo12_backflip/reference_30hz/motion.npy'))
    else:
        cfg.task.heading_observation = args.heading_observation
        cfg.task.longitudinal_observation = args.longitudinal_observation
    # WM 对照试验沿用 checkpoint 的奖励，避免改变 RSI 和数据采集目标。
    if args.wm_only and args.resume:
        saved_cfg = resume_state['env']['task']['reward']
        cfg.task.reward = RewardCfg()
        for key, value in saved_reward_config(saved_cfg).items():
            setattr(cfg.task.reward, key, value)
    env = WarpGo2Env(cfg,args.motion); env.enable_cuda_graph()
    if args.profile: env.profile = {}
    actor = MotionPolicy(history_length=cfg.task.history_length, future_length=cfg.task.future_length,
                         heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).to(env.device)
    model = Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).to(env.device)
    critic = StateValue(history_length=cfg.task.history_length, goal_dim=cfg.task.future_length*19).to(env.device)
    opt_m = torch.optim.Adam(model.parameters(),lr=2e-4)
    termination_head = TerminationHead(cfg.task.future_length*19).to(env.device) if args.learned_termination else None
    opt_d = torch.optim.Adam(termination_head.parameters(),lr=2e-4) if termination_head is not None else None
    wm_parameters = list(model.parameters()) + (list(termination_head.parameters()) if termination_head is not None else [])
    opt_a = torch.optim.Adam(actor.parameters(),lr=args.actor_lr,betas=(.9,.95))
    opt_v = torch.optim.Adam(critic.parameters(),lr=2e-4)
    start=0
    rsi_saved=None
    critic_warmup_until=args.warmup_cycles+args.critic_input_warmup_cycles
    model_warmup_until=0
    critic_input_migrated=False
    if args.resume and args.actor_checkpoint: p.error('--resume 与 --actor-checkpoint 不能同时使用')
    if args.actor_checkpoint:
        old = torch.load(args.actor_checkpoint,map_location=env.device,weights_only=False)
        if old.get('motion') != str(args.motion): raise ValueError('Actor 检查点参考动作不一致')
        load_actor(actor, old.get('policy', old.get('actor')))
        print('已加载旧 Actor；连续状态世界模型与 critic 从头初始化。', flush=True)
    if args.resume:
        ck = resume_state
        if ck.get('format') != 'go2_core_normalized_v3': raise ValueError('需要 37 维 go2_core_normalized_v3 检查点；旧版未归一化世界模型不兼容')
        old_cfg = ck['env'].copy(); old_cfg['task'] = old_cfg['task'].copy()
        old_heading = old_cfg['task'].get('heading_observation', False)
        old_longitudinal = old_cfg['task'].get('longitudinal_observation', False)
        if (old_heading and not cfg.task.heading_observation) or (old_longitudinal and not cfg.task.longitudinal_observation):
            raise ValueError('续训不可移除已使用的观测维度')
        if old_longitudinal and not old_heading and cfg.task.heading_observation:
            raise ValueError('不能在已有前后观测之前插入航向观测')
        old_cfg['task'].setdefault('reference_loop', True)
        old_cfg['num_envs'] = cfg.num_envs  # 环境数不属于模型配置，Replay 不随检查点保存，续训时重新采集
        old_cfg['task'].setdefault('terminate_on_nonfoot_contact', False)
        old_cfg['task'].setdefault('terminate_on_tracking_error', True)
        tracking_fields = ('terminate_on_tracking_error', 'max_height_error', 'max_tilt_error')
        tracking_changed = any(old_cfg['task'][name] != getattr(cfg.task, name) for name in tracking_fields)
        if tracking_changed and args.task == 'sideflip':
            for name in tracking_fields:
                old_cfg['task'][name] = getattr(cfg.task, name)
        if args.nonfoot_contact_fall:
            old_cfg['task']['terminate_on_nonfoot_contact'] = True
        # 缺失字段使用历史默认值；奖励形式也参与比较，且不修改 checkpoint 原字典。
        old_cfg['task']['reward'] = saved_reward_config(old_cfg['task']['reward'])
        old_cfg.setdefault('action_joint_scales', None)
        old_cfg.setdefault('clamp_joint_targets', False)
        old_cfg.setdefault('wm_foot_features', False)
        old_cfg['task']['heading_observation'] = cfg.task.heading_observation
        old_cfg['task']['longitudinal_observation'] = cfg.task.longitudinal_observation
        new_cfg = asdict(cfg)
        wm_changed = old_cfg['wm_foot_features'] != new_cfg['wm_foot_features']
        if wm_changed and args.wm_foot_features:
            old_cfg['wm_foot_features'] = new_cfg['wm_foot_features']
        mapping_changed = (old_cfg['action_joint_scales'], old_cfg['clamp_joint_targets']) != (new_cfg['action_joint_scales'], new_cfg['clamp_joint_targets'])
        if mapping_changed and args.backflip_joint_mapping:
            old_cfg['action_joint_scales'] = new_cfg['action_joint_scales']
            old_cfg['clamp_joint_targets'] = new_cfg['clamp_joint_targets']
        reward_changed = old_cfg['task']['reward'] != new_cfg['task']['reward']
        if args.use_current_reward:
            old_cfg['task']['reward'] = new_cfg['task']['reward']
        if old_cfg != new_cfg or ck['motion'] != str(args.motion): raise ValueError('续训配置或动作不同；奖励变化时可使用 --use-current-reward')
        expanded = load_actor(actor, ck['actor'])
        if not (mapping_changed or wm_changed):
            model.load_state_dict(ck['model'])
        if not (mapping_changed or wm_changed or termination_changed or tracking_changed or (args.use_current_reward and reward_changed)):
            if ck.get('critic_config') is not None and ck['critic_config'] != critic.config:
                raise ValueError('Critic 历史/目标配置与 checkpoint 不一致')
            critic_input_migrated=critic.load_for_resume(ck['critic'])
        if termination_head is not None and saved_learned_termination and not (mapping_changed or wm_changed):
            termination_head.load_state_dict(ck['termination_head'])
            if ck.get('opt_d') is not None:
                opt_d.load_state_dict(ck['opt_d'])
        if not expanded: opt_a.load_state_dict(ck['opt_a'])
        else: print('已保留旧 Actor 权重并为新增观测增加零权重；Actor 优化器重新初始化。', flush=True)
        rsi_saved=None if mapping_changed or tracking_changed or (args.use_current_reward and reward_changed) else ck.get('adaptive_rsi')
        for group in opt_a.param_groups: group['lr'] = args.actor_lr
        if not (mapping_changed or wm_changed):
            opt_m.load_state_dict(ck['opt_m'])
        if not (mapping_changed or wm_changed or termination_changed or tracking_changed or critic_input_migrated or (args.use_current_reward and reward_changed)):
            opt_v.load_state_dict(ck['opt_v'])
        elif not critic_input_migrated:
            print('映射、奖励或终止方式已变更：重新初始化 Critic；RSI 在奖励、动作映射或跟踪终止条件改变时重置。', flush=True)
            if bool(model.normalizer.ready):
                critic.normalizer.load_state_dict(model.normalizer.state_dict())
        start=ck['cycle']
        critic_warmup_until = ck.get('critic_warmup_until', 0)
        model_warmup_until = ck.get('model_warmup_until', 0)
        if mapping_changed or wm_changed:
            model_warmup_until = start + 20
            critic_warmup_until = model_warmup_until + 20
            print('动作映射或世界模型输入已改变：世界模型从头训练，先预热 20 cycle，再预热 Critic 20 cycle。', flush=True)
        elif args.use_current_reward and reward_changed:
            critic_warmup_until = start + 20
            print('先更新 Critic 20 个 cycle，再更新 Actor。', flush=True)
        if termination_changed:
            if args.learned_termination:
                model_warmup_until = max(model_warmup_until, start + 20)
                critic_warmup_until = max(critic_warmup_until, model_warmup_until + 20)
                print('新增 BCE 终止头：保留 Actor/动力学，先联合预热 WM+终止头 20 cycle，再预热 Critic 20 cycle。',flush=True)
            else:
                critic_warmup_until = max(critic_warmup_until, start + 20)
        if tracking_changed:
            critic_warmup_until = max(critic_warmup_until, start + 20)
            tracking_status = '开启' if cfg.task.terminate_on_tracking_error else '关闭'
            print(f'跟踪失败配置已变更：{tracking_status}，高度阈值 {cfg.task.max_height_error:g} m，'
                  f'倾斜误差阈值 {cfg.task.max_tilt_error:g}；保留 Actor 和动力学，'
                  '重置 RSI/Critic，先校准 Critic 20 cycle。', flush=True)
        if critic_input_migrated:
            critic_warmup_until = max(critic_warmup_until,
                                      max(start, model_warmup_until)+args.critic_input_warmup_cycles)
            print(f'[Critic 迁移] 输入 {critic.base_input_dim} → {critic.input_dim}：保留旧权重，'
                  '历史/时间列置零，重建 Critic 优化器；'
                  f'Actor 冻结至 cycle {critic_warmup_until}，之后恢复更新。', flush=True)
    run=Path('/workspace/go2_mujoco_mimic/logs')/f'{args.run_name}_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}'
    run.mkdir(parents=True)
    def save_current_checkpoint(payload):
        termination_mode = ('learned_bce' if termination_head is not None else
                            'analytic_sigmoid' if cfg.task.terminate_on_tracking_error else 'disabled')
        payload.update(termination_mode=termination_mode,
                       termination_head=termination_head.state_dict() if termination_head is not None else None,
                       opt_d=opt_d.state_dict() if opt_d is not None else None,
                       critic_config=critic.config,
                       critic_time_basis='remaining_reference_frames/full_reference_span_v1')
        save_checkpoint(run,payload)
    (run/'config.json').write_text(json.dumps({'args':vars(args),'env':asdict(cfg),'state_size':LAYOUT.size,
        'critic_config':critic.config,'critic_input_migrated':critic_input_migrated,
        'critic_warmup_until':critic_warmup_until,
        'critic_time_basis':'remaining_reference_frames/full_reference_span_v1',
        'context':'固定：地面摩擦 0.5、标定偏差/质心偏移为零'},default=str,ensure_ascii=False,indent=2))
    # 摩擦、12 个关节偏移、机身质心 xyz；当前 MuJoCo 未做 domain randomization。
    context=torch.zeros((args.num_envs,16),device=env.device);context[:,0]=0.5
    replay=Replay(args.buffer_steps,args.num_envs,env.device)
    wm_bins = math.ceil((len(env.ref.frames)-1) / max(1, round(args.rsi_bin_seconds * env.ref.fps)))
    wm_bin_frames = max(1, round(args.rsi_bin_seconds * env.ref.fps))
    wm_error_ema = torch.zeros(wm_bins, device=env.device)
    wm_error_seen = torch.zeros(wm_bins, dtype=torch.bool, device=env.device)
    wm_sample_counts = torch.zeros(wm_bins, device=env.device)
    if args.resume and args.wm_phase_priority and ck.get('wm_priority') is not None:
        saved_priority = ck['wm_priority']
        if saved_priority['bin_frames'] != wm_bin_frames or len(saved_priority['error_ema']) != wm_bins:
            raise ValueError('WM 分段采样配置与 checkpoint 不兼容')
        wm_error_ema.copy_(saved_priority['error_ema'])
        wm_error_seen.copy_(saved_priority['seen'])
        wm_sample_counts.copy_(saved_priority['counts'])
    reward_cfg = cfg.task.reward
    max_reward_rate = sum(getattr(reward_cfg, name) for name in (
        'body_position_weight', 'body_orientation_weight', 'body_linear_velocity_weight',
        'body_angular_velocity_weight', 'root_position_weight', 'root_orientation_weight'
    )) + reward_cfg.joint_position_max_reward
    rsi=None
    if args.adaptive_rsi:
        rsi=AdaptiveRSI(len(env.ref.frames)-1,env.ref.fps,env.device,bin_seconds=args.rsi_bin_seconds,
                        max_reward_rate=max_reward_rate, max_episode_steps=min(math.ceil(cfg.episode_seconds/env.dt),
                                              math.ceil((len(env.ref.frames)-1)*cfg.task.reference_cycles/env.ref.fps/env.dt)),
                        control_dt=env.dt)
        if not cfg.task.reference_loop:
            rsi.gmax = 1.0  # 非循环动作按各自剩余时长的最大可得回报归一化。
        if rsi_saved is not None:
            rsi.load_state_dict(rsi_saved)
            print('已恢复 RSI 回报与历史分布。',flush=True)
        env.reset(torch.ones(args.num_envs,device=env.device,dtype=torch.bool),rsi.probabilities(),rsi.frames_per_bin, start_probability=args.rsi_start_probability if not cfg.task.reference_loop else 0.)
        print(f'自适应 RSI：每箱约 {rsi.frames_per_bin/env.ref.fps:.3f}s，共 {rsi.n_bins} 箱；分箱按 50%均匀 + 40%近期困难 + 10%历史困难混合。',flush=True)
        if not cfg.task.reference_loop:
            print(f'单次动作 {args.rsi_start_probability:.0%} 的重置固定从开头，剩余重置采用上述分箱。',flush=True)
    state=build_state(env)
    normalization_ready = bool(model.normalizer.ready)
    episode_return = torch.zeros(args.num_envs, device=env.device)
    discounted_return = torch.zeros(args.num_envs, device=env.device)
    discount = torch.ones(args.num_envs, device=env.device)
    eval_env = None
    print(f'连续状态={LAYOUT.size}维；环境={args.num_envs}；日志={run}；身体奖励由可微运动学计算；世界模型输入{73 if cfg.wm_foot_features else 65}维、输出37维',flush=True)
    print(f'Actor 学习率={args.actor_lr:g}；Actor Critic 尾项系数={args.actor_critic_tail_weight:g}', flush=True)
    print(f'[Critic] 输入 {critic.input_dim} 维，含 {cfg.task.history_length} 帧策略历史及参考终点倒计时；'
          f'Actor 从 cycle {max(start,critic_warmup_until)+1} 起更新。', flush=True)
    print(f'缓冲装满后每轮新数据比例={min(1,args.collect_steps/args.buffer_steps):.1%}；Actor想象={args.actor_batch}条×{args.horizon}步，每 cycle 滚动更新 {args.actor_rounds} 次',flush=True)
    for cycle in range(start,start+args.cycles):
        if args.profile:
            torch.cuda.synchronize(); mark.last=time.perf_counter(); prof.clear(); env.profile.clear()
        rewards=[]; fails=torch.zeros((),device=env.device,dtype=torch.long)
        completed_count=torch.zeros((),device=env.device)
        completed_return=torch.zeros((),device=env.device)
        completed_time=torch.zeros((),device=env.device)
        for _ in range(args.collect_steps):
            with torch.no_grad():
                obs=env.observation(); frame=env.frame.clone(); history=env.history.clone(); previous=env.previous_action.clone()
                action=(actor(obs)+torch.randn_like(previous)*.1).clamp(-1,1)
                _,reward,terminated,truncated,_=env.step(action)
                next_state=build_state(env)
                done=terminated|truncated
                # done 用于跨 reset 的窗口掩码；BCE 仅监督真实失败，正常结束另行截断想象。
                replay.add((state,history,frame,previous,action,next_state,done.float(),context,
                            env.frame_limit.clone(),terminated.float(),env.frame.clone()))
                rewards.append(reward.mean());fails+=terminated.sum()
                episode_return += reward
                if rsi is not None:
                    discounted_return += discount*reward
                    discount *= .99
                    score=discounted_return
                    if not cfg.task.reference_loop:
                        remaining_steps=torch.ceil((env.frame_limit-env.start_frame).float()/env.ref.fps/env.dt).clamp_min(1)
                        best_possible=(max_reward_rate * env.dt)*(1.-torch.pow(.99,remaining_steps))/.01
                        score=discounted_return/best_possible.clamp_min(1e-6)
                    rsi.update(env.start_frame,score,done)
                    discounted_return = torch.where(done,0.,discounted_return)
                    discount = torch.where(done,1.,discount)
                completed_count += done.sum()
                completed_return += torch.where(done, episode_return, 0).sum()
                completed_time += torch.where(done, env.steps * env.dt, 0).sum()
                episode_return = torch.where(done, 0, episode_return)
                if rsi is None:
                    env.reset(done)
                else:
                    env.reset(done,rsi.probabilities(),rsi.frames_per_bin, start_probability=args.rsi_start_probability if not cfg.task.reference_loop else 0.)
                initial=build_state(env)
                state=torch.where(done[:,None],initial,next_state)
        mark('collect')
        if replay.count < max(args.window,args.horizon): continue
        if not normalization_ready:
            count = min(replay.count, replay.capacity)
            calibration = torch.cat((replay.storage[0][:count].flatten(0, 1),
                                     replay.storage[5][:count].flatten(0, 1)), 0)
            model.normalizer.calibrate(calibration)
            critic.normalizer.load_state_dict(model.normalizer.state_dict())
            normalization_ready = True
            (run/'normalization.json').write_text(json.dumps({
                'method':'first_collection_fixed_mean_std', 'samples':len(calibration),
                'mean':model.normalizer.mean.tolist(), 'scale':model.normalizer.scale.tolist()},indent=2))
            print('状态归一化已校准并冻结；state_loss 为归一化误差，每轮取全部 WM 更新的平均。',flush=True)
        wm_losses=[];wm_bce_losses=[]
        wm_valid_steps=torch.zeros((),device=env.device)
        term_counts=torch.zeros(5,device=env.device)  # 有效样本、正标签、TP、FP、Brier 和
        for _ in range(args.wm_updates):
            if args.wm_phase_priority:
                candidates = replay.sample(args.wm_batch * 4, args.window)
                candidate_bins = (candidates[2][:, 0].remainder(len(env.ref.frames)-1) // wm_bin_frames).clamp(max=wm_bins-1)
                fallback = (wm_error_ema * wm_error_seen).sum() / wm_error_seen.sum().clamp_min(1)
                estimates = torch.where(wm_error_seen, wm_error_ema, fallback).clamp_min(1e-6)
                # 截断比例避免少数异常窗口独占优先部分；采样不修改 RSI。
                weights = (estimates / estimates.mean().clamp_min(1e-6)).clamp(.25, 4.)
                n_uniform = args.wm_batch // 2
                uniform = torch.arange(n_uniform, device=env.device)
                weighted = torch.multinomial(weights[candidate_bins], args.wm_batch - n_uniform, replacement=True)
                chosen = torch.cat((uniform, weighted))
                states,hist,frames,previous,actions,targets,dones,contexts,limits,failure_labels,next_frames = (x[chosen] for x in candidates)
                sample_bins = candidate_bins[chosen]
                wm_sample_counts += torch.bincount(sample_bins, minlength=wm_bins)
                per_sample_loss = torch.zeros(args.wm_batch, device=env.device)
                per_sample_steps = torch.zeros(args.wm_batch, device=env.device)
            else:
                states,hist,frames,previous,actions,targets,dones,contexts,limits,failure_labels,next_frames=replay.sample(args.wm_batch,args.window)
            pred=states[:,0];live=torch.ones(args.wm_batch,device=env.device);loss=torch.zeros((),device=env.device)
            bce_loss=torch.zeros((),device=env.device)
            for t in range(args.window):
                with torch.autocast('cuda',dtype=torch.bfloat16): pred=model(pred,actions[:,t],contexts[:,t])
                target_state=targets[:,t]
                # 四元数 q 与 -q 对应同一姿态；取与预测同侧的符号再计算误差。
                target_quat=target_state[:,3:7]
                target_quat=torch.where((pred[:,3:7]*target_quat).sum(-1,keepdim=True)<0,
                                        -target_quat,target_quat)
                target_state=torch.cat((target_state[:,:3],target_quat,target_state[:,7:]),-1)
                step=((pred-target_state)/model.normalizer.scale).square().mean(-1)
                wm_valid_steps += live.sum()
                if args.wm_phase_priority:
                    per_sample_loss += step.detach() * live
                    per_sample_steps += live
                loss+= (step*live).sum()/live.sum().clamp_min(1)
                if termination_head is not None:
                    # 不 detach 预测状态：BCE 与状态损失共同优化 WM，符合论文 Eq.1。
                    logits=termination_head(model.normalizer(pred),goal(env,next_frames[:,t],limits[:,t]))
                    labels=failure_labels[:,t]
                    bce=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
                    bce_loss+=(bce*live).sum()/live.sum().clamp_min(1)
                    with torch.no_grad():
                        probability=logits.sigmoid();positive=probability>=.5
                        term_counts+=torch.stack((live.sum(),(live*labels).sum(),
                            (live*labels*positive).sum(),(live*(1-labels)*positive).sum(),
                            (live*(probability-labels).square()).sum()))
                live=live*(1-dones[:,t])
            loss/=args.window
            bce_loss/=args.window
            opt_m.zero_grad(set_to_none=True)
            if opt_d is not None:opt_d.zero_grad(set_to_none=True)
            (loss+bce_loss).backward()
            torch.nn.utils.clip_grad_norm_(wm_parameters,1.,error_if_nonfinite=True);opt_m.step()
            if opt_d is not None:opt_d.step()
            wm_losses.append(loss.detach())
            wm_bce_losses.append(bce_loss.detach())
            if args.wm_phase_priority:
                sample_mean = per_sample_loss / per_sample_steps.clamp_min(1)
                totals = torch.zeros_like(wm_error_ema).scatter_add_(0, sample_bins, sample_mean)
                counts = torch.bincount(sample_bins, minlength=wm_bins)
                present = counts > 0
                batch_mean = totals / counts.clamp_min(1)
                wm_error_ema = torch.where(present, torch.where(wm_error_seen, .9 * wm_error_ema + .1 * batch_mean, batch_mean), wm_error_ema)
                wm_error_seen |= present
        mean_wm_loss=torch.stack(wm_losses).mean().item()
        mean_bce_loss=torch.stack(wm_bce_losses).mean().item()
        mark('world_model')
        count, return_sum, time_sum = torch.stack((completed_count, completed_return, completed_time)).tolist()
        count = int(count)
        valid_train_steps = wm_valid_steps.item()
        episode_stats = {'completed_episodes': count,
                         'episode_return_mean': return_sum / count if count else None,
                         'episode_duration_mean_s': time_sum / count if count else None,
                         'wm_updates': args.wm_updates,
                         'wm_new_transitions': wm_new_transitions,
                         'wm_nominal_train_steps': wm_nominal_steps,
                         'wm_valid_train_steps': valid_train_steps,
                         'wm_effective_data_ratio': valid_train_steps / wm_new_transitions}
        if termination_head is not None:
            n,pos,tp,fp,brier=term_counts.tolist()
            episode_stats.update(termination_bce=mean_bce_loss,termination_positive_samples=int(pos),
                termination_recall=tp/pos if pos else None,
                termination_false_positive_rate=fp/(n-pos) if n>pos else None,
                termination_brier=brier/max(n,1),wm_total_loss=mean_wm_loss+mean_bce_loss)
            recall=f'{tp/pos:.1%}' if pos else '—（本轮无失败样本）'
            print(f'[终止 BCE {cycle+1}] loss={mean_bce_loss:.5f} 正样本={int(pos)}/{int(n)} '
                  f'召回={recall} Brier={brier/max(n,1):.5f}',flush=True)
        if (cycle+1) % 10 == 0:
            print(f'[WM 训练量 {cycle+1}] 更新={args.wm_updates}；'
                  f'有效训练/新增={episode_stats["wm_effective_data_ratio"]:.3f}；'
                  f'名义比例={wm_nominal_ratio:.3f}', flush=True)
        if rsi is not None:
            episode_stats.update({'rsi_probabilities':rsi.probabilities().tolist(),
                                  'rsi_bin_completed_counts':rsi.counts.tolist()})
        if args.wm_phase_priority:
            episode_stats.update({'wm_phase_error_ema':wm_error_ema.tolist(), 'wm_phase_sample_counts':wm_sample_counts.tolist()})
            if (cycle + 1) % 10 == 0:
                print(f'[WM 分段 {cycle+1}] 误差={wm_error_ema.tolist()} 累计窗口数={wm_sample_counts.tolist()}', flush=True)
        episode_text = (f"\033[1m回合收益={episode_stats['episode_return_mean']:.3f}\033[0m 存活={episode_stats['episode_duration_mean_s']:.2f}s 回合数={count}"
                        if count else '回合收益=— 存活=— 回合数=0')
        if args.wm_only or cycle < args.warmup_cycles or cycle < model_warmup_until:
            m={'cycle':cycle+1,'real_reward':torch.stack(rewards).mean().item(),'failures':fails.item(),'state_loss':mean_wm_loss,'phase':'world_model_only' if args.wm_only else 'world_model_warmup',**episode_stats}
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
            print(f"[{cycle+1}] 预训练 reward={m['real_reward']:.4f} fallen={m['failures']} {episode_text} state_loss={m['state_loss']:.5f}",flush=True)
            if (cycle+1)%args.save_interval==0 or cycle+1==start+args.cycles:
                save_current_checkpoint({'format':'go2_core_normalized_v3','cycle':cycle+1,'critic_warmup_until':critic_warmup_until,'model_warmup_until':model_warmup_until,'actor':actor.state_dict(),'model':model.state_dict(),'critic':critic.state_dict(),'opt_a':opt_a.state_dict(),'opt_m':opt_m.state_dict(),'opt_v':opt_v.state_dict(),'env':asdict(cfg),'motion':str(args.motion),'adaptive_rsi':rsi.state_dict() if rsi is not None else None,'wm_priority':{'bin_frames':wm_bin_frames,'error_ema':wm_error_ema,'seen':wm_error_seen,'counts':wm_sample_counts} if args.wm_phase_priority else None})
            continue
        round_stats=[]
        for _round in range(args.actor_rounds):
            states,hist,frames,previous,actions,targets,dones,contexts,limits,failure_labels,next_frames=replay.sample(args.actor_batch,args.horizon)
            s=states[:,0].detach(); history=hist[:,0].detach(); frame=frames[:,0].clone(); prev=previous[:,0].detach();ctx=contexts[:,0];frame_limit=limits[:,0]
            survive=torch.ones(args.actor_batch,device=env.device);ret=torch.zeros_like(survive)
            ss=[];gg=[];hh=[];tt=[];rr=[];dd=[];critic_valid=[]
            pose=None  # 上一步 next_s 的刚体位姿，下一步复用；第 0 步现算
            for param in model.parameters():param.requires_grad_(False)
            if termination_head is not None:
                for param in termination_head.parameters():param.requires_grad_(False)
            for param in critic.parameters():param.requires_grad_(False)
            for t in range(args.horizon):
                # 当前状态仍在参考动作内时有效；进入终点的那一步仍应参与 Critic 拟合。
                critic_valid.append(reference_critic_mask(frame, frame_limit))
                g=goal(env,frame,frame_limit);ss.append(s);gg.append(g)
                hh.append(history);tt.append(critic_remaining_time(env,frame,frame_limit))
                actor_input=policy_input(env,s,history,frame,frame_limit)
                action=actor(actor_input)
                if t == 0:
                    diagnostic_input = actor_input.detach()
                    diagnostic_action = action.detach()
                with torch.autocast('cuda',dtype=torch.bfloat16): next_s=model(s,action,ctx)
                advance=math.floor((t+1)*env.dt*env.ref.fps)-math.floor(t*env.dt*env.ref.fps)
                nf=torch.minimum(frame+int(advance),frame_limit)
                target=env.ref.sample(nf); previous_target=env.ref.sample(frame)
                reference={'pos':target['body_pos'],'quat':target['body_quat'],
                           'joint_pos':target['joints'],
                           'linear_vel':(target['body_pos']-previous_target['body_pos'])/env.dt,
                           'angular_vel':rotation_vector(target['body_quat'],previous_target['body_quat'])/env.dt}
                actual,pose=env.kinematics.transition_with_pose(pose,s,next_s,env.dt)
                target_q=env.default_q+env.action_scales*action
                if env.cfg.clamp_joint_targets: target_q=target_q.clamp(env.joint_limits[:,0],env.joint_limits[:,1])
                torque_sq = None
                if cfg.task.reward.torque_l2_weight and cfg.task.reward.torque_mode == 'substep_mean':
                    torque_sq = estimate_torque_squared(s[:,7:19],s[:,25:37],next_s[:,7:19],next_s[:,25:37],
                                                       target_q,cfg.kp,cfg.kd,cfg.torque_limit,env.substeps)
                reward,_=env.task.reward(actual,reference,action,prev,next_s[:,7:19],env.joint_limits,env.dt,
                                        joint_vel=next_s[:,25:37],prev_joint_vel=s[:,25:37],target=target_q,
                                        torque_squared_mean=torque_sq,kp=cfg.kp,kd=cfg.kd,torque_limit=cfg.torque_limit)
                failure=termination_probability(env,model,termination_head,next_s,nf,frame_limit)
                # 使用软失败概率的期望一次性惩罚，参考结束本身不扣分。
                reward = reward - env.cfg.task.reward.failure_penalty * failure
                end_of_reference = (nf >= frame_limit).float()
                failure = 1-(1-failure)*(1-end_of_reference)
                rr.append(reward);dd.append(failure)
                ret+=.99**t*survive*reward;survive=survive*(1-failure)
                new_obs=obs_from_state(next_s,action,env.default_q,noise=True)
                history=torch.cat((history[:,1:],new_obs[:,None]),1)
                s=next_s;frame=nf;prev=action
            last_goal=goal(env,frame,frame_limit)
            last_time=critic_remaining_time(env,frame,frame_limit)
            bootstrap=(.99**args.horizon)*survive*critic(s,last_goal,ctx,history,last_time)
            # 仅调整 Actor 使用尾部价值的强度；Critic 的 λ-return 仍学习完整回报。
            objective=ret+args.actor_critic_tail_weight*bootstrap
            mark('imagine_fwd')
            if cycle >= critic_warmup_until:
                opt_a.zero_grad(set_to_none=True);(-objective.mean()).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
                for param in actor.parameters():
                    if param.grad is not None:param.grad.clamp_(-.1,.1)
                opt_a.step()
            mark('actor_update')
            with torch.no_grad():
                action_delta=(actor(diagnostic_input)-diagnostic_action).abs()
                actor_update_mean,actor_update_max=torch.stack((action_delta.mean(),action_delta.max())).tolist()
            for param in model.parameters():param.requires_grad_(True)
            if termination_head is not None:
                for param in termination_head.parameters():param.requires_grad_(True)
            for param in critic.parameters():param.requires_grad_(True)
            critic_states=torch.stack(ss,1).detach().flatten(0,1)
            critic_goals=torch.stack(gg,1).detach().flatten(0,1)
            critic_contexts=ctx[:,None,:].expand(-1,args.horizon,-1).reshape(-1,ctx.shape[-1]).detach()
            critic_histories=torch.stack(hh,1).detach().flatten(0,1)
            critic_times=torch.stack(tt,1).detach().flatten(0,1)
            with torch.no_grad():
                target_v=critic(s.detach(),last_goal,ctx,history.detach(),last_time)
                old_values=critic(critic_states,critic_goals,critic_contexts,critic_histories,critic_times).reshape(args.actor_batch,args.horizon)
                next_values=torch.cat((old_values[:,1:],target_v[:,None]),1)
                returns=[]
                for t in reversed(range(args.horizon)):
                    vnext=next_values[:,t]
                    target_v=rr[t].detach()+.99*(1-dd[t].detach())*((1-.95)*vnext+.95*target_v)
                    returns.append(target_v)
                returns=torch.stack(list(reversed(returns)),1).flatten()
            # Actor 的 survive 已经屏蔽了参考终点后的回报；Critic 同样不能拟合
            # 真实回合结束后继续想象出来的状态。软失败概率仍按原 λ-return 处理。
            valid_indices=torch.stack(critic_valid,1).flatten().nonzero(as_tuple=True)[0]
            critic_states=critic_states.index_select(0,valid_indices)
            critic_goals=critic_goals.index_select(0,valid_indices)
            critic_contexts=critic_contexts.index_select(0,valid_indices)
            critic_histories=critic_histories.index_select(0,valid_indices)
            critic_times=critic_times.index_select(0,valid_indices)
            returns=returns.index_select(0,valid_indices)
            if valid_indices.numel() == 0:
                raise RuntimeError('Critic 没有参考动作结束前的有效状态，请检查 replay 起点。')
            critic_minibatches=min(args.critic_minibatches,valid_indices.numel())
            value_loss,critic_loss_before,critic_loss_after=fit_critic(
                critic,opt_v,critic_states,critic_goals,critic_contexts,returns,
                args.critic_epochs,critic_minibatches,histories=critic_histories,remaining_times=critic_times)
            mark('critic')
            round_stats.append((objective.mean().item(),ret.mean().item(),bootstrap.mean().item(),value_loss,actor_update_mean,actor_update_max,critic_loss_before,critic_loss_after,critic_minibatches))
        report()
        n_rounds=len(round_stats)
        imagined_objective,imagined_ret,imagined_boot,value_loss_mean=(sum(x[i] for x in round_stats)/n_rounds for i in range(4))
        actor_update_mean=sum(x[4] for x in round_stats)/n_rounds;actor_update_max=max(x[5] for x in round_stats)
        m={'cycle':cycle+1,'real_reward':torch.stack(rewards).mean().item(),'failures':fails.item(),'state_loss':mean_wm_loss,'imagined_return':imagined_objective,'imagined_reward_sum':imagined_ret,'critic_bootstrap':imagined_boot,'actor_critic_tail_weight':args.actor_critic_tail_weight,'actor_weighted_critic_bootstrap':args.actor_critic_tail_weight*imagined_boot,'value_loss':value_loss_mean,'actor_update_mean_abs':actor_update_mean,'actor_update_max_abs':actor_update_max,**episode_stats}
        m.update(critic_loss_before=sum(x[6] for x in round_stats)/n_rounds,
                 critic_loss_after=sum(x[7] for x in round_stats)/n_rounds,
                 critic_optimizer_steps=sum(x[8] for x in round_stats)*args.critic_epochs,
                 actor_updated=cycle >= critic_warmup_until,
                 critic_warmup_remaining_cycles=max(0,critic_warmup_until-(cycle+1)))
        print(f'[Critic {cycle+1}] 固定目标 MSE：{m["critic_loss_before"]:.5f} → {m["critic_loss_after"]:.5f}；'
              f'更新={m["critic_optimizer_steps"]} 次；Actor={"更新" if m["actor_updated"] else "冻结校准"}',flush=True)
        if args.eval_interval and (cycle+1) % args.eval_interval == 0:
            if eval_env is None:
                from copy import deepcopy
                eval_cfg = deepcopy(cfg)
                eval_cfg.num_envs = args.eval_envs
                eval_cfg.task.observation_noise = False
                if not eval_cfg.task.reference_loop:
                    eval_cfg.task.random_reference_start = False
                with torch.random.fork_rng(devices=[env.device.index]):
                    torch.manual_seed(args.eval_seed)
                    eval_env = WarpGo2Env(eval_cfg,args.motion)
                    eval_env.enable_cuda_graph()
            actor.eval()
            evaluation=evaluate_policy(eval_env,actor,args.eval_seed)
            actor.train()
            m.update(evaluation)
            if rsi is not None:
                print(f"[RSI {cycle+1}] 各箱概率={rsi.probabilities().tolist()} 已完成回合数={rsi.counts.tolist()}",flush=True)
            print(f"[评估 {cycle+1}] \033[1m回合收益={evaluation['eval_episode_return_mean']:.3f}\033[0m 完整率={evaluation['eval_completion_rate']:.1%} 存活={evaluation['eval_episode_duration_mean_s']:.2f}s yaw误差={evaluation['eval_yaw_error_mean_deg']:.1f}° 机身位置误差={evaluation['eval_root_position_error_mean_m']:.3f}m",flush=True)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
        print(f"[{cycle+1}] reward={m['real_reward']:.4f} fallen={m['failures']} {episode_text} state_loss={m['state_loss']:.5f} 动作更新均值={m['actor_update_mean_abs']:.5f} 动作更新最大={m['actor_update_max_abs']:.5f} 想象累计奖励={m['imagined_reward_sum']:.4f} Critic尾项={m['critic_bootstrap']:.4f} 加权尾项={m['actor_weighted_critic_bootstrap']:.4f} 想象总目标={m['imagined_return']:.4f}",flush=True)
        if (cycle+1)%args.save_interval==0 or cycle+1==start+args.cycles:
            save_current_checkpoint({'format':'go2_core_normalized_v3','cycle':cycle+1,'critic_warmup_until':critic_warmup_until,'model_warmup_until':model_warmup_until,'actor':actor.state_dict(),'model':model.state_dict(),'critic':critic.state_dict(),'opt_a':opt_a.state_dict(),'opt_m':opt_m.state_dict(),'opt_v':opt_v.state_dict(),'env':asdict(cfg),'motion':str(args.motion),'adaptive_rsi':rsi.state_dict() if rsi is not None else None,'wm_priority':{'bin_frames':wm_bin_frames,'error_ema':wm_error_ema,'seen':wm_error_seen,'counts':wm_sample_counts} if args.wm_phase_priority else None})


if __name__=='__main__':main()
