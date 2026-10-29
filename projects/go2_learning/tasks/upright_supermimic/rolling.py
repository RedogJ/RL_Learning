"""Isaac Lab 滚动采样、世界模型和短时可微想象的实验训练入口。"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from torch import nn
from tensordict import TensorDict
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner
from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright.task import UprightTask
from go2_learning.training.checkpoints import resolve_checkpoint
from go2_learning.tasks.upright_supermimic.world_model import WorldModel


def td(x):
    return TensorDict({'policy': x}, batch_size=[x.shape[0]])


def proxy_cost(obs, action, previous):
    """可微代理目标；不包含接触或原 upright 的全部奖励。"""
    upright_error = (1.0 + obs[:, 6]).square()  # 机身 +X 指向世界上方。
    height_error = obs[:, 48].square()
    body_speed = obs[:, :3].square().mean(-1)
    angular_speed = obs[:, 3:6].square().mean(-1)
    action_change = (action - previous).square().mean(-1)
    return 10.0 * upright_error + 10.0 * height_error + body_speed + 0.05 * angular_speed + 0.05 * action_change


class Replay:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.o = torch.empty(capacity, obs_dim, device=device)
        self.a = torch.empty(capacity, action_dim, device=device)
        self.n = torch.empty_like(self.o)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    def add(self, o, a, n):
        if len(o) == 0:
            return
        if len(o) > self.capacity:
            o, a, n = o[-self.capacity:], a[-self.capacity:], n[-self.capacity:]
        ids = (torch.arange(len(o), device=o.device) + self.cursor) % self.capacity
        self.o[ids], self.a[ids], self.n[ids] = o, a, n
        self.cursor = (self.cursor + len(o)) % self.capacity
        self.size = min(self.capacity, self.size + len(o))

    def sample(self, n):
        ids = torch.randint(self.size, (min(n, self.size),), device=self.o.device)
        return self.o[ids], self.a[ids], self.n[ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='latest')
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--cycles', type=int, default=10)
    p.add_argument('--collect-steps', type=int, default=40)
    p.add_argument('--wm-updates', type=int, default=50)
    p.add_argument('--warmup-updates', type=int, default=200)
    p.add_argument('--actor-updates', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--buffer-size', type=int, default=50000)
    p.add_argument('--noise', type=float, default=.08)
    p.add_argument('--actor-lr', type=float, default=1e-5)
    p.add_argument('--teacher-weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--randomization', action=argparse.BooleanOptionalAction, default=True)
    add_launcher_args(p)
    args = p.parse_args()
    if min(args.num_envs-1, args.cycles, args.collect_steps, args.wm_updates, args.warmup_updates,
           args.actor_updates, args.batch_size, args.horizon, args.buffer_size) <= 0 or args.noise < 0:
        p.error('环境数至少为 2，其余计数必须为正，噪声非负')
    torch.manual_seed(args.seed)
    ckpt = resolve_checkpoint(args.checkpoint, 'upright', LOG_ROOT)
    saved_cfg = json.loads(ckpt.with_name('run.json').read_text())
    task = UprightTask(); task.load_playback_config(saved_cfg)
    sim_cfg = SimulationCfg(dt=task.cfg.environment.physics_dt, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim = SimulationContext(sim_cfg)
        try:
            env = make_env(sim, args.num_envs, task.cfg.randomization if args.randomization else None,
                           args.seed, task=task)
            if list(env.robot.joint_names) != saved_cfg['joint_names']:
                raise ValueError('专家与当前仿真关节顺序不同')
            for key, value in [('default_joint_pos',env.default_joint_pos[0]),('action_joint_limits',env.action_joint_limits[0])]:
                if not torch.allclose(torch.tensor(saved_cfg[key],device=env.device),value,atol=1e-6,rtol=0):
                    raise ValueError(f'专家与当前仿真 {key} 不一致')
            env.finish_initial_settling()
            runner = OnPolicyRunner(env=env, train_cfg=task.cfg.agent, log_dir=None, device=str(sim.device))
            runner.load(str(ckpt), map_location=str(sim.device))
            actor = runner.get_inference_policy(device=str(sim.device))
            from copy import deepcopy
            teacher = deepcopy(actor).eval()
            for param in teacher.parameters(): param.requires_grad_(False)
            actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
            obs = env.get_observations(); dim = obs['policy'].shape[-1]
            wm = WorldModel(dim, env.num_actions).to(sim.device)
            wm_opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
            # 固定按环境划分，避免同一环境的相邻时刻同时出现在训练和验证集中。
            split = max(1, int(args.num_envs * .8))
            train_ids = torch.arange(args.num_envs, device=sim.device) < split
            train = Replay(args.buffer_size, dim, env.num_actions, sim.device)
            test = Replay(max(1000, args.buffer_size // 5), dim, env.num_actions, sim.device)
            run_dir = LOG_ROOT / 'upright_supermimic' / ('rolling_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
            run_dir.mkdir(parents=True)
            (run_dir / 'config.json').write_text(json.dumps({'args':vars(args),'teacher':str(ckpt),'task_config':saved_cfg,'note':'experimental proxy objective; no contact prediction'}, ensure_ascii=False, indent=2, default=str))
            print(f'滚动实验：专家 {ckpt}，输出 {run_dir}，实验性代理奖励，不是完整 SuperMimic', flush=True)
            for cycle in range(args.cycles):
                failures = 0; active_steps = 0
                with torch.no_grad():
                    for _ in range(args.collect_steps):
                        active = env.policy_active.clone(); old = obs['policy'].clone()
                        actions = env.standing_actions.clone()
                        if active.any(): actions[active] = actor(obs[active]) + args.noise * torch.randn_like(actions[active])
                        obs, _, done, extras = env.step(actions)
                        valid = active & ~done.bool() & env.policy_active
                        for buffer, mask in ((train, valid & train_ids), (test, valid & ~train_ids)):
                            buffer.add(old[mask], actions[mask], obs['policy'][mask])
                        failures += int((done.bool() & ~extras['time_outs'].bool()).sum().item())
                        active_steps += int(active.sum().item())
                if not train.size or not test.size:
                    raise RuntimeError('没有有效训练或验证转移，请增加采样时间')
                if cycle == 0:
                    wm.calibrate(*train.sample(min(train.size, 10000)))
                wm.train()
                for _ in range(args.warmup_updates if cycle == 0 else args.wm_updates):
                    x, a, y = train.sample(args.batch_size)
                    loss = wm.loss(x, a, y)
                    wm_opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); wm_opt.step()
                with torch.no_grad():
                    x, a, y = test.sample(min(test.size, 4096))
                    val = wm.loss(x, a, y).item()
                    baseline = (((x-y)/wm.delta_std).square()).mean().item()
                    joint_mae = (wm(x,a)[:,9:21]-y[:,9:21]).abs().mean().item()
                # 冻结世界模型参数，但保留它对状态和动作的梯度。
                for param in wm.parameters(): param.requires_grad_(False)
                for _ in range(args.actor_updates):
                    x, _, _ = train.sample(args.batch_size); imagined = x.detach()
                    cost = torch.zeros(len(x),device=sim.device)
                    imitation = torch.zeros_like(cost)
                    for _ in range(args.horizon):
                        action = actor(td(imagined))
                        with torch.no_grad(): target = teacher(td(imagined.detach()))
                        cost = cost + proxy_cost(imagined, action, imagined[:,33:45])
                        imitation = imitation + (action-target).square().mean(-1)
                        imagined = wm(imagined, action)
                    objective = (cost + args.teacher_weight * imitation).mean() / args.horizon
                    actor_opt.zero_grad(set_to_none=True); objective.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5); actor_opt.step()
                for param in wm.parameters(): param.requires_grad_(True)
                metrics = {'cycle':cycle+1,'transitions':train.size,'validation_transitions':test.size,
                           'real_failures':failures,'real_active_steps':active_steps,'wm_validation_mse':val,
                           'persistence_baseline_mse':baseline,'joint_mae_rad':joint_mae,
                           'imagined_proxy_cost':objective.item()}
                with (run_dir/'metrics.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
                print(' | '.join(f'{k}={v:.5f}' if isinstance(v,float) else f'{k}={v}' for k,v in metrics.items()),flush=True)
                torch.save({'actor':actor.state_dict(),'world_model':wm.state_dict(),'cycle':cycle+1,
                            'teacher':str(ckpt),'metrics':metrics},run_dir/'experimental.pt')
            print(f'实验结果：{run_dir}',flush=True)
        finally:
            SimulationContext.clear_instance()


if __name__ == '__main__':
    main()
