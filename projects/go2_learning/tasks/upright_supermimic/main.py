"""采集 upright 仿真数据并训练世界模型：SuperMimic 接入的第一步。"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner

from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright.task import UprightTask
from go2_learning.training.checkpoints import resolve_checkpoint
from go2_learning.tasks.upright_supermimic.world_model import WorldModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='latest', help='upright 专家检查点')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=600, help='采集的控制步数')
    parser.add_argument('--updates', type=int, default=1000, help='世界模型更新次数')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--noise', type=float, default=0.08, help='专家动作上叠加的高斯探索噪声')
    parser.add_argument('--randomization', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=42)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 2 or args.updates < 1 or args.batch_size < 1 or args.noise < 0:
        parser.error('至少需要两个环境和两个采样步，更新次数和批量须为正，噪声非负')
    torch.manual_seed(args.seed)
    checkpoint = resolve_checkpoint(args.checkpoint, 'upright', LOG_ROOT)
    cfg = json.loads(checkpoint.with_name('run.json').read_text())
    task = UprightTask()
    task.load_playback_config(cfg)
    sim_cfg = SimulationCfg(dt=task.cfg.environment.physics_dt, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim = SimulationContext(sim_cfg)
        try:
            env = make_env(sim, args.num_envs, task.cfg.randomization if args.randomization else None,
                           args.seed, task=task)
            if list(env.robot.joint_names) != cfg['joint_names']:
                raise ValueError('专家模型与当前关节顺序不一致')
            limits = torch.tensor(cfg['action_joint_limits'], device=env.device)
            if not torch.allclose(limits, env.action_joint_limits[0], atol=1e-6, rtol=0):
                raise ValueError('专家模型与当前关节限位不一致')
            env.finish_initial_settling()
            runner = OnPolicyRunner(env=env, train_cfg=task.cfg.agent, log_dir=None, device=str(sim.device))
            runner.load(str(checkpoint), map_location=str(sim.device))
            policy = runner.get_inference_policy(device=str(sim.device))
            obs = env.get_observations()
            collected = []
            print(f'采集专家 {checkpoint}，环境 {args.num_envs}，步数 {args.steps}，滤波系数 {env.action_filter_alpha}', flush=True)
            with torch.inference_mode():
                for step in range(args.steps):
                    active = env.policy_active.clone()
                    old = obs['policy'].clone()
                    action = env.standing_actions.clone()
                    if active.any():
                        action[active] = policy(obs[active]) + args.noise * torch.randn_like(action[active])
                    obs, _, dones, _ = env.step(action)
                    # 结束和重置等待不能作为普通单步状态转移；防止跨回合污染模型。
                    valid = active & ~dones.bool() & env.policy_active
                    collected.append((old.cpu(), action.cpu(), obs['policy'].clone().cpu(), valid.cpu()))
                    if (step + 1) % 100 == 0:
                        print(f'采集 {step + 1}/{args.steps}', flush=True)
            o, a, n, valid = (torch.stack(values) for values in zip(*collected))
            split = max(1, int(args.num_envs * .8))
            train_mask = valid.clone(); train_mask[:, split:] = False
            test_mask = valid.clone(); test_mask[:, :split] = False
            if not train_mask.any() or not test_mask.any():
                raise RuntimeError('有效转移不足，请增加环境数或采样步数')
            train = (o[train_mask].to(sim.device), a[train_mask].to(sim.device), n[train_mask].to(sim.device))
            test = (o[test_mask].to(sim.device), a[test_mask].to(sim.device), n[test_mask].to(sim.device))
            model = WorldModel(train[0].shape[-1], train[1].shape[-1]).to(sim.device)
            model.calibrate(*train)
            opt = torch.optim.Adam(model.parameters(), lr=3e-4)
            print(f'有效转移：训练 {len(train[0])}，验证 {len(test[0])}；按环境划分', flush=True)
            for update in range(args.updates):
                idx = torch.randint(len(train[0]), (min(args.batch_size, len(train[0])),), device=sim.device)
                loss = model.loss(*(item[idx] for item in train))
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                if (update + 1) % 100 == 0 or update == args.updates - 1:
                    with torch.no_grad():
                        val = model.loss(*test).item()
                        mae = (model(test[0], test[1]) - test[2]).abs().mean().item()
                        baseline = (((test[0] - test[2]) / model.delta_std).square()).mean().item()
                        prediction = model(test[0], test[1])
                        joint_mae = (prediction[:, 9:21] - test[2][:, 9:21]).abs().mean().item()
                        gravity_mae = (prediction[:, 6:9] - test[2][:, 6:9]).abs().mean().item()
                    print(f'更新 {update+1}/{args.updates} 训练归一化MSE={loss.item():.4f} '
                          f'验证归一化MSE={val:.4f} 保持原状态基线={baseline:.4f} '
                          f'关节角MAE={joint_mae:.5f}rad 重力投影MAE={gravity_mae:.5f}', flush=True)
            dest = LOG_ROOT / 'upright_supermimic' / datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
            dest.mkdir(parents=True)
            torch.save({'model': model.state_dict(), 'obs_dim': train[0].shape[-1], 'action_dim': train[1].shape[-1],
                        'teacher': str(checkpoint), 'num_envs': args.num_envs, 'seed': args.seed,
                        'filter_alpha': env.action_filter_alpha, 'validation_loss': val, 'validation_mae': mae,
                        'persistence_baseline_loss': baseline, 'joint_mae_rad': joint_mae,
                        'gravity_mae': gravity_mae}, dest / 'world_model.pt')
            print(f'世界模型已保存：{dest / "world_model.pt"}', flush=True)
        finally:
            SimulationContext.clear_instance()


if __name__ == '__main__':
    main()
