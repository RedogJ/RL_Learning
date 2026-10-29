"""用一份冻结策略在 Isaac 中采集独立世界模型数据。"""
import argparse
import json
from copy import deepcopy
from datetime import datetime

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner

from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot
from go2_learning.tasks.upright_supermimic.paper_context import PaperContext, configure_ground, configure_paper_physics
from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.paper_aligned_train import clean_state, noisy_observation, td


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--policy', choices=['expert', 'actor'], required=True,
                   help='expert 使用源检查点中保存的原专家；actor 使用当前训练策略')
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2141)
    p.add_argument('--action-noise', type=float, default=.1)
    p.add_argument('--report-interval', type=int, default=200)
    add_launcher_args(p)
    args = p.parse_args()
    if min(args.num_envs, args.steps, args.report_interval) < 1 or args.action_noise < 0:
        p.error('环境数、步数和打印间隔必须为正，动作噪声不能为负')
    torch.manual_seed(args.seed)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved.get('world_model_format') != 'paper_context_v3' or abs(saved.get('control_dt', -1) - .02) > 1e-9:
        raise ValueError('需要 50 Hz paper_context_v3 检查点')
    task = task_from_snapshot(deepcopy(saved['task_config']))
    configure_paper_physics(task)
    configure_ground(task)
    cfg = SimulationCfg(dt=task.cfg.environment.physics_dt, physics=PhysxCfg())
    run = LOG_ROOT / 'upright_supermimic' / 'deeponet_study' / 'datasets' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    with launch_simulation(cfg, args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim = SimulationContext(cfg)
        try:
            env = make_env(sim, args.num_envs, task.cfg.randomization, args.seed, task=task)
            if list(env.robot.joint_names) != saved['task_config']['joint_names']:
                raise ValueError('关节顺序与检查点不一致')
            context = PaperContext(env, args.seed + 101)
            verified = context.verify()
            if max(verified['friction_max_abs_error'], verified['com_max_abs_error_m']) > 1e-5:
                raise RuntimeError('真实物理 context 回读不一致')
            env.diagnostics_interval = 24
            env.detailed_diagnostics = False
            env.finish_initial_settling()
            runner = OnPolicyRunner(env=env, train_cfg=task.cfg.agent, log_dir=None, device=str(sim.device))
            student = runner.get_inference_policy(device=str(sim.device))
            teacher = deepcopy(student)
            student.load_state_dict(saved['actor'])
            teacher.load_state_dict(saved['teacher_actor'])
            teacher.eval()
            student.eval()
            # Each collection uses exactly one policy, including every environment.
            policy = teacher if args.policy == 'expert' else student
            teacher_mask = torch.full((args.num_envs,), args.policy == 'expert', device=sim.device, dtype=torch.bool)
            buf = PaperBuffer(args.steps, args.num_envs, 90, 12, sim.device)
            original_reset = env._reset_finished_envs
            terminal = {}
            def capture(dones):
                terminal['state'] = clean_state(env, env.get_observations()).clone()
                terminal['executed'] = env.actions.clone()
                return original_reset(dones)
            env._reset_finished_envs = capture
            obs = env.get_observations()
            state = clean_state(env, obs)
            failures = torch.zeros(2, device=sim.device)
            active_counts = torch.zeros(2, device=sim.device)
            with torch.no_grad():
                for step in range(args.steps):
                    active = env.policy_active.clone()
                    old, ctx = state.clone(), context.value.clone()
                    remaining = (env.max_episode_length - env.episode_length_buf).clamp_min(0) / env.max_episode_length
                    raw = env.standing_actions.clone()
                    if active.any():
                        raw[active] = policy(td(noisy_observation(old[active]))) + args.action_noise * torch.randn_like(raw[active])
                    active_counts[0 if args.policy == 'expert' else 1] += active.sum()
                    obs, _, done, extra = env.step(raw)
                    state = clean_state(env, obs)
                    fail = done.bool() & ~extra['time_outs'].bool()
                    true_next = torch.where(done.bool()[:, None], terminal['state'], state)
                    failures[0] += (fail & teacher_mask).sum()
                    failures[1] += (fail & ~teacher_mask).sum()
                    buf.add(old, terminal['executed'], ctx, true_next, fail, done.bool(), active, remaining)
                    if (step + 1) % args.report_interval == 0 or step + 1 == args.steps:
                        print(json.dumps({'step': step + 1, 'total_steps': args.steps, 'policy': args.policy,
                                          'active_transitions': active_counts.long().tolist(),
                                          'failures': failures.long().tolist()}, ensure_ascii=False), flush=True)
            output = {'world_model_format': 'deeponet_dataset_v1', 'control_dt': env.dt,
                      'source_checkpoint': args.checkpoint, 'cycle': saved['cycle'],
                      'args': {'num_envs': args.num_envs, 'replay_steps': args.steps},
                      'collection_args': vars(args), 'teacher_env_ids': teacher_mask.nonzero().flatten().cpu(),
                      'model': {k: saved['model'][k] for k in ('input_mean', 'input_std', 'delta_mean', 'delta_std')},
                      'buffer': {**{k: getattr(buf, k).cpu() for k in ('state', 'next_state', 'action', 'context', 'fail', 'done', 'active', 'remaining', 'step_id')}, 'count': buf.count},
                      'active_counts': active_counts.long().cpu(), 'failures': failures.long().cpu(),
                      'context_readback': verified}
            path = run / 'dataset.pt'
            torch.save(output, path)
            print('独立数据集：', path, flush=True)
        finally:
            SimulationContext.clear_instance()


if __name__ == '__main__':
    main()
