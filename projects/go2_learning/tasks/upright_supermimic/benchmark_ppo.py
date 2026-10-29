"""同一专家快照、奖励、重置与随机化的独立 PPO 训练基线。"""
import argparse
import json
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner
from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot
from go2_learning.training.settling import SettlingLogger, SettlingPPO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--randomization', action=argparse.BooleanOptionalAction, default=True)
    add_launcher_args(p)
    args = p.parse_args()
    if args.num_envs < 1 or args.updates < 1:
        p.error('环境数及更新次数必须为正')
    torch.manual_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    cfg = json.loads(checkpoint.with_name('run.json').read_text())
    task = task_from_snapshot(deepcopy(cfg))
    sim_cfg = SimulationCfg(dt=task.cfg.environment.physics_dt, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim = SimulationContext(sim_cfg)
        try:
            env = make_env(sim, args.num_envs, task.cfg.randomization if args.randomization else None,
                           args.seed, task=task)
            if list(env.robot.joint_names) != cfg['joint_names']:
                raise ValueError('关节顺序不匹配')
            for key, actual in [('default_joint_pos', env.default_joint_pos[0]),
                                ('action_joint_limits', env.action_joint_limits[0])]:
                if not torch.allclose(torch.tensor(cfg[key], device=env.device), actual, atol=1e-6, rtol=0):
                    raise ValueError(f'{key} 不匹配')
            env.diagnostics_interval = 24
            env.detailed_diagnostics = False
            env.finish_initial_settling()
            name = datetime.now(timezone(timedelta(hours=8))).strftime('ppo_%Y%m%d_%H%M%S_%f')
            run = LOG_ROOT / 'upright_supermimic' / 'benchmarks' / name
            run.mkdir(parents=True)
            runner = OnPolicyRunner(env=env, train_cfg=task.cfg.agent, log_dir=str(run), device=str(sim.device))
            runner.load(str(checkpoint), map_location=str(sim.device), load_cfg={
                'actor': True, 'critic': True, 'optimizer': False, 'iteration': False, 'rnd': False,
            })
            teacher = deepcopy(runner.get_inference_policy(device=str(sim.device))).eval()
            runner.alg = SettlingPPO(runner.alg, env)
            runner.logger = SettlingLogger(runner.logger)
            (run/'config.json').write_text(json.dumps({'args': vars(args), 'teacher': str(checkpoint),
                                                       'task_config': cfg}, ensure_ascii=False, indent=2, default=str))
            torch.cuda.synchronize(); start = time.perf_counter()
            runner.learn(num_learning_iterations=args.updates)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            runner.save(str(run/'ppo_latest.pt'))
            actor = runner.get_inference_policy(device=str(sim.device))
            torch.save({'teacher': str(checkpoint), 'teacher_actor': teacher.state_dict(),
                        'actor': actor.state_dict(), 'task_config': cfg}, run/'for_evaluation.pt')
            result = {'elapsed_training_seconds': elapsed, 'updates': args.updates,
                      'num_envs': args.num_envs, 'control_slots': args.updates * args.num_envs * task.cfg.agent['num_steps_per_env']}
            (run/'timing.json').write_text(json.dumps(result, indent=2))
            print('PPO_BENCHMARK', json.dumps(result), 'OUTPUT', run, flush=True)
        finally:
            SimulationContext.clear_instance()


if __name__ == '__main__':
    main()
