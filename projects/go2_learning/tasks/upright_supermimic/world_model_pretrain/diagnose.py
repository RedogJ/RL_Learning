"""Evaluate a pretrained world model against Isaac using one expert policy."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner

from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot
from go2_learning.tasks.upright_supermimic.paper_aligned_diagnose import reward_parts, summarize
from go2_learning.tasks.upright_supermimic.paper_aligned_train import clean_state
from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.paper_context import PaperContext, configure_ground, configure_paper_physics
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel
from go2_learning.tasks.upright_supermimic.state_bridge import STATE_DIM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='世界模型预训练检查点')
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--seed', type=int, default=4141)
    p.add_argument('--report-interval', type=int, default=100)
    add_launcher_args(p)
    args = p.parse_args()
    if min(args.num_envs, args.steps, args.samples, args.horizon, args.report_interval) < 1 or args.steps < args.horizon:
        p.error('数量应为正，且采集步数不能小于预测长度')
    path = Path(args.checkpoint)
    pretrained = torch.load(path, map_location='cpu', weights_only=False)
    if pretrained.get('format') != 'world_model_pretrain_v1':
        raise ValueError('需要世界模型预训练检查点')
    config_path = path.with_name('config.json')
    config = json.loads(config_path.read_text())
    if config.get('policy') != 'expert':
        raise ValueError('此诊断使用 v32 专家策略，检查点必须由单一专家数据预训练')
    source = torch.load(pretrained['source_checkpoint'], map_location='cpu', weights_only=False)
    if source.get('world_model_format') != 'paper_context_v3' or abs(source.get('control_dt', -1) - .02) > 1e-9:
        raise ValueError('源检查点不是 50 Hz paper_context_v3')

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    task = task_from_snapshot(deepcopy(source['task_config']))
    configure_paper_physics(task)
    configure_ground(task)
    cfg = SimulationCfg(dt=task.cfg.environment.physics_dt, physics=PhysxCfg())
    with launch_simulation(cfg, args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim = SimulationContext(cfg)
        try:
            env = make_env(sim, args.num_envs, task.cfg.randomization, args.seed, task=task)
            context = PaperContext(env, args.seed + 101)
            verified = context.verify()
            env.diagnostics_interval = 24
            env.detailed_diagnostics = False
            env.finish_initial_settling()
            runner = OnPolicyRunner(env=env, train_cfg=task.cfg.agent, log_dir=None, device=str(sim.device))
            policy = runner.get_inference_policy(device=str(sim.device))
            policy.load_state_dict(source['teacher_actor'])
            model = PaperWorldModel(STATE_DIM, env.num_actions).to(sim.device)
            model.load_state_dict(pretrained['model'])
            model.eval()
            buf = PaperBuffer(args.steps, args.num_envs, STATE_DIM, 24, sim.device)
            original_reset = env._reset_finished_envs
            terminal = {}

            def capture(dones):
                terminal['state'] = clean_state(env, env.get_observations()).clone()
                terminal['executed'] = env.actions.clone()
                return original_reset(dones)

            env._reset_finished_envs = capture
            obs = env.get_observations()
            state = clean_state(env, obs)
            parity = torch.zeros((), device=sim.device)
            parity_count = torch.zeros_like(parity)
            count = torch.zeros_like(parity)
            brier_sum = torch.zeros_like(parity)
            prob_sum = torch.zeros_like(parity)
            fail_total = torch.zeros_like(parity)
            prob_at_failure = torch.zeros_like(parity)
            with torch.no_grad():
                for step in range(args.steps):
                    active = env.policy_active.clone()
                    old, ctx = state.clone(), context.value.clone()
                    raw = env.standing_actions.clone()
                    remaining = (env.max_episode_length - env.episode_length_buf).clamp_min(0) / env.max_episode_length
                    if active.any():
                        raw[active] = policy(obs[active])
                    obs, reward, done, extras = env.step(raw)
                    state = clean_state(env, obs)
                    fail = done.bool() & ~extras['time_outs'].bool()
                    true_next = torch.where(done.bool()[:, None], terminal['state'], state)
                    executed = terminal['executed']
                    valid = active & ~done.bool()
                    if valid.any():
                        parts = reward_parts(true_next[valid], old[valid], raw[valid], env)
                        parity += (sum(parts.values()) - reward[valid]).abs().sum()
                        parity_count += valid.sum()
                    if active.any():
                        _, logits = model(old[active], executed[active], ctx[active])
                        prob, labels = logits.sigmoid(), fail[active].float()
                        brier_sum += (prob - labels).square().sum()
                        prob_sum += prob.sum()
                        fail_total += labels.sum()
                        prob_at_failure += (prob * labels).sum()
                        count += active.sum()
                    buf.add(old, torch.cat((executed, raw), -1), ctx, true_next, fail, done.bool(), active, remaining)
                    if (step + 1) % args.report_interval == 0 or step + 1 == args.steps:
                        print(json.dumps({'step': step + 1, 'total_steps': args.steps,
                                          'active_transitions': int(count), 'failures': int(fail_total)},
                                         ensure_ascii=False), flush=True)
                ids = torch.arange(args.num_envs, device=sim.device)
                errors = summarize(model, env, buf, ids, args.samples, args.horizon)
                result = {'seed': args.seed, 'num_envs': args.num_envs, 'steps': args.steps,
                          'sampled_windows': args.samples, 'horizon': args.horizon,
                          'pretraining_cycle': pretrained['cycle'], 'context_readback': verified,
                          'reward_reconstruction_mae_per_nonterminal_step': float(parity / parity_count.clamp_min(1)),
                          'failure_brier': float(brier_sum / count.clamp_min(1)),
                          'failure_rate': float(fail_total / count.clamp_min(1)),
                          'failure_probability_mean': float(prob_sum / count.clamp_min(1)),
                          'mean_predicted_probability_at_failures': float(prob_at_failure / fail_total.clamp_min(1)),
                          'failure_events': int(fail_total), 'prediction': errors,
                          'note': 'single v32 policy; true recorded actions and context; nonterminal windows; no critic bootstrap'}
            out = path.with_name(f'expert_diagnostic_seed{args.seed}.json')
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print('诊断结果：', out, flush=True)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            SimulationContext.clear_instance()


if __name__ == '__main__':
    main()
