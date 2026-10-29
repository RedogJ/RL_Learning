"""对真实动作序列比较世界模型与 Isaac 的多步状态、奖励预测。"""
import argparse
import json
from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner

from go2_learning.tasks.upright.rewards import reward_components
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot
from go2_learning.tasks.upright_supermimic.dynamics import Dynamics
from go2_learning.tasks.upright_supermimic.sequence_buffer import SequenceBuffer
from go2_learning.tasks.upright_supermimic.state_bridge import (
    STATE_DIM, observe, predicted_state, enforce_action_state, enforce_context,
)


class PaperDynamics(Dynamics):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, 2048), nn.SiLU(),
                                 nn.Linear(2048, 2048), nn.SiLU(),
                                 nn.Linear(2048, 2048), nn.SiLU(),
                                 nn.Linear(2048, obs_dim + 2))


def reward_parts(x, old, action, env):
    state = predicted_state(x, action, env, old)
    failed = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    return reward_components(state, action, failed, env.dt, env.task.cfg.rewards)


def summarize(model, env, buf, ids, samples, horizon, conditional):
    obs, actions, actual = buf.windows(samples, horizon, ids)
    pred = obs[:, 0]
    accumulated_pred = torch.zeros(samples, device=pred.device)
    accumulated_true = torch.zeros_like(accumulated_pred)
    output = {}
    for k in range(horizon):
        prev = pred
        actual_prev = obs[:, k]
        nxt, _, _ = model(prev, actions[:, k])
        pred = enforce_action_state(nxt, prev, actions[:, k], env.action_filter_alpha)
        pred = enforce_context(pred, prev, env.dt, actual[:, k] if conditional else None)
        pred_parts = reward_parts(pred, prev, actions[:, k], env)
        true_parts = reward_parts(actual[:, k], actual_prev, actions[:, k], env)
        pr = sum(pred_parts.values()); tr = sum(true_parts.values())
        accumulated_pred += (.99**k)*pr
        accumulated_true += (.99**k)*tr
        if (k+1) not in (1, 5, 8, 16, 32):
            continue
        diff = (pred-actual[:,k])/model.delta_std
        norm_mse = torch.cat((diff[:,:80],diff[:,88:]),-1).square().mean()
        row = {'normalized_state_mse': float(norm_mse),
               'joint_position_rmse_rad': float((pred[:,9:21]-actual[:,k,9:21]).square().mean().sqrt()),
               'reward_rate_pred_mean': float(pr.mean()/env.dt),
               'reward_rate_true_mean': float(tr.mean()/env.dt),
               'reward_rate_bias': float((pr-tr).mean()/env.dt),
               'reward_rate_mae': float((pr-tr).abs().mean()/env.dt),
               'discounted_return_pred_mean': float(accumulated_pred.mean()),
               'discounted_return_true_mean': float(accumulated_true.mean())}
        for term in ('rear_symmetry','action_rate','upright_hold','height','angular_velocity_x'):
            err = (pred_parts[term]-true_parts[term])/env.dt
            row[term+'_bias'] = float(err.mean())
            row[term+'_mae'] = float(err.abs().mean())
        output[str(k+1)] = row
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--num-envs',type=int,default=32)
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--push-force-range',type=float,nargs=2,default=None,metavar=('MIN','MAX'))
    add_launcher_args(p); args=p.parse_args()
    if args.num_envs < 2 or args.num_envs % 2 or args.steps < args.horizon or args.samples < 1:
        p.error('环境数须为正偶数，步数不少于窗口，样本数须为正')
    if args.push_force_range is not None and not 0 <= args.push_force_range[0] <= args.push_force_range[1]:
        p.error('推力范围应为 0 ≤ MIN ≤ MAX')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    path=Path(args.checkpoint)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    cfg=deepcopy(saved['task_config']);task=task_from_snapshot(cfg)
    if args.push_force_range is not None:
        task.cfg.randomization.push_force_range=tuple(args.push_force_range)
    task.cfg.reset.orientation_cone_degrees=0.0
    sim_cfg=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(sim_cfg,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim=SimulationContext(sim_cfg)
        try:
            env=make_env(sim,args.num_envs,task.cfg.randomization,args.seed,task=task)
            if list(env.robot.joint_names)!=cfg['joint_names']:
                raise ValueError('关节顺序不匹配')
            env.diagnostics_interval=24;env.detailed_diagnostics=False
            env.finish_initial_settling()
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            student=runner.get_inference_policy(device=str(sim.device));expert=deepcopy(student)
            expert.load_state_dict(saved['teacher_actor']);student.load_state_dict(saved['actor'])
            model=PaperDynamics(STATE_DIM,env.num_actions).to(sim.device)
            model.load_state_dict(saved['model']);model.eval()
            buffer=SequenceBuffer(args.steps,args.num_envs,STATE_DIM,env.num_actions,sim.device)
            obs=env.get_observations(); state=observe(env,obs['policy']);half=args.num_envs//2
            parity_abs=torch.zeros((),device=sim.device);parity_count=torch.zeros_like(parity_abs)
            failures=torch.zeros(2,device=sim.device)
            with torch.no_grad():
                for _ in range(args.steps):
                    active=env.policy_active.clone();old=state.clone();a=env.standing_actions.clone()
                    for group,policy in ((0,expert),(1,student)):
                        select=active.clone(); select[:half] &= group==0;select[half:] &= group==1
                        if select.any():a[select]=policy(obs[select])
                    obs,r,done,extra=env.step(a);state=observe(env,obs['policy'])
                    failed=done.bool() & ~extra['time_outs'].bool()
                    valid=active & ~done.bool()
                    if valid.any():
                        parts=reward_parts(state[valid],old[valid],a[valid],env)
                        parity_abs+=(sum(parts.values())-r[valid]).abs().sum()
                        parity_count+=valid.sum()
                    failures[0]+=failed[:half].sum();failures[1]+=failed[half:].sum()
                    remaining=torch.zeros(args.num_envs,device=sim.device)
                    buffer.add(old,a,state,r,failed,done.bool(),active,remaining,remaining)
                result={'seed':args.seed,'steps':args.steps,'samples':args.samples,'horizon':args.horizon,
                        'push_force_range':task.cfg.randomization.push_force_range,
                        'note':'same observed actions; valid windows exclude terminals; conditional context uses future observed pushes',
                        'reward_reconstruction_mae_per_step':float(parity_abs/parity_count.clamp_min(1)),
                        'failures':{'teacher':int(failures[0]),'student':int(failures[1])},'groups':{}}
                for group,ids in (('teacher',torch.arange(half,device=sim.device)),
                                  ('student',torch.arange(half,args.num_envs,device=sim.device))):
                    # Reuse the exact same sampled windows across the two context assumptions.
                    rng=torch.cuda.get_rng_state()
                    conditional=summarize(model,env,buffer,ids,args.samples,args.horizon,True)
                    torch.cuda.set_rng_state(rng)
                    autonomous=summarize(model,env,buffer,ids,args.samples,args.horizon,False)
                    result['groups'][group]={'observed_push_context':conditional,'autonomous_push_context':autonomous}
                force_tag = '' if args.push_force_range is None else '_force_' + '_'.join(f'{x:g}' for x in args.push_force_range)
                dest=path.with_name(f'model_rollout_diagnostic_{args.seed}{force_tag}.json')
                dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
                print('MODEL_DIAGNOSTIC',dest,flush=True)
                print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:SimulationContext.clear_instance()


if __name__=='__main__':main()
