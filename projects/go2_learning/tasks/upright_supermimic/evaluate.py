"""在同一 Isaac 仿真中对比原始专家和模型训练策略。"""
import argparse,json
from pathlib import Path
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--pairs',type=int,default=4)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--randomization',action=argparse.BooleanOptionalAction,default=False,
                   help='开启初始关节随机化与随机推力；两组扰动为独立采样')
    p.add_argument('--push-force-range',type=float,nargs=2,default=None,metavar=('MIN','MAX'),
                   help='仅本次评估覆盖推力幅度范围（N），须开启 --randomization')
    add_launcher_args(p);args=p.parse_args()
    if args.pairs<1 or args.steps<1:p.error('pairs、steps 须为正数')
    if args.push_force_range is not None and (not args.randomization or not 0 <= args.push_force_range[0] <= args.push_force_range[1]):
        p.error('--push-force-range 须与 --randomization 一起使用，且范围应为 0 ≤ MIN ≤ MAX')
    torch.manual_seed(args.seed)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    teacher_file=Path(state['teacher']);cfg=state.get('task_config') or json.loads(teacher_file.with_name('run.json').read_text())
    task=task_from_snapshot(cfg)
    if args.push_force_range is not None:
        task.cfg.randomization.push_force_range=tuple(args.push_force_range)
    # 同一确定性初始状态，避免不同随机初态干扰对比。
    task.cfg.reset.orientation_cone_degrees=0.
    sim_cfg=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(sim_cfg,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim=SimulationContext(sim_cfg)
        try:
            env=make_env(sim,args.pairs*2,task.cfg.randomization if args.randomization else None,args.seed,task=task)
            if list(env.robot.joint_names)!=cfg['joint_names']:raise ValueError('关节顺序不匹配')
            for k,v in [('default_joint_pos',env.default_joint_pos[0]),('action_joint_limits',env.action_joint_limits[0])]:
                if not torch.allclose(torch.tensor(cfg[k],device=env.device),v,atol=1e-6,rtol=0):raise ValueError(f'{k} 不匹配')
            env.diagnostics_interval=24;env.detailed_diagnostics=False;env.finish_initial_settling()
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            if 'teacher_actor' not in state:runner.load(str(teacher_file),map_location=str(sim.device))
            student=runner.get_inference_policy(device=str(sim.device))
            from copy import deepcopy
            expert=deepcopy(student)
            if 'teacher_actor' in state:expert.load_state_dict(state['teacher_actor'])
            student.load_state_dict(state['actor'])
            obs=env.get_observations(); totals=torch.zeros(2,6,device=sim.device)
            with torch.no_grad():
                for _ in range(args.steps):
                    active=env.policy_active.clone();a=env.standing_actions.clone()
                    for group,policy in ((0,expert),(1,student)):
                        mask=active.clone();mask[:args.pairs] &= group==0;mask[args.pairs:] &= group==1
                        if mask.any():a[mask]=policy(obs[mask])
                    obs,r,done,extra=env.step(a)
                    failed=done.bool() & ~extra['time_outs'].bool()
                    for group,sl in ((0,slice(0,args.pairs)),(1,slice(args.pairs,None))):
                        ok=active[sl];done_ok=done[sl].bool();nonterminal=ok & ~done_ok
                        totals[group,0]+=ok.sum();totals[group,1]+=failed[sl].sum();totals[group,2]+=extra['time_outs'][sl].sum();totals[group,3]+=r[sl].sum()
                        totals[group,4]+=(-obs['policy'][sl,6][nonterminal]).sum();totals[group,5]+=nonterminal.sum()
            for group,label in ((0,'teacher'),(1,'student')):
                t=totals[group].cpu().tolist();out={'policy':label,'randomization':args.randomization,'push_force_range':task.cfg.randomization.push_force_range if args.randomization else None,'active_seconds':round(t[0]*env.dt,3),'failures':int(t[1]),'timeouts':int(t[2]),'reward_per_active_second':t[3]/max(t[0]*env.dt,1e-9),'mean_upright_alignment':t[4]/max(t[5],1)}
                print(json.dumps(out,ensure_ascii=False),flush=True)
        finally:SimulationContext.clear_instance()

if __name__=='__main__':main()
