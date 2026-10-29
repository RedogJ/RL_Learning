"""在真实 Isaac 中评估论文 context 实验的 Actor 与 v32 专家。"""
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
from go2_learning.tasks.upright_supermimic.paper_context import PaperContext,configure_ground,configure_paper_physics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--pairs',type=int,default=8);p.add_argument('--steps',type=int,default=600)
    p.add_argument('--seed',type=int,default=123);add_launcher_args(p);args=p.parse_args()
    if args.pairs<1 or args.steps<1:p.error('评估参数须为正')
    torch.manual_seed(args.seed)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved.get('world_model_format')!='continuous_mlp_v1':raise ValueError('检查点不是连续状态 MLP 在线训练版本')
    if abs(saved.get('control_dt',-1.)-0.02)>1e-9:raise ValueError('检查点控制频率不是论文的 50 Hz')
    task=task_from_snapshot(deepcopy(saved['task_config']));configure_paper_physics(task);configure_ground(task)
    cfg=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(cfg,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim=SimulationContext(cfg)
        try:
            env=make_env(sim,args.pairs*2,task.cfg.randomization,args.seed,task=task)
            context=PaperContext(env,args.seed+101);verified=context.verify()
            env.diagnostics_interval=24;env.detailed_diagnostics=False;env.finish_initial_settling()
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            student=runner.get_inference_policy(device=str(sim.device));expert=deepcopy(student)
            student.load_state_dict(saved['actor']);expert.load_state_dict(saved['teacher_actor'])
            obs=env.get_observations();totals=torch.zeros(2,5,device=sim.device)
            with torch.no_grad():
                for _ in range(args.steps):
                    active=env.policy_active.clone();actions=env.standing_actions.clone()
                    for i,policy in ((0,expert),(1,student)):
                        mask=active.clone();mask[:args.pairs]&=i==0;mask[args.pairs:]&=i==1
                        if mask.any():actions[mask]=policy(obs[mask])
                    obs,reward,done,extra=env.step(actions)
                    failed=done.bool() & ~extra['time_outs'].bool()
                    for i,sl in ((0,slice(0,args.pairs)),(1,slice(args.pairs,None))):
                        totals[i,0]+=active[sl].sum();totals[i,1]+=failed[sl].sum();totals[i,2]+=extra['time_outs'][sl].sum()
                        totals[i,3]+=reward[sl].sum();totals[i,4]+=(-obs['policy'][sl,6][active[sl] & ~done[sl].bool()]).sum()
            for i,name in ((0,'teacher'),(1,'student')):
                t=totals[i].cpu().tolist()
                print(json.dumps({'policy':name,'seed':args.seed,'active_seconds':t[0]*env.dt,'failures':int(t[1]),
                                  'timeouts':int(t[2]),'reward_per_active_second':t[3]/max(t[0]*env.dt,1e-9),
                                  'alignment_per_active_step_approx':t[4]/max(t[0],1),
                                  'independent_contexts':True,'context_readback':verified},ensure_ascii=False),flush=True)
        finally:SimulationContext.clear_instance()


if __name__=='__main__':main()
