"""独立 Isaac 轨迹上的论文式世界模型多步误差，不更新任何网络。"""
import argparse,json
from copy import deepcopy
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner
from go2_learning.tasks.upright.rewards import reward_components
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot
from go2_learning.tasks.upright_supermimic.paper_context import PaperContext,configure_ground,configure_paper_physics
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel,enforce_executed_action
from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.paper_aligned_train import clean_state
from go2_learning.tasks.upright_supermimic.state_bridge import STATE_DIM,predicted_state


def reward_parts(x,old,raw,env):
    return reward_components(predicted_state(x,raw,env,old),raw,
                             torch.zeros(len(x),dtype=torch.bool,device=x.device),env.dt,env.task.cfg.rewards)


def summarize(model,env,buf,ids,samples,horizon):
    windows=buf.windows(samples*3,horizon,ids)
    valid=(~windows[5].any(1)).nonzero().flatten()
    if not len(valid):raise ValueError('没有不含终止步骤的诊断窗口')
    chosen=valid[torch.randint(len(valid),(samples,),device=valid.device)]
    states,actions,contexts,actual,_,_,_=(v[chosen] for v in windows)
    pred=states[:,0];baseline=pred.clone();rp=torch.zeros(samples,device=pred.device);rt=torch.zeros_like(rp)
    rows={}
    for k in range(horizon):
        old=pred;executed=actions[:,k,:12];raw=actions[:,k,12:]
        pred,_=model(old,executed,contexts[:,k]);baseline=enforce_executed_action(baseline,executed)
        pp=reward_parts(pred,old,raw,env);tp=reward_parts(actual[:,k],states[:,k],raw,env)
        prediction=sum(pp.values());truth=sum(tp.values())
        rp+=(.99**k)*prediction;rt+=(.99**k)*truth
        if k+1 not in (1,5,8,16,32):continue
        err=(pred-actual[:,k])/model.delta_std;base=(baseline-actual[:,k])/model.delta_std
        mse=torch.cat((err[:,:80],err[:,88:]),-1).square().mean()
        base_mse=torch.cat((base[:,:80],base[:,88:]),-1).square().mean()
        rows[str(k+1)]={'normalized_state_mse':float(mse),'persistence_state_mse':float(base_mse),
                        'joint_rmse_rad':float((pred[:,9:21]-actual[:,k,9:21]).square().mean().sqrt()),
                        'reward_rate_true':float(truth.mean()/env.dt),'reward_rate_predicted':float(prediction.mean()/env.dt),
                        'reward_rate_mae':float((prediction-truth).abs().mean()/env.dt),
                        'rear_symmetry_bias_per_s':float((pp['rear_symmetry']-tp['rear_symmetry']).mean()/env.dt),
                        'discounted_sum_true':float(rt.mean()),'discounted_sum_predicted':float(rp.mean())}
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--pairs',type=int,default=8);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--samples',type=int,default=512);p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--seed',type=int,default=123);add_launcher_args(p);args=p.parse_args()
    if args.pairs<1 or args.steps<args.horizon or args.samples<1:p.error('诊断参数无效')
    torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=True
    path=Path(args.checkpoint);saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('world_model_format')!='paper_context_v3' or abs(saved.get('control_dt',-1)-.02)>1e-9:
        raise ValueError('此诊断仅支持 50 Hz paper_context_v3')
    task=task_from_snapshot(deepcopy(saved['task_config']));configure_paper_physics(task);configure_ground(task)
    cfg=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(cfg,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim=SimulationContext(cfg)
        try:
            n=args.pairs*2;env=make_env(sim,n,task.cfg.randomization,args.seed,task=task)
            context=PaperContext(env,args.seed+101);verified=context.verify()
            env.diagnostics_interval=24;env.detailed_diagnostics=False;env.finish_initial_settling()
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            student=runner.get_inference_policy(device=str(sim.device));expert=deepcopy(student)
            student.load_state_dict(saved['actor']);expert.load_state_dict(saved['teacher_actor'])
            model=PaperWorldModel(STATE_DIM,env.num_actions).to(sim.device);model.load_state_dict(saved['model']);model.eval()
            buf=PaperBuffer(args.steps,n,STATE_DIM,24,sim.device)
            original=env._reset_finished_envs;terminal={}
            def capture(dones):
                terminal['state']=clean_state(env,env.get_observations()).clone()
                terminal['executed']=env.actions.clone()
                return original(dones)
            env._reset_finished_envs=capture
            obs=env.get_observations();state=clean_state(env,obs)
            parity=torch.zeros((),device=sim.device);parity_count=torch.zeros_like(parity);count=torch.zeros_like(parity)
            fail_count=torch.zeros(2,device=sim.device)
            # 对所有有效单步估计终止概率校准；终止标签来自真实仿真。
            brier_sum=torch.zeros_like(parity);prob_sum=torch.zeros_like(parity);fail_total=torch.zeros_like(parity)
            prob_at_failure=torch.zeros_like(parity)
            with torch.no_grad():
                for _ in range(args.steps):
                    active=env.policy_active.clone();old=state.clone();ctx=context.value.clone()
                    raw=env.standing_actions.clone();remaining=(env.max_episode_length-env.episode_length_buf).clamp_min(0)/env.max_episode_length
                    for i,policy in ((0,expert),(1,student)):
                        mask=active.clone();mask[:args.pairs]&=i==0;mask[args.pairs:]&=i==1
                        if mask.any():raw[mask]=policy(obs[mask])
                    obs,reward,done,extras=env.step(raw);state=clean_state(env,obs)
                    fail=done.bool() & ~extras['time_outs'].bool();true_next=torch.where(done.bool()[:,None],terminal['state'],state)
                    executed=terminal['executed'];valid=active & ~done.bool()
                    if valid.any():
                        parts=reward_parts(true_next[valid],old[valid],raw[valid],env)
                        parity+=(sum(parts.values())-reward[valid]).abs().sum()
                        parity_count+=valid.sum()
                    if active.any():
                        _,logits=model(old[active],executed[active],ctx[active]);prob=logits.sigmoid();labels=fail[active].float()
                        brier_sum+=(prob-labels).square().sum();prob_sum+=prob.sum();fail_total+=labels.sum()
                        prob_at_failure+=(prob*labels).sum();count+=active.sum()
                    fail_count[0]+=fail[:args.pairs].sum();fail_count[1]+=fail[args.pairs:].sum()
                    buf.add(old,torch.cat((executed,raw),-1),ctx,true_next,fail,done.bool(),active,remaining)
                result={'seed':args.seed,'pairs':args.pairs,'steps':args.steps,'sampled_windows':args.samples,
                        'horizon':args.horizon,'format':saved['world_model_format'],'cycle':saved['cycle'],
                        'context_readback':verified,'reward_reconstruction_mae_per_nonterminal_step':float(parity/parity_count.clamp_min(1)),
                        'failure_brier':float(brier_sum/count.clamp_min(1)),
                        'failure_rate':float(fail_total/count.clamp_min(1)),
                        'failure_probability_mean':float(prob_sum/count.clamp_min(1)),
                        'mean_predicted_probability_at_failures':float(prob_at_failure/fail_total.clamp_min(1)),
                        'failures':{'teacher':int(fail_count[0]),'student':int(fail_count[1])},'groups':{},
                        'note':'unseen seed; actual fixed action sequence and true recorded episode context; nonterminal windows; no critic bootstrap'}
                for label,ids in (('teacher',torch.arange(args.pairs,device=sim.device)),
                                  ('student',torch.arange(args.pairs,n,device=sim.device))):
                    result['groups'][label]=summarize(model,env,buf,ids,args.samples,args.horizon)
            out=path.with_name(f'independent_model_diagnostic_nonterminal_seed{args.seed}.json')
            out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print('INDEPENDENT_MODEL_DIAGNOSTIC',out,flush=True);print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:SimulationContext.clear_instance()


if __name__=='__main__':main()
