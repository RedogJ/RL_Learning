import json,math,torch
from pathlib import Path
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics,StateValue,TerminationHead,build_state,obs_from_state
from go2_mujoco_mimic.tasks.mimic.state_train import goal,termination_probability,policy_input
from go2_mujoco_mimic.tasks.mimic.rewards import gravity_local
@torch.no_grad()
def main():
 folder=Path('/workspace/go2_mujoco_mimic/logs/backflip_critic_mask_test_2026-11-05_07-17-43_496089')
 ck0=torch.load(folder/'model_3300.pt',map_location='cpu',weights_only=False);cfg=load_config(ck0);cfg.num_envs=1;cfg.task.observation_noise=False;cfg.task.random_reference_start=False
 env=WarpGo2Env(cfg,Path(ck0['motion']));env.enable_cuda_graph();ctx=torch.zeros(1,16,device='cuda');ctx[:,0]=.5
 out={}
 for cycle in (3290,3300,3340,3380):
  ck=torch.load(folder/f'model_{cycle}.pt',map_location='cpu',weights_only=False)
  actor=MotionPolicy(history_length=cfg.task.history_length,future_length=cfg.task.future_length,heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval();actor.load_state_dict(ck['actor'])
  model=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda().eval();model.load_state_dict(ck['model']);head=TerminationHead(cfg.task.future_length*19).cuda().eval();head.load_state_dict(ck['termination_head']);critic=StateValue().cuda().eval();critic.load_state_dict(ck['critic'])
  torch.manual_seed(123);env.reset(torch.ones(1,dtype=torch.bool,device='cuda'));rows=[];states=[]
  for t in range(65):
   s=build_state(env).clone();frame=env.frame.clone();lim=env.frame_limit.clone();hist=env.history.clone();prev=env.previous_action.clone();act=actor(env.observation());states.append((s,frame,lim,hist,prev))
   with torch.autocast('cuda',dtype=torch.bfloat16):ns=model(s,act,ctx)
   predframe=torch.minimum(frame+1,lim);p_pred=termination_probability(env,model,head,ns,predframe,lim)
   _,rew,term,trunc,terms=env.step(act);truth=build_state(env);ref=env.ref.sample(env.frame);p_true=termination_probability(env,model,head,truth,env.frame,lim)
   row={'step':t+1,'t':round((t+1)*env.dt,5),'z':float(truth[0,2]),'ref_z':float(ref['pos'][0,2]),'vz':float(truth[0,21]),'ref_vz':float(env.ref.velocity[env.frame[0],2]),'zerr':float((truth[:,2]-ref['pos'][:,2]).abs()),'failed':bool(term),'end':bool(trunc),'failure_p_pred':float(p_pred),'failure_p_true':float(p_true),'reward':float(rew),'terms_rate':{k:float(v) for k,v in terms.items()},'critic_before':float(critic(s,goal(env,frame,lim),ctx)),'action':act[0].cpu().tolist(),'target_error_mean_abs':float((env.default_q+env.action_scales*act-env.qpos[:,env.qids]).abs().mean()),'torque_rms':float((env.torque_squared_sum/env.substeps).mean().sqrt())}
   rows.append(row)
   if bool(term|trunc):break
  horizons={}
  for start in [10,15,20,24]:
   if start>=len(states):continue
   s,frame,lim,hist,prev=[x.clone() for x in states[start]];surv=1.;plist=[];zs=[]
   for j in range(16):
    action=actor(policy_input(env,s,hist,frame,lim))
    with torch.autocast('cuda',dtype=torch.bfloat16):ns=model(s,action,ctx)
    nf=torch.minimum(frame+1,lim);p=float(termination_probability(env,model,head,ns,nf,lim));surv*=1-p;plist.append(p);zs.append(float(ns[0,2]));hist=torch.cat((hist[:,1:],obs_from_state(ns,action,env.default_q,noise=False)[:,None]),1);s,frame,prev=ns,nf,action
   actual=rows[start:min(start+16,len(rows))];horizons[str(start)]={'phase_s':round(start*env.dt,3),'pred_failure_probability':1-surv,'prob_steps':plist,'pred_z':zs,'actual_failed_within_window':any(x['failed'] for x in actual),'actual_window_steps':len(actual),'real_end_height':actual[-1]['z']}
  out[str(cycle)]={'rows':rows,'horizons':horizons}
  print('cycle',cycle,'steps',len(rows),'last',json.dumps(rows[-1]),'windows',json.dumps({k:{x:v[x] for x in ('phase_s','pred_failure_probability','actual_failed_within_window','actual_window_steps')} for k,v in horizons.items()}),flush=True)
  del ck,actor,model,head,critic
 Path('/tmp/go2_regression_deep.json').write_text(json.dumps(out))
if __name__=='__main__':main()
