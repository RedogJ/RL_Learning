import math,json,torch
from pathlib import Path
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics,StateValue,TerminationHead,build_state,obs_from_state
from go2_mujoco_mimic.tasks.mimic.state_train import goal,termination_probability,policy_input
from go2_mujoco_mimic.tasks.mimic.adaptive_rsi import AdaptiveRSI
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector,estimate_torque_squared

@torch.no_grad()
def main():
 torch.manual_seed(9132); ck=torch.load('/workspace/go2_mujoco_mimic/logs/backflip_v19_bce_2026-11-05_04-25-36_439009/model_3280.pt',map_location='cuda',weights_only=False)
 cfg=load_config(ck);cfg.num_envs=256;cfg.task.observation_noise=True
 env=WarpGo2Env(cfg,Path(ck['motion']));env.enable_cuda_graph()
 rsi=AdaptiveRSI(len(env.ref.frames)-1,env.ref.fps,env.device,bin_seconds=.2,control_dt=env.dt)
 if not cfg.task.reference_loop:rsi.gmax=1.
 rsi.load_state_dict(ck['adaptive_rsi']);mask=torch.ones(cfg.num_envs,device='cuda',dtype=torch.bool)
 env.reset(mask,rsi.probabilities(),rsi.frames_per_bin,start_probability=.2 if not cfg.task.reference_loop else 0.)
 actor=MotionPolicy(heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval();actor.load_state_dict(ck['actor'])
 model=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda().eval();model.load_state_dict(ck['model'])
 head=TerminationHead(cfg.task.future_length*19).cuda().eval();head.load_state_dict(ck['termination_head'])
 critic=StateValue().cuda().eval();critic.load_state_dict(ck['critic'])
 ctx=torch.zeros(cfg.num_envs,16,device='cuda');ctx[:,0]=.5
 columns={k:[] for k in ['s','hist','frame','limit','prev','value','reward','done','alive','fail']}
 alive=mask.clone(); print('Collecting on-policy trajectories from checkpoint RSI, with observation noise; no action noise; no learning.',flush=True)
 for t in range(math.ceil(cfg.episode_seconds/env.dt)+2):
  st=build_state(env).clone();frame=env.frame.clone();limit=env.frame_limit.clone();history=env.history.clone();prev=env.previous_action.clone();v=critic(st,goal(env,frame,limit),ctx)
  action=actor(env.observation());_,reward,failed,truncated,_=env.step(action); done=failed|truncated
  for k,val in [('s',st),('hist',history),('frame',frame),('limit',limit),('prev',prev),('value',v),('reward',reward),('done',done),('alive',alive.clone()),('fail',failed)]:columns[k].append(val.clone())
  alive &= ~done
  if not alive.any():break
 c={k:torch.stack(v) for k,v in columns.items()};T,N=c['alive'].shape
 if alive.any():raise RuntimeError('trajectory did not finish')
 mc=torch.zeros_like(c['reward']);running=torch.zeros(N,device='cuda')
 for t in reversed(range(T)):
  running=c['reward'][t]+.99*(~c['done'][t]).float()*running
  mc[t]=running
 eligible=c['alive'].nonzero();torch.manual_seed(9876)
 idx=eligible[torch.randperm(len(eligible),device='cuda')[:min(768,len(eligible))]];ti,ei=idx.unbind(1);B=len(idx)
 selected={k:v[ti,ei].clone() for k,v in c.items()};target_mc=mc[ti,ei]
 print('collected',T,'steps; valid states',len(eligible),'sampled',B,'failures',int((c['alive']&c['fail']).sum()),flush=True)
 # Real lambda targets from actual on-policy transitions, using the same frozen critic at bootstrap.
 real_rewards=[];real_dones=[];real_next_values=[]
 for j in range(16):
  tt=(ti+j).clamp(max=T-1);nextt=(ti+j+1).clamp(max=T-1)
  real_rewards.append(c['reward'][tt,ei]);real_dones.append(c['done'][tt,ei]);real_next_values.append(c['value'][nextt,ei])
 rv=torch.stack(real_next_values,1);rr=torch.stack(real_rewards,1);rd=torch.stack(real_dones,1)
 target=rv[:,-1]
 for j in reversed(range(16)):target=rr[:,j]+.99*(~rd[:,j]).float()*(.05*rv[:,j]+.95*target)
 real_lambda=target
 # Training-style imagined rollout starting from actual on-policy states/history.
 s=selected['s'];hist=selected['hist'];frame=selected['frame'];limit=selected['limit'];prev=selected['prev'];context=torch.zeros(B,16,device='cuda');context[:,0]=.5
 values=[];rewards=[];dones=[];invalid=[];survive=torch.ones(B,device='cuda')
 for j in range(16):
  invalid.append(frame >= limit);values.append(critic(s,goal(env,frame,limit),context)); action=actor(policy_input(env,s,hist,frame,limit))
  with torch.autocast('cuda',dtype=torch.bfloat16):ns=model(s,action,context)
  advance=math.floor((j+1)*env.dt*env.ref.fps)-math.floor(j*env.dt*env.ref.fps);nf=torch.minimum(frame+advance,limit)
  a=env.ref.sample(nf);b=env.ref.sample(frame)
  reference={'pos':a['body_pos'],'quat':a['body_quat'],'linear_vel':(a['body_pos']-b['body_pos'])/env.dt,'joint_pos':a['joints'],'angular_vel':rotation_vector(a['body_quat'],b['body_quat'])/env.dt}
  tq=env.default_q+env.action_scales*action
  if cfg.clamp_joint_targets:tq=tq.clamp(env.joint_limits[:,0],env.joint_limits[:,1])
  torque=None
  if cfg.task.reward.torque_l2_weight and cfg.task.reward.torque_mode=='substep_mean':torque=estimate_torque_squared(s[:,7:19],s[:,25:37],ns[:,7:19],ns[:,25:37],tq,cfg.kp,cfg.kd,cfg.torque_limit,env.substeps)
  rew,_=env.task.reward(env.kinematics.transition(s,ns,env.dt),reference,action,prev,ns[:,7:19],env.joint_limits,env.dt,joint_vel=ns[:,25:37],prev_joint_vel=s[:,25:37],target=tq,torque_squared_mean=torque,kp=cfg.kp,kd=cfg.kd,torque_limit=cfg.torque_limit)
  fail=termination_probability(env,model,head,ns,nf,limit);rewards.append(rew-cfg.task.reward.failure_penalty*fail)
  dones.append(1-(1-fail)*(nf<limit).float());hist=torch.cat((hist[:,1:],obs_from_state(ns,action,env.default_q,noise=True)[:,None]),1);s,frame,prev=ns,nf,action
 values=torch.stack(values,1);rewards=torch.stack(rewards,1);dones=torch.stack(dones,1)
 final=critic(s,goal(env,frame,limit),context);nextv=torch.cat((values[:,1:],final[:,None]),1);target=final
 targets=[]
 for j in reversed(range(16)):
  target=rewards[:,j]+.99*(1-dones[:,j])*(.05*nextv[:,j]+.95*target);targets.append(target)
 targets=torch.stack(list(reversed(targets)),1)
 imag_lambda=target
 invalid=torch.stack(invalid,1)
 def stats(a,b):
  x=a-b;return {'bias':round(x.mean().item(),5),'mae':round(x.abs().mean().item(),5),'rmse':round(x.square().mean().sqrt().item(),5)}
 value=selected['value']; bins=(selected['frame']*5//(len(env.ref.frames)-1)).clamp(max=4)
 out={'config':{'cycle':3280,'seed':9132,'sample_seed':9876,'envs':N,'states':B,'observ_noise':True,'action_noise':False,'horizon':16,'rsi':'checkpoint probabilities; first reset only','valid_real_transitions':len(eligible),'fails':int((c['alive']&c['fail']).sum()),'dt':env.dt,'ref_fps':env.ref.fps,'reference_frames':len(env.ref.frames)},'means':{k:round(v.mean().item(),5) for k,v in [('critic',value),('imag_lambda',imag_lambda),('real_lambda',real_lambda),('real_mc',target_mc)]},'all':{'critic_vs_imag_lambda':stats(value,imag_lambda),'critic_vs_real_lambda':stats(value,real_lambda),'imag_vs_real_lambda':stats(imag_lambda,real_lambda),'critic_vs_real_mc':stats(value,target_mc),'imag_lambda_vs_real_mc':stats(imag_lambda,target_mc),'real_lambda_vs_real_mc':stats(real_lambda,target_mc)},'bins':{},'past_reference_end':{'fraction_all_critic_training_states':round(invalid.float().mean().item(),5),'windows_containing_invalid':round(invalid.any(1).float().mean().item(),5),'mean_invalid_target':round(targets[invalid].mean().item(),5),'mean_invalid_critic':round(values[invalid].mean().item(),5),'invalid_target_abs_mean':round(targets[invalid].abs().mean().item(),5),'invalid_fraction_by_step':[round(z,4) for z in invalid.float().mean(0).tolist()]}}
 for b in range(5):
  m=bins==b
  if m.any():out['bins'][str(b)]={'n':int(m.sum()),'critic_vs_mc':stats(value[m],target_mc[m]),'critic_vs_imag_lambda':stats(value[m],imag_lambda[m]),'imag_vs_real_lambda':stats(imag_lambda[m],real_lambda[m]),'real_lambda_vs_mc':stats(real_lambda[m],target_mc[m])}
 print(json.dumps(out,indent=2),flush=True)
 with open('/tmp/go2_critic_audit_final_result.json','w') as f:json.dump(out,f,indent=2)
if __name__=='__main__':main()
