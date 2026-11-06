import math,copy,json,torch
from pathlib import Path
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics,StateValue,TerminationHead,build_state,obs_from_state
from go2_mujoco_mimic.tasks.mimic.state_train import goal,termination_probability,policy_input
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector,estimate_torque_squared
from go2_mujoco_mimic.tasks.mimic.adaptive_rsi import AdaptiveRSI

def main():
 torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=True
 ck=torch.load('/workspace/go2_mujoco_mimic/logs/backflip_critic_mask_test_2026-11-05_07-17-43_496089/model_3380.pt',map_location='cpu',weights_only=False)
 cfg=load_config(ck);cfg.num_envs=32;cfg.task.observation_noise=False;cfg.task.random_reference_start=True
 env=WarpGo2Env(cfg,Path(ck['motion']));env.enable_cuda_graph();device='cuda';ctx=torch.zeros(cfg.num_envs,16,device=device);ctx[:,0]=.5
 rsi=AdaptiveRSI(len(env.ref.frames)-1,env.ref.fps,device,bin_seconds=.2,control_dt=env.dt)
 if not cfg.task.reference_loop:rsi.gmax=1.
 rsi.load_state_dict(ck['adaptive_rsi']);allmask=torch.ones(cfg.num_envs,dtype=torch.bool,device=device)
 actor=MotionPolicy(history_length=cfg.task.history_length,future_length=cfg.task.future_length,heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).to(device).eval();actor.load_state_dict(ck['actor'])
 model=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).to(device).eval();model.load_state_dict(ck['model']);model.requires_grad_(False)
 head=TerminationHead(cfg.task.future_length*19).to(device).eval();head.load_state_dict(ck['termination_head']);head.requires_grad_(False)
 critic=StateValue().to(device).eval();critic.load_state_dict(ck['critic']);critic.requires_grad_(False)
 params=list(actor.parameters());base=[p.detach().clone() for p in params];H=16;gamma=.99
 def reset(seed,mode='rsi'):
  torch.manual_seed(seed)
  if mode=='rsi':env.reset(allmask,rsi.probabilities(),rsi.frames_per_bin,start_probability=.2 if not cfg.task.reference_loop else 0.)
  else:
   old=cfg.task.random_reference_start;cfg.task.random_reference_start=False;env.reset(allmask);cfg.task.random_reference_start=old
  return (build_state(env).clone(),env.history.clone(),env.frame.clone(),env.frame_limit.clone(),env.previous_action.clone())
 def imagine(initial):
  s,hist,frame,lim,prev=[x.clone() for x in initial];alive_ref=torch.ones(len(s),device=device);survive=torch.ones_like(alive_ref);raw=torch.zeros_like(alive_ref);soft=torch.zeros_like(alive_ref);pose=None
  for t in range(H):
   action=actor(policy_input(env,s,hist,frame,lim))
   with torch.autocast('cuda',dtype=torch.bfloat16):ns=model(s,action,ctx)
   adv=math.floor((t+1)*env.dt*env.ref.fps)-math.floor(t*env.dt*env.ref.fps);nf=torch.minimum(frame+adv,lim);a=env.ref.sample(nf);b=env.ref.sample(frame)
   ref={'pos':a['body_pos'],'quat':a['body_quat'],'linear_vel':(a['body_pos']-b['body_pos'])/env.dt,'joint_pos':a['joints'],'angular_vel':rotation_vector(a['body_quat'],b['body_quat'])/env.dt}
   actual,pose=env.kinematics.transition_with_pose(pose,s,ns,env.dt);tq=env.default_q+env.action_scales*action
   if cfg.clamp_joint_targets:tq=tq.clamp(env.joint_limits[:,0],env.joint_limits[:,1])
   tq2=None
   if cfg.task.reward.torque_l2_weight and cfg.task.reward.torque_mode=='substep_mean':tq2=estimate_torque_squared(s[:,7:19],s[:,25:37],ns[:,7:19],ns[:,25:37],tq,cfg.kp,cfg.kd,cfg.torque_limit,env.substeps)
   reward,_=env.task.reward(actual,ref,action,prev,ns[:,7:19],env.joint_limits,env.dt,joint_vel=ns[:,25:37],prev_joint_vel=s[:,25:37],target=tq,torque_squared_mean=tq2,kp=cfg.kp,kd=cfg.kd,torque_limit=cfg.torque_limit)
   f=termination_probability(env,model,head,ns,nf,lim)
   raw=raw+(gamma**t)*alive_ref*reward
   soft=soft+(gamma**t)*survive*(reward-cfg.task.reward.failure_penalty*f)
   alive_ref=alive_ref*(nf<lim).float();survive=survive*(1-f)*(nf<lim).float()
   hist=torch.cat((hist[:,1:],obs_from_state(ns,action,env.default_q,noise=False)[:,None]),1);s,frame,prev=ns,nf,action
  tail=(gamma**H)*survive*critic(s,goal(env,frame,lim),ctx)
  return {'reward':raw,'termination':soft-raw,'critic':tail,'total':soft+tail}
 def gradients(vals):
  gs={}
  for name in ('reward','termination','critic'):
   g=torch.autograd.grad(vals[name].mean(),params,retain_graph=True,allow_unused=False)
   gs[name]=[x.detach().clone() for x in g]
  gs['total']=[a+b+c for a,b,c in zip(gs['reward'],gs['termination'],gs['critic'])]
  return gs
 def dot(a,b):return sum((x*y).sum() for x,y in zip(a,b))
 def install(direction,scale):
  with torch.no_grad():
   for p,b,d in zip(params,base,direction):p.copy_(b+scale*d)
 def restore():
  with torch.no_grad():
   for p,b in zip(params,base):p.copy_(b)
 def real(initial):
  active=torch.ones(cfg.num_envs,dtype=torch.bool,device=device);ret=torch.zeros(cfg.num_envs,device=device);fail=torch.zeros_like(active);dur=torch.zeros_like(ret)
  for t in range(math.ceil(cfg.episode_seconds/env.dt)+1):
   a=actor(env.observation());_,r,f,tr,_=env.step(a);ret+=gamma**t*active*r;dur+=active*env.dt;fail|=active&f;active &= ~(f|tr)
   if not active.any():break
  if active.any():raise RuntimeError('real rollout not finished')
  return {'return':ret.cpu().tolist(),'failure':fail.cpu().tolist(),'duration':dur.cpu().tolist()}
 initial=reset(123);anchors=policy_input(env,*initial[:1],initial[1],initial[2],initial[3]) if False else policy_input(env,initial[0],initial[1],initial[2],initial[3])
 with torch.no_grad():action0=actor(anchors).detach()
 with torch.enable_grad():vals=imagine(initial);gs=gradients(vals)
 norms={k:math.sqrt(dot(v,v).item()) for k,v in gs.items()};cos={f'{a}:{b}':dot(gs[a],gs[b]).item()/(norms[a]*norms[b]+1e-30) for a,b in [('reward','termination'),('reward','critic'),('termination','critic')]}
 print('GRADIENT norm',norms,'cos',cos,'mean objective',{k:round(v.mean().item(),5) for k,v in vals.items()},flush=True)
 dirs={k:[x/max(norms[k],1e-20) for x in v] for k,v in gs.items()}
 scales={}
 for name,d in dirs.items():
  scale=.001
  for i in range(6):
   install(d,scale)
   with torch.no_grad():delta=(actor(anchors)-action0).abs().mean().item()
   if delta<1e-12:scale*=10
   else:scale*=.002/delta
  scales[name]=scale;restore()
 print('parameter scales to mean |Δaction|=.002 at gradient anchors:',scales,flush=True)
 out={'settings':{'checkpoint':3380,'gradient_seed':123,'num_envs':32,'horizon':16,'obs_noise':False,'action_noise':False,'fixed_model_head_critic':True,'target_mean_initial_action_change':.002},'gradient_norms':norms,'gradient_cosines':cos,'baseline_imag_means':{k:v.mean().item() for k,v in vals.items()},'scales':scales,'tests':{}}
 del vals,gs
 for seed,mode in [(123,'rsi'),(456,'rsi'),(123,'start')]:
  key=f'{seed}_{mode}';out['tests'][key]={}
  for name,sign in [('base',0)]+[(n,s) for n in ('reward','termination','critic','total') for s in (-1,1)]:
   restore();init=reset(seed,mode)
   if sign:install(dirs[name],sign*scales[name])
   with torch.no_grad():
    at=actor(policy_input(env,init[0],init[1],init[2],init[3]));restore_action=None
    imag=imagine(init)
    result=real(init)
   out['tests'][key][f'{name}:{sign}']={'initial_frames':init[2].cpu().tolist(),'imag':{k:v.cpu().tolist() for k,v in imag.items()},'real':result,'initial_action':at.cpu().tolist()}
   print('DONE',key,name,sign,'realmean',round(sum(result['return'])/len(result['return']),5),'failed',sum(result['failure']),'imagtotal',round(imag['total'].mean().item(),5),flush=True)
 restore()
 Path('/tmp/go2_gradient_direction_results.json').write_text(json.dumps(out))
if __name__=='__main__':main()
