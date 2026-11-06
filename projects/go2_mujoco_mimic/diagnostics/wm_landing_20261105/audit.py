"""Evaluate frozen WM on fresh physical rollouts, including impact crossing windows."""
import json, math
from pathlib import Path
import torch
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics,build_state,obs_from_state
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector
from go2_mujoco_mimic.tasks.mimic.kinematics import rotate
from go2_mujoco_mimic.tasks.mimic.state_train import policy_input

ROOT=Path('/workspace/go2_mujoco_mimic/logs/backflip_v21_2026-11-05_09-22-06_245558')
OUTPUT=Path('/output/wm_landing_results.json')
ORIGINS=(0,10,16,20,24,26,27,28,29,30,32,35,37,40,44,48,50,52)

@torch.no_grad()
def main():
 torch.backends.cuda.matmul.allow_tf32=True
 result={'description':'Fresh real Mujoco Warp trajectories, 128 environments per checkpoint; random observation noise, no action noise; all start from reference first frame. Predictions use recorded real actions (open-loop), no optimizer updates. BF16 WM as training.',
         'seed':123,'envs':128,'origins':list(ORIGINS),'models':{}}
 for cycle in (4050,4100):
  ck=torch.load(ROOT/f'model_{cycle}.pt',map_location='cpu',weights_only=False)
  cfg=load_config(ck);cfg.num_envs=128;cfg.task.random_reference_start=False;cfg.task.observation_noise=True
  env=WarpGo2Env(cfg,Path(ck['motion']));env.enable_cuda_graph()
  actor=MotionPolicy(history_length=cfg.task.history_length,future_length=cfg.task.future_length,
      heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
  actor.load_state_dict(ck['actor'])
  model=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda().eval();model.load_state_dict(ck['model'])
  torch.manual_seed(123);env.reset(torch.ones(cfg.num_envs,dtype=torch.bool,device='cuda'))
  state=[build_state(env).clone()];histories=[env.history.clone()];actions=[];valid=[];dones=[];rewards=[]
  alive=torch.ones(cfg.num_envs,dtype=torch.bool,device='cuda');failures=0;completed=0
  for t in range(53):
   valid.append(alive.clone());a=actor(env.observation());actions.append(a.clone())
   _,reward,failed,truncated,_=env.step(a)
   failures+=int((alive&failed).sum());completed+=int((alive&truncated&~failed).sum());done=failed|truncated;dones.append(done.clone());rewards.append(reward.clone())
   state.append(build_state(env).clone());histories.append(env.history.clone());alive &= ~done
  S=torch.stack(state);H=torch.stack(histories);A=torch.stack(actions);V=torch.stack(valid);D=torch.stack(dones)
  C=torch.zeros((cfg.num_envs,16),device='cuda');C[:,0]=.5
  def metrics(pred,actual,mask):
   pred,actual=pred[mask],actual[mask]
   if not len(pred):return None
   diff=pred-actual
   mean=lambda x:x.square().mean().sqrt().item()
   angles=rotation_vector(pred[:,3:7],actual[:,3:7]).norm(dim=-1)
   ppos,pquat=env.kinematics(pred);apos,aquat=env.kinematics(actual)
   calf_ids=[3,7,11,15]
   geom_offset=pred.new_tensor([-.002,0.,-.213])
   pred_feet=ppos[:,calf_ids]+rotate(pquat[:,calf_ids],geom_offset)
   true_feet=apos[:,calf_ids]+rotate(aquat[:,calf_ids],geom_offset)
   pred_height=pred_feet[:,:,2]-.022;true_height=true_feet[:,:,2]-.022
   near_pred=pred_height<=.005;near_true=true_height<=.005
   return {'n':len(pred),'foot_bottom_height_rmse_cm':100*mean(pred_height-true_height),
     'foot_near_ground_mismatch_fraction':(near_pred!=near_true).float().mean().item(),
     'root_position_euclidean_rmse_cm':100*diff[:,:3].square().sum(-1).mean().sqrt().item(),
     'height_rmse_cm':100*mean(diff[:,2]),'height_bias_cm':100*diff[:,2].mean().item(),
     'orientation_rms_deg':torch.rad2deg(angles.square().mean().sqrt()).item(),
     'joint_angle_rmse_deg':math.degrees(mean(diff[:,7:19])),
     'front_joint_rmse_deg':math.degrees(mean(diff[:,7:13])),
     'rear_joint_rmse_deg':math.degrees(mean(diff[:,13:19])),
     'linear_velocity_component_rmse_m_s':mean(diff[:,19:22]),
     'vertical_velocity_rmse_m_s':mean(diff[:,21]),
     'angular_velocity_component_rmse_rad_s':mean(diff[:,22:25]),
     'joint_velocity_component_rmse_rad_s':mean(diff[:,25:37])}
  with torch.autocast('cuda',dtype=torch.bfloat16): one=model(S[:-1].flatten(0,1),A.flatten(0,1),C.unsqueeze(0).expand(len(A),-1,-1).reshape(-1,16)).reshape_as(S[:-1])
  per_frame={str(t+1):metrics(one[t],S[t+1],V[t]) for t in range(len(A))}
  phases={}
  for name,(lo,hi) in {'launch':(1,18),'flight':(19,28),'landing':(29,34),'recovery':(35,40),'hold':(41,53)}.items():
   p=one[lo-1:hi].flatten(0,1);a=S[lo:hi+1].flatten(0,1);m=V[lo-1:hi].flatten()
   phases[name]={'frames':[lo,hi],'one_step':metrics(p,a,m)}
  windows={}
  for origin in ORIGINS:
   pred=S[origin].clone(); key={};live=V[origin].clone()
   for j in range(1,min(16,53-origin)+1):
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,A[origin+j-1],C)
    if j in (1,4,8,16):key[str(j)]=metrics(pred,S[origin+j],live)
    live &= ~D[origin+j-1]
   windows[str(origin)]=key
  # Same sensor-noise realization as the recorded rollout, isolating divergence
  # caused by using predicted states in the policy's reconstructed history.
  closed={}
  for origin in (16,20,24,28,29,32,35):
   pred=S[origin].clone();hist=H[origin].clone();live=V[origin].clone();key={}
   limit=torch.full((cfg.num_envs,),53,device='cuda',dtype=torch.long)
   for j in range(1,min(16,53-origin)+1):
    frame=torch.full_like(limit,origin+j-1)
    a=actor(policy_input(env,pred,hist,frame,limit))
    with torch.autocast('cuda',dtype=torch.bfloat16):nxt=model(pred,a,C)
    noise=H[origin+j,:,-1]-obs_from_state(S[origin+j],A[origin+j-1],env.default_q,noise=False)
    pred_obs=obs_from_state(nxt,a,env.default_q,noise=False)+noise
    hist=torch.cat((hist[:,1:],pred_obs[:,None]),1);pred=nxt
    if j in (1,4,8,16):
     key[str(j)]=metrics(pred,S[origin+j],live)
     if key[str(j)] is not None:
      key[str(j)]['action_component_rmse']=(a[live]-A[origin+j-1,live]).square().mean().sqrt().item()
    live &= ~D[origin+j-1]
   closed[str(origin)]=key
  result['models'][str(cycle)]={'failed':failures,'completed':completed,'dt':env.dt,
      'one_step_by_phase':phases,'one_step_by_frame':per_frame,'windows':windows,'closed_loop_windows':closed}
  print('MODEL',cycle,'failures',failures,'phases',json.dumps(phases),flush=True)
  del env,model,actor,ck,S,A,V,D
 OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print('RESULT',OUTPUT,flush=True)

if __name__=='__main__':main()
