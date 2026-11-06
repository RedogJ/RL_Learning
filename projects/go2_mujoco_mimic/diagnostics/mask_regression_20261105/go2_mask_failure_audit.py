import json,torch
from pathlib import Path
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.rewards import gravity_local,rotation_vector
@torch.no_grad()
def main():
 folder=Path('/workspace/go2_mujoco_mimic/logs/backflip_critic_mask_test_2026-11-05_07-17-43_496089')
 first=torch.load(folder/'model_3300.pt',map_location='cpu',weights_only=False);cfg=load_config(first);cfg.num_envs=1;cfg.task.observation_noise=False;cfg.task.random_reference_start=False
 env=WarpGo2Env(cfg,Path(first['motion']));env.enable_cuda_graph();actor=MotionPolicy(history_length=cfg.task.history_length,future_length=cfg.task.future_length,heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
 out={}
 for cycle in (3290,3300,3310,3340,3350):
  ck=torch.load(folder/f'model_{cycle}.pt',map_location='cpu',weights_only=False);actor.load_state_dict(ck['actor']);torch.manual_seed(123);env.reset(torch.ones(1,dtype=torch.bool,device='cuda'))
  rows=[]
  for t in range(65):
   action=actor(env.observation());_,reward,term,trunc,_=env.step(action);ref=env.ref.sample(env.frame);q=env.qpos[:,3:7];zerr=(env.qpos[:,2]-ref['pos'][:,2]).abs();gerr=(gravity_local(q)[:,2]-gravity_local(ref['quat'])[:,2]).abs()
   jq=env.qpos[:,env.qids];jqerr=(jq-ref['joints']).square().mean().sqrt();ori=torch.rad2deg(rotation_vector(q,ref['quat']).square().sum(-1).sqrt());position=(env.qpos[:,:3]-ref['pos']).norm(dim=-1)
   rows.append({'step':t+1,'seconds':(t+1)*env.dt,'frame':int(env.frame[0]),'reward':float(reward),'height_error':float(zerr),'tilt_error':float(gerr),'height_m':float(env.qpos[0,2]),'ref_height_m':float(ref['pos'][0,2]),'orientation_error_deg':float(ori),'position_error_m':float(position),'joint_rms_rad':float(jqerr),'action':action[0].cpu().tolist(),'joint_pos':jq[0].cpu().tolist(),'terminated':bool(term),'truncated':bool(trunc)})
   if bool(term|trunc):break
  out[str(cycle)]={'cycle':cycle,'thresholds':{'height_error_m':cfg.task.max_height_error,'tilt_error':cfg.task.max_tilt_error},'rows':rows}
  print('cycle',cycle,'last',rows[-1],flush=True)
 with open('/tmp/go2_mask_failure_audit.json','w') as f:json.dump(out,f)
if __name__=='__main__':main()
