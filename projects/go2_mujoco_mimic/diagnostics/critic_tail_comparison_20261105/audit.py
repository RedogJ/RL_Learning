import json,torch,math
from pathlib import Path
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector
@torch.no_grad()
def main():
 root=Path('/workspace/go2_mujoco_mimic/logs');paths={'baseline':root/'backflip_critic_mask_test_2026-11-05_07-17-43_496089/model_3380.pt','tail1':root/'backflip_tail_1.0_2026-11-05_07-38-55_280798/model_3400.pt','tail05':root/'backflip_tail_0.5_2026-11-05_07-41-15_227013/model_3400.pt'}
 ck0=torch.load(paths['baseline'],map_location='cpu',weights_only=False);cfg=load_config(ck0);cfg.num_envs=64;cfg.task.random_reference_start=False;cfg.task.observation_noise=True
 env=WarpGo2Env(cfg,Path(ck0['motion']));env.enable_cuda_graph();actor=MotionPolicy(history_length=cfg.task.history_length,future_length=cfg.task.future_length,heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval();out={}
 for seed in (121,988):
  for name,p in paths.items():
   ck=torch.load(p,map_location='cpu',weights_only=False);actor.load_state_dict(ck['actor']);torch.manual_seed(seed);env.reset(torch.ones(cfg.num_envs,dtype=torch.bool,device='cuda'))
   alive=torch.ones(cfg.num_envs,dtype=torch.bool,device='cuda');failure=torch.zeros_like(alive);ret=torch.zeros(cfg.num_envs,device='cuda');duration=torch.zeros_like(ret);height_peak=torch.zeros_like(ret);possum=torch.zeros_like(ret);anglesum=torch.zeros_like(ret);count=torch.zeros_like(ret)
   landerr=torch.zeros_like(ret);landn=torch.zeros_like(ret);fallheight=torch.zeros_like(ret);falltilt=torch.zeros_like(ret)
   for t in range(60):
    a=actor(env.observation());_,r,f,tr,_=env.step(a);ref=env.ref.sample(env.frame);zerr=(env.qpos[:,2]-ref['pos'][:,2]).abs();pos=(env.qpos[:,:3]-ref['pos']).norm(dim=-1);ang=rotation_vector(env.qpos[:,3:7],ref['quat']).norm(dim=-1)
    ret+=alive*r;duration+=alive*env.dt;possum+=alive*pos;anglesum+=alive*ang;count+=alive;height_peak=torch.maximum(height_peak,torch.where(alive,zerr,0));land=alive&(env.frame>=30);landerr+=land*zerr;landn+=land
    newfail=alive&f;fallheight+=newfail*(zerr>cfg.task.max_height_error);falltilt+=newfail*(zerr<=cfg.task.max_height_error);failure|=newfail;alive &= ~(f|tr)
    if not alive.any():break
   out[f'{seed}:{name}']={'return':ret.cpu().tolist(),'duration':duration.cpu().tolist(),'failure':failure.cpu().tolist(),'peak_height_error':height_peak.cpu().tolist(),'mean_position_error':(possum/count.clamp_min(1)).cpu().tolist(),'mean_orientation_deg':torch.rad2deg(anglesum/count.clamp_min(1)).cpu().tolist(),'landing_height_error':(landerr/landn.clamp_min(1)).cpu().tolist(),'landing_count':landn.cpu().tolist(),'height_fail_count':int(fallheight.sum()),'tilt_fail_count':int(falltilt.sum())}
   print(seed,name,'complete',float((~failure).float().mean()),'return',float(ret.mean()),'pos',float((possum/count.clamp_min(1)).mean()),'land',float((landerr/landn.clamp_min(1)).mean()),flush=True)
 Path('/tmp/go2_tail_compare.json').write_text(json.dumps(out))
if __name__=='__main__':main()
