"""Frozen policy playback and step-by-step true simulator reward audit."""
import json
from pathlib import Path
import mujoco
import numpy as np
import torch

from go2_mujoco_mimic.environment import Go2Environment, JOINT_NAMES
from go2_mujoco_mimic.play_policy import load_config, update_reference
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector
from go2_mujoco_mimic.warp_env import WarpGo2Env

ROOT=Path('/workspace/go2_mujoco_mimic/logs')
CASES={
 '3500':ROOT/'backflip_critic_history_time_2026-11-05_08-41-55_084475/model_3500.pt',
 '4050':ROOT/'backflip_v21_2026-11-05_09-22-06_245558/model_4050.pt',
 '4100':ROOT/'backflip_v21_2026-11-05_09-22-06_245558/model_4100.pt',
}

@torch.no_grad()
def main():
 torch.manual_seed(123)
 base=torch.load(CASES['4050'],map_location='cpu',weights_only=False)
 cfg=load_config(base); cfg.num_envs=1; cfg.task.random_reference_start=False; cfg.task.observation_noise=False
 env=WarpGo2Env(cfg,Path(base['motion'])); env.enable_cuda_graph()
 actor=MotionPolicy(history_length=cfg.task.history_length,future_length=cfg.task.future_length,
   heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
 cpu=Go2Environment(cfg.physics_hz,cfg.control_hz,checkerboard=True)
 cpu.model.vis.global_.offwidth=960;cpu.model.vis.global_.offheight=540
 ghost=mujoco.MjData(cpu.model)
 ground=mujoco.mj_name2id(cpu.model,mujoco.mjtObj.mjOBJ_GEOM,'ground')
 feet={mujoco.mj_name2id(cpu.model,mujoco.mjtObj.mjOBJ_GEOM,leg):leg for leg in ('FL','FR','RL','RR')}
 def contact(d):
  result=[]
  for c in d.contact[:d.ncon]:
   a,b=int(c.geom1),int(c.geom2)
   if ground not in (a,b):continue
   other=b if a==ground else a
   name=feet.get(other)
   if name and name not in result:result.append(name)
  return sorted(result)
 render=mujoco.Renderer(cpu.model,height=540,width=960)
 camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(cpu.model,camera)
 camera.distance=2.7;camera.azimuth=135;camera.elevation=-20
 opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[2]=1
 raw=open('/output/backflip_4050.rgb','wb')
 out={'checkpoint':{k:str(v) for k,v in CASES.items()},'dt':env.dt,'joint_names':list(JOINT_NAMES),
      'noise':False,'reference':'single backflip + held final pose','records':{}}
 for label,checkpoint in CASES.items():
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False);actor.load_state_dict(ck['actor'])
  torch.manual_seed(123);env.reset(torch.ones(1,dtype=torch.bool,device='cuda'))
  rows=[]
  for step in range(65):
   act=actor(env.observation())
   _,reward,failed,truncated,terms=env.step(act)
   ref=env.ref.sample(env.frame)
   cpu.data.qpos[:]=env.qpos[0].cpu().numpy();cpu.data.qvel[:]=env.qvel[0].cpu().numpy();mujoco.mj_forward(cpu.model,cpu.data)
   frame=int(env.frame[0]);update_reference(cpu,ghost,env.ref,frame)
   q=env.qpos[0,env.qids];jref=ref['joints'][0];jerr=q-jref
   rot=rotation_vector(env.qpos[:,3:7],ref['quat']).norm(dim=-1)[0]
   row={'step':step+1,'frame':frame,'time_s':(step+1)*env.dt,'reward':float(reward[0]),
      'terms_rate':{k:float(v[0]) for k,v in terms.items()},'failed':bool(failed[0]),'truncated':bool(truncated[0]),
      'root_height':float(env.qpos[0,2]),'reference_height':float(ref['pos'][0,2]),
      'root_position_error_m':float((env.qpos[0,:3]-ref['pos'][0]).norm()),
      'root_orientation_error_deg':float(torch.rad2deg(rot)),
      'joint_error_deg':torch.rad2deg(jerr).cpu().tolist(),
      'joint_rms_deg':float(torch.rad2deg(jerr.square().mean().sqrt())),
      'front_joint_rms_deg':float(torch.rad2deg(jerr[:6].square().mean().sqrt())),
      'rear_joint_rms_deg':float(torch.rad2deg(jerr[6:].square().mean().sqrt())),
      'foot_contacts':contact(cpu.data),'reference_foot_contacts':contact(ghost),
      'reference_foot_bottom_m':{name:float(ghost.geom_xpos[idx,2]-cpu.model.geom_size[idx,0]) for idx,name in feet.items()},
      'actual_foot_bottom_m':{name:float(cpu.data.geom_xpos[idx,2]-cpu.model.geom_size[idx,0]) for idx,name in feet.items()},
      'nonfoot_contact_during_step':bool(env.contact_fall[0]),'root_velocity':env.qvel[0,:6].cpu().tolist()}
   rows.append(row)
   if label=='4050':
    camera.lookat[:]=cpu.data.qpos[:3]+np.array([0.,.30,0.])
    render.update_scene(cpu.data,camera=camera)
    start=render.scene.ngeom
    mujoco.mjv_addGeoms(cpu.model,ghost,opt,mujoco.MjvPerturb(),mujoco.mjtCatBit.mjCAT_ALL,render.scene)
    for i in range(start,render.scene.ngeom):
     geom=render.scene.geoms[i];geom.pos[1]+=.6;geom.matid=-1;geom.rgba[:]=(.08,.45,1.,.30)
    rgb=render.render().copy();raw.write(rgb.tobytes())
   if bool(failed[0] or truncated[0]): break
  if label=='4050':
   for _ in range(8):raw.write(rgb.tobytes())
  out['records'][label]=rows
  print(label,'steps',len(rows),'failed',rows[-1]['failed'],'last heights',rows[-1]['root_height'],rows[-1]['reference_height'],flush=True)
 raw.close();render.close()
 Path('/output/landing_reward_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
