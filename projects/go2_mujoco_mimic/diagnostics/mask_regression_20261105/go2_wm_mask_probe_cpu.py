import torch,math,json
from pathlib import Path
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 root=Path('/workspace/go2_mujoco_mimic');data=torch.load(root/'diagnostics/wm_retention_20261105/old_policy_3270_trajectories.pt',map_location='cpu',weights_only=False);inds=torch.load(root/'diagnostics/wm_retention_20261105/probe_indices.pt',map_location='cpu',weights_only=False)['old_policy_3270']['16'];ti,ei=[x[:64] for x in inds]
 d={k:data[k] for k in ('states','actions','dones','context')};out={}
 for cycle in (3300,3310,3340,3350):
  ck=torch.load(root/f'logs/backflip_critic_mask_test_2026-11-05_07-17-43_496089/model_{cycle}.pt',map_location='cpu',weights_only=False);model=Dynamics().eval();model.load_state_dict(ck['model']);p=d['states'][ti,ei];alive=torch.ones(len(ei),dtype=torch.bool)
  for j in range(16):
   p=model(p,d['actions'][ti+j,ei],d['context'][ei]);
   if j<15:alive &= ~d['dones'][ti+j,ei]
  true=d['states'][ti+16,ei];error=(p-true)[alive];ang=rotation_vector(p[alive,3:7],true[alive,3:7]).norm(dim=-1)
  result={'n':int(alive.sum()),'position_cm':round(error[:,:3].norm(dim=-1).square().mean().sqrt().item()*100,3),'rotation_deg':round(torch.rad2deg(ang.square().mean().sqrt()).item(),3),'joint_deg':round(math.degrees(error[:,7:19].square().mean().sqrt().item()),3)}
  print(cycle,result,flush=True);out[str(cycle)]=result;del model,ck
 Path('/tmp/go2_wm_mask_probe_cpu.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
