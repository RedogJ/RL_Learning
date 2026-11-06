"""Replay the first recorded bad WM step and interpolate its actual Adam update."""
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics,TerminationHead
from go2_mujoco_mimic.tasks.mimic.state_train import goal
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector

root=Path('/workspace/go2_mujoco_mimic')
parser=argparse.ArgumentParser()
parser.add_argument('--experiment',type=Path,default=root/'diagnostics/wm_update_isolation_20261105')
out=parser.parse_args().experiment
config=json.loads((out/'config.json').read_text())
ck=torch.load(config['source_checkpoint'],map_location='cpu',weights_only=False)
torch.backends.cuda.matmul.allow_tf32=True
cfg=load_config(ck);cfg.num_envs=1
env=WarpGo2Env(cfg,Path(ck['motion']))
m=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda();m.load_state_dict(ck['model'])
h=TerminationHead(cfg.task.future_length*19).cuda();h.load_state_dict(ck['termination_head'])
om=torch.optim.Adam(m.parameters(),lr=2e-4);om.load_state_dict(ck['opt_m'])
od=torch.optim.Adam(h.parameters(),lr=2e-4);od.load_state_dict(ck['opt_d'])
parameters=list(m.parameters())+list(h.parameters())
source=torch.load(config.get('fixed_stream_source',str(out/'fixed_stream.pt')),map_location='cpu',weights_only=False)
data={k:v.cuda() for k,v in source['fields'].items()};context=source['context'].cuda();del source
schedule=torch.load(out/'sample_schedule.pt',map_location='cuda',weights_only=False)
probes={}
for name,path in [('old_policy_3270',config.get('old_data',str(root/'diagnostics/wm_retention_20261105/old_policy_3270_trajectories.pt'))),('recent_policy_3330',str(root/'diagnostics/wm_fixed_data_20261105/fixed_trajectories.pt'))]:
 s=torch.load(path,map_location='cpu',weights_only=False)
 probes[name]={k:s[k].cuda() for k in ('states','actions','dones','context')}
indices=torch.load(root/'diagnostics/wm_retention_20261105/probe_indices.pt',map_location='cuda',weights_only=False)

def get_row(i):
 ti,ei=schedule[i].unbind(-1)
 slots=ti[:,None]+torch.arange(8,device='cuda')[None];ids=ei[:,None].expand_as(slots)
 row={k:v[slots,ids] for k,v in data.items()};row['context']=context[ei]
 row['goal']=goal(env,row['next_frame'].reshape(-1),row['frame_limit'].reshape(-1)).reshape(len(ti),8,-1)
 return row

def loss(row,amp=True):
 pred=row['state'][:,0];live=torch.ones(len(pred),device='cuda');lm=pred.new_zeros(());ld=lm.clone()
 for t in range(8):
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):pred=m(pred,row['action'][:,t],row['context'])
  truth=row['next_state'][:,t].clone()
  truth[:,3:7]=torch.where((pred[:,3:7]*truth[:,3:7]).sum(-1,keepdim=True)<0,-truth[:,3:7],truth[:,3:7])
  step=((pred-truth)/m.normalizer.scale).square().mean(-1)
  lm+=(step*live).sum()/live.sum().clamp_min(1)
  bce=F.binary_cross_entropy_with_logits(h(m.normalizer(pred),row['goal'][:,t]),row['failure'][:,t],reduction='none')
  ld+=(bce*live).sum()/live.sum().clamp_min(1);live=live*(1-row['done'][:,t])
 return lm/8,ld/8

@torch.no_grad()
def physical():
 values={}
 for name,d in probes.items():
  ti,ei=indices[name]['16'];pred=d['states'][ti,ei];live=torch.ones(len(ti),device='cuda',dtype=torch.bool)
  for t in range(16):
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=m(pred,d['actions'][ti+t,ei],d['context'][ei])
   if t<15:live &= ~d['dones'][ti+t,ei]
  error=(pred-d['states'][ti+16,ei])[live]
  angle=rotation_vector(pred[live,3:7],d['states'][ti+16,ei][live,3:7]).norm(dim=-1)
  values[name]={'root_position_rms_cm':error[:,:3].square().sum(-1).mean().sqrt().item()*100,
    'root_rotation_rms_deg':torch.rad2deg(angle.square().mean().sqrt()).item(),
    'joint_position_rmse_deg':math.degrees(error[:,7:19].square().mean().sqrt().item())}
 return values

for i in range(20):
 row=get_row(i);lm,ld=loss(row)
 if i==19:
  before=[p.detach().clone() for p in parameters]
  measured_before=[lm.item(),ld.item()]
 om.zero_grad(set_to_none=True);od.zero_grad(set_to_none=True);(lm+ld).backward()
 torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);om.step();od.step()
after=[p.detach().clone() for p in parameters]
expected=next(x for x in map(json.loads,(out/'gradients.jsonl').read_text().splitlines()) if x['cycle']==0 and x['update']==20)
with torch.no_grad():
 lm,ld=loss(row);measured_after=[lm.item(),ld.item()]
print('REPLAY CHECK', measured_before, measured_after, 'original', expected['state_loss_before'], expected['state_loss_after'],flush=True)
result={'diagnostic_update':'first cycle, update 20','expected_original':expected,
        'replayed_before':measured_before,'replayed_after':measured_after,
        'exact_loss_match':abs(measured_before[0]-expected['state_loss_before'])<1e-5 and abs(measured_after[0]-expected['state_loss_after'])<1e-5,
        'description':'interpolate both WM and head actual Adam parameter changes. Same-batch MSE/BCE measured with BF16 and with float32 matmul (TF32 allowed). Probes use BF16 as original.', 'fractions':[]}
with torch.no_grad():
 for fraction in (0.,.1,.25,.5,.75,1.):
  for p,b,a in zip(parameters,before,after):p.copy_(b if fraction==0 else a if fraction==1 else b+(a-b)*fraction)
  bl,bd=loss(row,True);fl,fd=loss(row,False)
  entry={'fraction':fraction,'bf16_state_loss':bl.item(),'bf16_bce':bd.item(),
         'float32_state_loss':fl.item(),'float32_bce':fd.item(),'physical16':physical()}
  result['fractions'].append(entry);print('STEP FRACTION',json.dumps(entry),flush=True)
(out/'step_size_check.json').write_text(json.dumps(result,indent=2));print('DONE',out/'step_size_check.json',flush=True)
