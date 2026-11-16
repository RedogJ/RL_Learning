"""原生 MuJoCo 策略回放/真实物理评估；不使用世界模型代替物理。"""
import argparse
import json
import hashlib
import time
import torch
from .config import from_dict
from .environment import Environment
from .kinematics import Kinematics
from .models import Actor
from .tasks.standing.task import StandingTask
from .train import resolve_checkpoint
from .build_model import OUTPUT,JOINTS
from .mouse import MouseInteraction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='latest')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--seconds',type=float,default=10)
    parser.add_argument('--output',help='保存评估 JSON')
    parser.add_argument('--random-reset',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    path=resolve_checkpoint(args.checkpoint)
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp.get('format')!='middle_supermimic_v1' or cp['joint_names']!=list(JOINTS):
        raise ValueError('模型类型或关节顺序不匹配')
    digest=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    if cp.get('model_sha256',digest)!=digest:raise ValueError('当前 MJCF 与检查点不一致')
    cfg=from_dict(cp['config']);cfg.env.num_envs=1;cfg.env.cuda_graph=False
    env=Environment(cfg.env,backend='cpu',device='cpu')
    if not torch.allclose(cp['reference'].cpu(),env.reference,atol=1e-6):
        raise ValueError('当前模型参考站姿与检查点不一致')
    actor=Actor(cfg).eval();actor.load_state_dict(cp['actor'])
    task=StandingTask(cfg.env,cfg.reward,Kinematics(env.model),env.reference,env.limits,env.context)
    print(f'播放：{path}\ncycle={cp["cycle"]}；真实 MuJoCo；确定性 Actor，无探索噪声',flush=True)
    mask=torch.ones(1,dtype=torch.bool)
    env.reset(mask,randomize=args.random_reset)
    reward_sum=0.0;fallen=0;elapsed=0.0;ended_returns=[];ended_times=[]
    heights=[];orientation=[]
    viewer=None
    mouse=None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer=mj_viewer.launch_passive(env.model,env.cpu_data[0])
        viewer.cam.lookat[:]=(0,-0.05,0.1);viewer.cam.distance=1.8
        viewer.cam.azimuth=130;viewer.cam.elevation=-20
        viewer.opt.geomgroup[3]=0;viewer.opt.sitegroup[4]=0
        mouse=MouseInteraction(viewer,env.model,env.cpu_data[0])
    try:
        with torch.no_grad():
            total_steps=0
            while viewer.is_running() if viewer else total_steps<round(args.seconds/env.dt):
                begin=time.monotonic()
                if mouse:mouse.sync()
                state=env.state();previous_action=env.last_action.clone()
                action=actor(env.history,task.goal(state))
                next_state=env.step(action)
                reward=sum(task.reward(state,next_state,action,previous_action).values()).item()
                reward_sum+=reward;elapsed+=env.dt;total_steps+=1
                failed=bool(task.termination(next_state).item())
                timeout=bool((env.steps>=round(cfg.env.episode_seconds*cfg.env.control_hz)).item())
                heights.append(float(next_state[0,2]));orientation.append(float(abs(next_state[0,3])))
                if failed or timeout:
                    fallen+=int(failed);ended_returns.append(reward_sum);ended_times.append(elapsed)
                    print(f'回合 return={reward_sum:.3f} time={elapsed:.2f}s fallen={failed}',flush=True)
                    reward_sum=0.0;elapsed=0.0
                    env.reset(mask,randomize=args.random_reset)
                if viewer:
                    time.sleep(max(0,env.dt-(time.monotonic()-begin)))
    finally:
        if viewer:viewer.close()
    result={'checkpoint':str(path),'cycle':cp['cycle'],'fallen':fallen,'simulated_seconds':total_steps*env.dt,
            'completed_episodes':len(ended_returns),'completed_returns':ended_returns,
            'completed_seconds':ended_times,'unfinished_return':reward_sum,'unfinished_seconds':elapsed,
            'height_min':min(heights) if heights else None,'height_max':max(heights) if heights else None}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('回放已退出。')
