"""检查图重放与逐步物理仿真的数值一致性。"""
from pathlib import Path
import torch
from .warp_env import WarpGo2Env, EnvCfg


def main():
    torch.manual_seed(42)
    cfg = EnvCfg(num_envs=4)
    cfg.task.observation_noise = False
    path = Path('/workspace/go2_mujoco_mimic/data/kine2go/ai4_dog_canter/motion.npy')
    old = WarpGo2Env(cfg, path)
    new = WarpGo2Env(cfg, path)
    new.enable_cuda_graph()
    mx = torch.zeros(4, device=old.device)
    for i in range(20):
        action = (torch.sin(torch.arange(48,device=old.device).view(4,12)*0.3+i*0.1)*0.15)
        with torch.no_grad():
            o1,r1,t1,tr1,_ = old.step(action)
            o2,r2,t2,tr2,_ = new.step(action)
            mx = torch.maximum(mx, torch.stack(((old.qpos-new.qpos).abs().max(),
                               (old.qvel-new.qvel).abs().max(), (o1-o2).abs().max(), (r1-r2).abs().max())))
            if not torch.equal(t1,t2) or not torch.equal(tr1,tr2):
                raise RuntimeError(f'终止条件不一致，第 {i} 步')
            old.reset(t1 | tr1); new.reset(t2 | tr2)
    print('20步含重置的最大差值：qpos qvel obs reward =', mx.tolist(), flush=True)
    if mx[0] > 1e-4 or mx[1] > 1e-3 or mx[2] > 1e-3 or mx[3] > 1e-4:
        raise RuntimeError('图重放结果偏差超过容许值')


if __name__ == '__main__': main()
