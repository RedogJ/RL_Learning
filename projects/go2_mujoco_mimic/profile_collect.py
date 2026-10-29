"""只运行采集性能诊断，不训练或保存模型。"""
import time
import os
from pathlib import Path
import torch
from .warp_env import WarpGo2Env, EnvCfg
from .tasks.mimic.networks import MotionPolicy


def main():
    env = WarpGo2Env(EnvCfg(num_envs=64), Path('/workspace/go2_mujoco_mimic/data/kine2go/ai4_dog_canter/motion.npy'))
    if os.environ.get('GO2_PHYSICS_GRAPH') == '1':
        env.enable_cuda_graph()
    policy = MotionPolicy().to(env.device).eval()
    obs = env.observation()
    timings = {k: 0.0 for k in ('policy', 'reset', 'post_reset_observation')}
    env.profile = {}
    for step in range(48):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad(): action = policy(obs)
        torch.cuda.synchronize(); timings['policy'] += time.perf_counter()-t
        with torch.no_grad(): obs, _, terminated, truncated, _ = env.step(action)
        torch.cuda.synchronize(); t = time.perf_counter()
        env.reset(terminated | truncated)
        torch.cuda.synchronize(); timings['reset'] += time.perf_counter()-t
        t = time.perf_counter(); obs = env.observation()
        torch.cuda.synchronize(); timings['post_reset_observation'] += time.perf_counter()-t
        if step == 15:
            env.profile = {}; timings = dict.fromkeys(timings, 0.0)
    print('采集细分：32 控制步（同步计时；有测量开销）', flush=True)
    for key, value in {**env.profile, **timings}.items(): print(f'{key}={value:.4f}s', flush=True)


if __name__ == '__main__': main()
