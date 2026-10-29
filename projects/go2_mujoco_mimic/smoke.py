"""GPU 并行环境冒烟测试；不会训练或写入旧任务。"""
import argparse
from pathlib import Path
import time
import torch
from .warp_env import EnvCfg, WarpGo2Env
from .tasks.mimic.networks import MotionPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--motion', type=Path, default=Path('/workspace/go2_mujoco_mimic/data/kine2go/ai4_dog_canter/motion.npy'))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps 必须大于零')
    env = WarpGo2Env(EnvCfg(num_envs=args.num_envs), args.motion)
    policy = MotionPolicy(history_length=env.cfg.task.history_length, future_length=env.cfg.task.future_length).to(env.device)
    with torch.no_grad():
        check_action = policy(env.observation())
    assert check_action.shape == (args.num_envs, 12) and torch.isfinite(check_action).all()
    assert torch.allclose(env.qvel, env.ref.velocity[0].expand_as(env.qvel))
    print('目标编码器与策略解码器前向通过；reset 参考速度已写入。', flush=True)
    print(f'环境={args.num_envs} 观测={env.observation().shape[-1]} 动作=12 参考帧={len(env.ref.frames)} 控制频率={env.cfg.control_hz}Hz', flush=True)
    start = time.perf_counter()
    failures = torch.zeros((), device=env.device, dtype=torch.long)
    truncations = torch.zeros_like(failures)
    for step in range(args.steps):
        action = torch.zeros((args.num_envs, 12), device=env.device)
        obs, reward, terminated, truncated, terms = env.step(action)
        if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
            raise RuntimeError('出现非有限观测或奖励')
        failures += terminated.sum()
        truncations += truncated.sum()
        if step % 20 == 0 or step == args.steps - 1:
            print(f'步={step+1} reward={reward.mean().item():.4f} 失败={terminated.sum().item()} 截断={truncated.sum().item()} 高度={env.qpos[:,2].mean().item():.3f}m', flush=True)
        env.reset(terminated | truncated)
    torch.cuda.synchronize()
    print(f'完成，用时 {time.perf_counter()-start:.2f}s；累计失败={failures.item()} 截断={truncations.item()}。零动作使用默认关节目标，不是训练。', flush=True)


if __name__ == '__main__':
    main()
