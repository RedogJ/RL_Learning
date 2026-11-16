"""训练关键边界检查：复合关节 FK、零误差奖励、梯度隔离、并行环境独立性。"""
import argparse
from dataclasses import replace
import numpy as np
import mujoco
import torch

from .build_model import JOINTS
from .config import Config
from .environment import Environment
from .train import make_modules
from .learner import imagine
from .math_utils import exp_quat


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip-warp',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(7)
    cfg=Config();cfg.env.num_envs=4;cfg.model.wm_width=64
    cfg.train.horizon=3;cfg.train.critic_epochs=1;cfg.train.critic_minibatches=1
    env=Environment(cfg.env,'cpu','cpu')
    task,actor,wm,critic=make_modules(cfg,env)
    cpu=mujoco.MjData(env.model)
    state=env.reference.repeat(32,1)
    state[:,:3]=torch.randn(32,3)*0.1
    state[:,3:7]=exp_quat(torch.randn(32,3))
    state[:,7:19]=torch.rand(32,12)*2-1
    pos,quat=task.fk(state)
    for i in range(32):
        cpu.qpos[:7]=state[i,:7].numpy();cpu.qpos[env.qids_np]=state[i,7:19].numpy()
        mujoco.mj_forward(env.model,cpu)
        np.testing.assert_allclose(pos[i].numpy(),cpu.xpos[task.fk.body_ids],atol=3e-6)
        # q 与 -q 等价。
        dots=np.abs((quat[i].numpy()*cpu.xquat[task.fk.body_ids]).sum(-1))
        np.testing.assert_allclose(dots,1,atol=3e-6)
    print('PASS：多 hinge Torch FK 与 MuJoCo 32 组随机位姿一致')
    ref=env.reference.repeat(4,1).requires_grad_(True)
    action=torch.zeros(4,12,requires_grad=True)
    terms=task.reward(ref,ref,action,action)
    torch.testing.assert_close(sum(terms.values()),torch.full((4,),task.max_step_reward))
    sum(terms.values()).sum().backward()
    assert torch.isfinite(ref.grad).all() and torch.isfinite(action.grad).all()
    print('PASS：零误差奖励达到上限，姿态零误差反传无 NaN')
    env.reset(torch.ones(4,dtype=torch.bool),randomize=False)
    start={'state':env.state()[:,None,:], 'history':env.history[:,None,:,:].clone(),
           'step':env.steps[:,None].clone(),'previous_action':env.last_action[:,None,:].clone()}
    ao=torch.optim.Adam(actor.parameters(),lr=1e-4);co=torch.optim.Adam(critic.parameters(),lr=1e-3)
    actor_before=[x.clone() for x in actor.parameters()];wm_before=[x.clone() for x in wm.parameters()]
    logs=imagine(actor,critic,wm,task,start,cfg,ao,co,update_actor=True)
    assert all(torch.isfinite(x).all() for x in logs.values())
    assert any(not torch.equal(a,b) for a,b in zip(actor_before,actor.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(wm_before,wm.parameters()))
    assert all(x.grad is None for x in wm.parameters())
    print('PASS：Actor 经可微想象实际更新，世界模型参数/梯度隔离，Critic 回归有限')
    if not args.skip_warp:
        gpu=Environment(replace(cfg.env,num_envs=4),'warp','cuda:0')
        mask=torch.ones(4,device='cuda:0',dtype=torch.bool)
        gpu.reset(mask,randomize=False)
        # 两种物理步进路径必须一致，捕获不能意外覆盖目标指针。
        test_action=torch.zeros(4,12,device='cuda:0');test_action[0,0]=0.5
        with_graph=gpu.step(test_action).clone()
        graph=gpu.graph;gpu.graph=None
        gpu.reset(mask,randomize=False)
        without_graph=gpu.step(test_action).clone()
        torch.testing.assert_close(with_graph,without_graph,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(with_graph[1],with_graph[2],atol=2e-6,rtol=2e-6)
        assert not torch.allclose(with_graph[0,7:19],with_graph[1,7:19],atol=1e-6)
        gpu.graph=graph
        print('PASS：Warp Graph 与普通步进一致；单环境动作不串扰其他环境')


if __name__=='__main__':
    try:main()
    finally:
        import gc
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize();torch.cuda.empty_cache()
