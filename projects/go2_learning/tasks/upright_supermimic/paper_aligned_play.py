"""在 50 Hz Isaac 和论文物理随机化下播放独立实验策略。"""
import argparse,time
from copy import deepcopy
import torch
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot
from go2_learning.tasks.upright_supermimic.paper_context import PaperContext,configure_ground,configure_paper_physics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--teacher',action='store_true');p.add_argument('--steps',type=int,default=0)
    p.add_argument('--seed',type=int,default=123);add_launcher_args(p);args=p.parse_args()
    if args.steps<0:p.error('steps 不能为负')
    torch.manual_seed(args.seed);saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved.get('world_model_format')!='paper_context_v3' or abs(saved.get('control_dt',-1)-.02)>1e-9:
        raise ValueError('此入口仅支持 50 Hz 实际执行动作版本的论文实验检查点')
    task=task_from_snapshot(deepcopy(saved['task_config']));configure_paper_physics(task);configure_ground(task)
    cfg=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(cfg,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        from go2_learning.ui.height_display import attach_height_display
        from go2_learning.ui.mouse import attach_mouse_push
        sim=SimulationContext(cfg);display=None;mouse=None
        try:
            env=make_env(sim,1,task.cfg.randomization,args.seed,task=task)
            context=PaperContext(env,args.seed+101);env.failure_termination_enabled=False
            env.finish_initial_settling();display=attach_height_display(env);mouse=attach_mouse_push(env)
            if mouse is not None:env.before_physics_step=mouse.apply;env.on_reset=mouse.clear
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            actor=runner.get_inference_policy(device=str(sim.device))
            actor.load_state_dict(saved['teacher_actor'] if args.teacher else saved['actor'])
            obs=env.get_observations();step=0
            print('播放专家' if args.teacher else '播放论文世界模型训练策略',
                  '；50 Hz；随机摩擦、质心、关节标定；仅超时重置。',flush=True)
            with torch.no_grad():
                while sim.is_headless_or_exist_active_visualizer() and (args.steps==0 or step<args.steps):
                    start=time.perf_counter();active=env.policy_active;actions=env.standing_actions.clone()
                    if active.any():actions[active]=actor(obs[active])
                    obs,_,_,_=env.step(actions,evaluate=False);step+=1
                    if 'none' not in (args.visualizer or []):time.sleep(max(0.,env.dt-(time.perf_counter()-start)))
        finally:
            if display is not None:display.close()
            if mouse is not None:mouse.close()
            SimulationContext.clear_instance()


if __name__=='__main__':main()
