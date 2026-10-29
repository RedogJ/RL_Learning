"""在真实 Isaac Lab 仿真中播放世界模型训练保存的策略。"""
import argparse, json, time
from pathlib import Path
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,help='SuperMimic 训练产生的 latest.pt')
    p.add_argument('--teacher',action='store_true',help='以同样环境播放原始专家，作为对照')
    p.add_argument('--steps',type=int,default=0,help='控制步数；0 为一直播放')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--randomization',action=argparse.BooleanOptionalAction,default=False)
    add_launcher_args(p);args=p.parse_args()
    if args.steps<0:p.error('steps 必须非负')
    torch.manual_seed(args.seed)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    teacher=Path(state['teacher']); cfg=state.get('task_config') or json.loads(teacher.with_name('run.json').read_text())
    task=task_from_snapshot(cfg)
    sim_cfg=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(sim_cfg,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim=SimulationContext(sim_cfg);mouse=None;display=None
        try:
            env=make_env(sim,1,task.cfg.randomization if args.randomization else None,args.seed,task=task)
            if list(env.robot.joint_names)!=cfg['joint_names']:raise ValueError('关节顺序不匹配')
            for k,v in [('default_joint_pos',env.default_joint_pos[0]),('action_joint_limits',env.action_joint_limits[0])]:
                if not torch.allclose(torch.tensor(cfg[k],device=env.device),v,atol=1e-6,rtol=0):raise ValueError(f'{k} 不匹配')
            env.failure_termination_enabled=False
            env.finish_initial_settling()
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            if 'teacher_actor' not in state:runner.load(str(teacher),map_location=str(sim.device))
            actor=runner.get_inference_policy(device=str(sim.device))
            if args.teacher and 'teacher_actor' in state:
                actor.load_state_dict(state['teacher_actor'])
            elif not args.teacher:
                actor.load_state_dict(state['actor'])
            from go2_learning.ui.height_display import attach_height_display
            from go2_learning.ui.mouse import attach_mouse_push
            display=attach_height_display(env);mouse=attach_mouse_push(env)
            if mouse is not None:env.before_physics_step=mouse.apply;env.on_reset=mouse.clear
            obs=env.get_observations(); step=0
            print('播放原始专家' if args.teacher else '播放 SuperMimic 实验策略',flush=True)
            print('仅超时重置，Ctrl+C 退出。',flush=True)
            with torch.no_grad():
                while sim.is_headless_or_exist_active_visualizer() and (args.steps==0 or step<args.steps):
                    start=time.perf_counter();active=env.policy_active;action=env.standing_actions.clone()
                    if active.any():action[active]=actor(obs[active])
                    obs,_,_,_=env.step(action,evaluate=False);step+=1
                    if 'none' not in (args.visualizer or []):time.sleep(max(0.,env.dt-(time.perf_counter()-start)))
            print(f'播放完成 {step} 步',flush=True)
        finally:
            if display is not None:display.close()
            if mouse is not None:mouse.close()
            SimulationContext.clear_instance()

if __name__=='__main__':main()
