"""验证随机 context 真正写入 Isaac，包含单环境 reset 的隔离。"""
import argparse,json
from pathlib import Path
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot
from go2_learning.tasks.upright_supermimic.paper_context import PaperContext, configure_ground, configure_paper_physics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    add_launcher_args(p);args=p.parse_args()
    cfg=json.loads(Path(args.checkpoint).with_name('run.json').read_text())
    task=task_from_snapshot(cfg);configure_paper_physics(task);configure_ground(task)
    sc=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(sc,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim=SimulationContext(sc)
        try:
            env=make_env(sim,4,task.cfg.randomization,42,task=task)
            context=PaperContext(env,123); first=context.value.clone()
            result=context.verify();context.reset(torch.tensor([1],device=env.device))
            second=context.verify();same=torch.equal(first[[0,2,3]],context.value[[0,2,3]])
            changed=not torch.equal(first[1],context.value[1])
            env.finish_initial_settling()
            env._set_action(env.standing_actions)
            actual_target=env.robot.actuators.target_command.position.torch
            expected=env.action_center+env.action_half_range*env.actions+context.value[:,1:13]
            target_error=float((actual_target-expected).abs().max())
            for _ in range(4):env.step(env.standing_actions)
            print('CONTEXT_PROBE',json.dumps({'initial':result,'after_reset':second,'other_envs_unchanged':same,
                                             'reset_env_changed':changed,'target_calibration_max_error':target_error}),flush=True)
            assert same and changed and target_error<1e-5 and max(result['friction_max_abs_error'],result['com_max_abs_error_m'],
                                           second['friction_max_abs_error'],second['com_max_abs_error_m']) < 1e-5
        finally:SimulationContext.clear_instance()


if __name__=='__main__':main()
