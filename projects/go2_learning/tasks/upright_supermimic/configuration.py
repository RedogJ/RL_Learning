"""恢复本实验任务快照，避免运行时依赖原任务后续调参。"""
from dataclasses import asdict
from go2_learning.tasks.upright.task import UprightTask
from go2_learning.tasks.upright.randomization_cfg import RandomizationCfg
from go2_learning.tasks.upright.config import TerminationCfg
from go2_learning.tasks.upright.reset_cfg import ResetCfg


def task_from_snapshot(cfg):
    task=UprightTask()
    task.load_playback_config(cfg)
    # 普通 upright 播放会按当前代码覆盖终止和后腿预载；本实验需要原实验快照。
    if cfg.get('termination_cfg') is not None:
        task.cfg.termination=TerminationCfg(**cfg['termination_cfg'])
    if cfg.get('reset_cfg') is not None:
        task.cfg.reset=ResetCfg(**cfg['reset_cfg'])
    task.cfg.lock_front_legs=cfg.get('front_legs_fixed',False)
    if 'front_lock_solver_iterations' in cfg:
        task.cfg.front_lock_position_iterations,task.cfg.front_lock_velocity_iterations=cfg['front_lock_solver_iterations']
    if cfg.get('randomization_cfg') is not None:
        task.cfg.randomization=RandomizationCfg(**cfg['randomization_cfg'])
    else:
        # 专家若没开随机化，固定当前默认范围于本实验快照，供未来启用。
        cfg['randomization_cfg']=asdict(task.cfg.randomization)
    return task
