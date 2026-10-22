"""任务注册表；具体任务按需加载，导入配置不启动仿真器。"""

from importlib import import_module

TASKS = {
    "standing": "go2_learning.tasks.standing.task:StandingTask",
    "running": "go2_learning.tasks.running.task:RunningTask",
    "upright": "go2_learning.tasks.upright.task:UprightTask",
}


def get_task(name="standing"):
    if name not in TASKS:
        raise ValueError(f"未知任务 {name!r}，可用任务：{', '.join(TASKS)}")
    module, class_name = TASKS[name].split(":")
    return getattr(import_module(module), class_name)()


def checkpoint_task_name(run_cfg):
    if "task_name" in run_cfg:
        return run_cfg["task_name"]
    from .standing import TASK_VERSION

    return "standing" if run_cfg.get("task_version") == TASK_VERSION else None
