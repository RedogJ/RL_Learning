"""按任务查找模型，兼容旧扁平日志和新的 logs/<task>/<run>/ 结构。"""

import json
from pathlib import Path

from go2_learning.tasks import checkpoint_task_name, get_task


def validate_checkpoint_task(run_cfg, task):
    if checkpoint_task_name(run_cfg) != task.name or run_cfg.get("task_version") != task.version:
        raise ValueError(f"检查点任务与 {task.name!r} 不一致，或任务版本不兼容")


def validate_resume_task(run_cfg, task, source_task_name=None):
    """续训可显式从 standing 转到 running；播放仍要求检查点任务完全一致。"""
    source = get_task(source_task_name or task.name)
    validate_checkpoint_task(run_cfg, source)
    if source.name != task.name and (source.name, task.name) != ("standing", "running"):
        raise ValueError(f"尚未支持从 {source.name!r} 迁移到 {task.name!r}")
    # 这两个版本共用同一 Go2 环境及 52 维观测、12 维完整关节范围动作。
    # 网络结构和运行时关节映射仍由入口逐项核对。
    return source.name


def relocate_checkpoint(value, log_root, task_name=None):
    """定位改目录前的文件；同名副本优先按来源任务消歧，禁止猜选。"""
    path, root = Path(value).expanduser(), Path(log_root)
    if not path.exists() and "logs" in path.parts:
        suffix = path.parts[path.parts.index("logs") + 1:]
        relocated = root.joinpath(*suffix)
        if relocated.exists():
            path = relocated
        else:
            # 兼容旧 logs/<run>/ → logs/<task>/<run>/。
            if task_name and suffix:
                scoped = root / task_name / Path(*suffix)
                if scoped.exists():
                    path = scoped
            if not path.exists() and len(suffix) >= 2:
                matches = list(root.rglob(f"{path.parent.name}/{path.name}"))
                if len(matches) > 1:
                    raise ValueError(f"旧模型路径存在多个副本，请指定任务及具体路径：{value}")
                if matches:
                    path = matches[0]
    if path.is_dir():
        path = path / "latest.pt"
    return path.resolve()


def _latest_candidates(log_root, task_name):
    root = Path(log_root)
    # 仅查所选任务目录和旧扁平布局，其他任务目录中的副本不参与选择。
    return list((root / task_name).glob("*/latest.pt")) + list(root.glob("*/latest.pt"))


def resolve_checkpoint(value, task_name, log_root):
    if value != "latest":
        return relocate_checkpoint(value, log_root, task_name)
    task = get_task(task_name)
    checkpoints = []
    for path in _latest_candidates(log_root, task_name):
        metadata = path.with_name("run.json")
        if not metadata.is_file():
            continue
        cfg = json.loads(metadata.read_text())
        if checkpoint_task_name(cfg) == task.name and cfg.get("task_version") == task.version:
            checkpoints.append(path)
    if not checkpoints:
        raise FileNotFoundError(f"任务 {task_name!r} 没有可用的已完成训练，请先运行 train")
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def resolve_resume_checkpoint(value, task_name, log_root, source_task_name=None):
    """优先续训本任务；尚无 running 模型时允许使用其目录里的 standing 副本。"""
    if value != "latest":
        return relocate_checkpoint(value, log_root, source_task_name or task_name)
    if source_task_name is None or source_task_name == task_name:
        try:
            return resolve_checkpoint(value, task_name, log_root)
        except FileNotFoundError:
            pass
    task = get_task(task_name)
    candidates = []
    for path in (Path(log_root) / task_name).glob("*/latest.pt"):
        metadata = path.with_name("run.json")
        if not metadata.is_file():
            continue
        cfg = json.loads(metadata.read_text())
        source = checkpoint_task_name(cfg)
        if source_task_name is not None and source != source_task_name:
            continue
        try:
            validate_resume_task(cfg, task, source)
        except ValueError:
            continue
        candidates.append(path)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)
    if source_task_name is not None:
        # 显式指定来源任务时，也支持直接从该任务目录读取。
        source = get_task(source_task_name)
        path = resolve_checkpoint("latest", source.name, log_root)
        validate_resume_task(json.loads(path.with_name("run.json").read_text()), task, source.name)
        return path
    raise FileNotFoundError(f"{Path(log_root) / task_name} 中没有可用于续训的模型")
