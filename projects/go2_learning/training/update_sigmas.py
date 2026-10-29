"""离线将 upright 检查点中的 sigma，或重新校准的 sigma 写回当前配置。"""

import argparse
import ast
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch

from go2_learning.paths import LOG_ROOT, PROJECT_ROOT
from go2_learning.tasks import get_task
from go2_learning.tasks.upright.tracking_errors import ERROR_SPECS
from go2_learning.training.checkpoints import resolve_checkpoint, validate_checkpoint_task
from go2_learning.training.reward_calibration import estimate_sigmas, measurement_binding


def render_config(source, updates):
    """只替换 RewardCfg 的对应数值，保留注释、其他参数及格式。"""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardCfg")
    lines = source.splitlines(keepends=True)
    remaining = set(updates)
    replacements = []
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if name not in updates:
            continue
        value = float(updates[name])
        if not name.endswith("_sigma") or not math.isfinite(value) or value <= 0:
            raise ValueError(f"无效 sigma：{name}={value}")
        expr = node.value
        if expr is None or expr.lineno != expr.end_lineno:
            raise ValueError(f"{name} 必须为单行赋值")
        # AST 列号是 UTF-8 字节偏移，不能当作 Unicode 字符索引。
        line = lines[expr.lineno - 1].encode("utf-8")
        replacements.append((expr.lineno - 1,
                             line[:expr.col_offset] + repr(value).encode() + line[expr.end_col_offset:]))
        remaining.remove(name)
    if remaining:
        raise ValueError(f"reward_cfg.py 缺少字段：{sorted(remaining)}")
    for index, line in replacements:
        lines[index] = line.decode("utf-8")
    result = "".join(lines)
    ast.parse(result)
    return result


def checkpoint_sigmas(checkpoint, recalibrate=False):
    task = get_task("upright")
    metadata = json.loads(checkpoint.with_name("run.json").read_text())
    validate_checkpoint_task(metadata, task)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    infos = payload.get("infos") or {}
    # model_N.pt 必须使用该检查点的参数；run.json 只作为旧模型的兼容回退。
    saved = infos.get("reward_cfg") or metadata["reward_cfg"]
    names = [name for name, (_, weight, _) in ERROR_SPECS.items()
             if getattr(task.cfg.rewards, weight) != 0]
    fields = [ERROR_SPECS[name][0] for name in names]
    if not recalibrate:
        return {name: saved[name] for name in fields}

    snapshot = infos.get("reward_error_buffer") or {}
    if snapshot.get("schema_version") != 1:
        raise ValueError("该检查点没有近期误差缓冲，不能离线重新校准；可去掉 --recalibrate 同步已用数值")
    binding = measurement_binding(task, metadata.get("randomization_cfg") is not None)
    if snapshot.get("binding") != binding:
        raise ValueError("当前目标或物理/重置配置与误差缓冲不匹配，需要先按当前配置采集训练数据")
    stored_names = snapshot.get("names", [])
    missing = set(names) - set(stored_names)
    if missing:
        raise ValueError(f"误差缓冲缺少当前启用项：{sorted(missing)}")
    rows = snapshot.get("errors")
    if (not isinstance(rows, torch.Tensor) or rows.ndim != 2
            or rows.shape[1] != len(stored_names) or not rows.shape[0]):
        raise ValueError("检查点误差缓冲为空或形状无效")
    rows = rows[:, [stored_names.index(name) for name in names]]
    previous = {name: saved[name] for name in fields}
    updates, _ = estimate_sigmas(rows, names, task.cfg.rewards, task.cfg.sigma_calibration, previous)
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest", help="upright 权重文件/目录，默认 latest")
    parser.add_argument("--recalibrate", action="store_true", help="按近期误差重新校准，遵守 sigma_cfg 的变化比例及绝对上下界")
    parser.add_argument("--write", action="store_true", help="写入 reward_cfg.py；不加则仅预览")
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint, "upright", LOG_ROOT)
    config = PROJECT_ROOT / "tasks/upright/reward_cfg.py"
    try:
        updates = checkpoint_sigmas(checkpoint, args.recalibrate)
        source = config.read_text()
        rendered = render_config(source, updates)
        print(f"检查点：{checkpoint}")
        print("模式：" + ("近期误差重新校准（相对该检查点的 sigma）" if args.recalibrate else "同步检查点实际使用值"))
        current = get_task("upright").cfg.rewards
        for name, value in updates.items():
            print(f"  {name}: {getattr(current, name):.8g} -> {value:.8g}")
        if not args.write:
            print("仅预览；添加 --write 写回配置")
            return
        if source == rendered:
            print("配置已一致，无需写入")
            return
        backup = Path(tempfile.mkdtemp(prefix="go2-sigma-backup-")) / config.name
        shutil.copy2(config, backup)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=config.parent,
                                         prefix=".reward-sigmas-", delete=False) as stream:
            stream.write(rendered)
            temporary = Path(stream.name)
        try:
            temporary.chmod(config.stat().st_mode)
            temporary.replace(config)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"已更新：{config}\n原文件备份：{backup}")
        print("训练时加 --no-adaptive-sigma 才会固定使用这些值；否则启动时会再次自动校准。")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
