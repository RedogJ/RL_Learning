"""训练阶段历史：每个检查点携带截至该轮的奖励配置和来源链。"""

from copy import deepcopy
import json
from pathlib import Path
import warnings

from go2_learning.tasks import checkpoint_task_name
from go2_learning.training.checkpoints import relocate_checkpoint


def clip_history(segments, completed_rounds):
    """轮数采用从 1 开始的闭区间；中间检查点不继承之后的训练。"""
    result = []
    for original in segments:
        if original["start_round"] > completed_rounds:
            break
        segment = deepcopy(original)
        segment["end_round"] = min(segment["end_round"], completed_rounds)
        if segment["end_round"] < segment["start_round"]:
            continue
        expected = result[-1]["end_round"] + 1 if result else 1
        if segment["start_round"] != expected:
            raise ValueError("训练历史的轮数区间不连续，不能自动继承")
        segment["completed_updates"] = segment["end_round"] - segment["start_round"] + 1
        result.append(segment)
    if completed_rounds and (not result or result[-1]["end_round"] != completed_rounds):
        raise ValueError("训练历史没有覆盖检查点对应的全部轮数")
    return result


def make_segment(run_cfg, run_id, start_round, end_round, origin):
    keys = ("task_name", "task_version", "env_cfg", "termination_cfg", "reward_cfg", "agent_cfg",
            "command_cfg", "command_source", "integral_cfg", "randomization_cfg", "num_envs", "seed")
    return {
        "run_id": run_id,
        "source_checkpoint": run_cfg.get("resumed_from"),
        "source_checkpoint_relative": run_cfg.get("resumed_from_relative"),
        "source_task": run_cfg.get("resumed_from_task"),
        "start_round": start_round,
        "end_round": end_round,
        "completed_updates": end_round - start_round + 1,
        "origin": origin,
        **{key: deepcopy(run_cfg.get(key)) for key in keys},
    }


def inherited_history(checkpoint, run_cfg, completed_rounds, infos, log_root, _seen=None):
    """优先读取权重内的历史；旧权重沿 run.json 来源链补建，只读旧文件。"""
    checkpoint, log_root = Path(checkpoint), Path(log_root)
    if isinstance(infos, dict) and "training_history" in infos:
        return clip_history(infos["training_history"]["segments"], completed_rounds)
    seen = set() if _seen is None else _seen
    identity = str(checkpoint.resolve())
    if identity in seen:
        raise ValueError("旧训练的 resumed_from 出现循环，无法继承历史")
    seen.add(identity)
    parent_history, parent_rounds = [], 0
    source = run_cfg.get("resumed_from")
    relative_source = run_cfg.get("resumed_from_relative")
    if relative_source:
        source = str(log_root / relative_source)
    if source:
        source_task = run_cfg.get("resumed_from_task") or checkpoint_task_name(run_cfg)
        parent = relocate_checkpoint(source, log_root, source_task)
        metadata = parent.with_name("run.json")
        if not parent.is_file() or not metadata.is_file():
            warnings.warn(f"旧历史来源缺失：{source}；继承轮数将标为配置未知，不猜测奖励配置。")
            return [{"run_id": checkpoint.parent.name, "source_checkpoint": source,
                     "start_round": 1, "end_round": completed_rounds,
                     "completed_updates": completed_rounds, "reward_cfg": None,
                     "origin": "unknown_missing_ancestor"}] if completed_rounds else []
        import torch

        data = torch.load(parent, map_location="cpu", weights_only=False)
        parent_rounds = int(data["iter"]) + 1
        parent_history = inherited_history(
            parent, json.loads(metadata.read_text()), parent_rounds, data.get("infos"), log_root, seen,
        )
    if completed_rounds <= parent_rounds:
        return clip_history(parent_history, completed_rounds)
    return parent_history + [make_segment(run_cfg, checkpoint.parent.name, parent_rounds + 1,
                                         completed_rounds, "legacy_run_json")]


def validate_network_config(current, saved):
    """检查结构；初始标准差会被网络权重覆盖，不属于结构差异。"""
    for name in ("actor", "critic"):
        def structure(cfg):
            cfg = deepcopy(cfg)
            cfg.setdefault("distribution_cfg", {}).pop("init_std", None)
            return cfg
        if structure(current[name]) != structure(saved[name]):
            raise ValueError(f"当前 {name} 网络结构与检查点不一致，不能仅加载权重")


def write_json(path, data):
    """同目录替换，避免进程中断留下半个 JSON。"""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class TrainingHistoryMixin:
    """接入 OnPolicyRunner.save，覆盖定期检查点和最终模型的历史记录。"""

    def configure_history(self, log_dir, run_cfg, parent_history):
        self._history_dir = Path(log_dir)
        self._history_cfg = deepcopy(run_cfg)
        self._history_start = self.current_learning_iteration + 1
        self._parent_history = clip_history(parent_history, self._history_start - 1)
        self._write_history(self._history_start - 1)

    def _history_document(self, completed_rounds):
        segments = deepcopy(self._parent_history)
        if completed_rounds >= self._history_start:
            segments.append(make_segment(self._history_cfg, self._history_dir.name, self._history_start,
                                         completed_rounds, "checkpoint"))
        return {"schema_version": 1, "round_numbering": "one_based_inclusive",
                "completed_rounds": completed_rounds, "segments": segments}

    def _write_history(self, completed_rounds, last_checkpoint=None):
        document = self._history_document(completed_rounds)
        write_json(self._history_dir / "history.json", document)
        metadata = {**self._history_cfg, "start_round": self._history_start,
                    "end_round": completed_rounds, "completed_rounds": completed_rounds,
                    "completed_updates": completed_rounds - self._history_start + 1,
                    "planned_end_round": self._history_start - 1 + self._history_cfg["requested_updates"],
                    "history_file": "history.json", "last_checkpoint": last_checkpoint}
        write_json(self._history_dir / "run.json", metadata)

    def save(self, path, infos=None):
        completed_rounds = self.current_learning_iteration + 1
        document = self._history_document(completed_rounds)
        payload = {**(infos or {}), "training_history": document,
                   "reward_cfg": deepcopy(self._history_cfg["reward_cfg"])}
        super().save(path, infos=payload)
        self._write_history(completed_rounds, Path(path).name)
