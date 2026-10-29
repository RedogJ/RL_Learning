"""直接加载 RSL-RL 5 的独立 Actor，推理只使用高斯分布均值。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Policy:
    def __init__(self, checkpoint):
        self.path = Path(checkpoint).expanduser().resolve()
        self.cfg = json.loads(self.path.with_name("run.json").read_text())
        self.task = self.cfg.get("task_name", "running")
        if self.task not in ("standing", "running", "upright"):
            raise ValueError(f"不支持的任务: {self.task}")
        self.filter_alpha = self.cfg.get("reward_cfg", {}).get("action_filter_alpha", 0.0)
        if not 0 <= self.filter_alpha < 1:
            raise ValueError("action_filter_alpha 必须在 [0, 1) 范围内")
        actor_cfg = self.cfg["agent_cfg"]["actor"]
        if (actor_cfg["class_name"] != "MLPModel" or actor_cfg["activation"] != "tanh"
                or actor_cfg.get("obs_normalization", False)):
            raise ValueError("本接入支持无观测归一化的 Tanh MLP Actor")
        self.joint_names = list(self.cfg["joint_names"])
        self.default_q = np.array(self.cfg["default_joint_pos"], dtype=np.float64)
        self.limits = np.array(self.cfg["action_joint_limits"], dtype=np.float64)
        self.control_dt = self.cfg["env_cfg"]["physics_dt"] * self.cfg["env_cfg"]["decimation"]
        self.center = self.limits.mean(axis=1)
        self.half_range = (self.limits[:, 1] - self.limits[:, 0]) / 2
        if len(self.joint_names) != 12 or self.limits.shape != (12, 2) or not (self.half_range > 0).all():
            raise ValueError("检查点的 12 关节定义或机械限位无效")
        checkpoint_data = torch.load(self.path, map_location="cpu", weights_only=True)
        state = checkpoint_data["actor_state_dict"]
        dims = [52, *actor_cfg["hidden_dims"], 12]
        layers = []
        for i, (inputs, outputs) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(inputs, outputs))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        weights = {k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")}
        extra = set(state) - {"distribution.log_std_param"} - {f"mlp.{k}" for k in weights}
        if extra:
            raise ValueError(f"Actor 中存在未处理的参数：{extra}")
        self.net.load_state_dict(weights, strict=True)
        self.net.eval()
        self.iteration = checkpoint_data["iter"]
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @torch.inference_mode()
    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (52,) or not np.isfinite(observation).all():
            raise ValueError("Actor 要求 52 维有限观测")
        result = self.net(torch.from_numpy(observation)).numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Actor 输出非有限动作")
        return result

    def target(self, actions):
        return self.center + self.half_range * np.clip(actions, -1., 1.)

    @property
    def standing_action(self):
        return np.clip((self.default_q - self.center) / self.half_range, -1., 1.)
