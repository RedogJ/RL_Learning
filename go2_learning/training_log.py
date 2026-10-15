"""精简 RSL-RL 5.5.1 的终端输出，完整日志仍由原 Logger 负责。"""

import statistics
import math
from contextlib import redirect_stdout
from pathlib import Path

import torch


class CompactTrainingLogger:
    """包装现有 Logger，仅替换每轮更新的终端显示，不改采样和 PPO。"""

    def __init__(self, logger, control_dt):
        self.logger = logger
        self.control_dt = control_dt
        self.fallen_total = 0
        self.details_path = Path(logger.log_dir) / "training_details.log"

    def __getattr__(self, name):
        # process_env_step、writer、save_model 等继续使用 RSL-RL 原实现。
        return getattr(self.logger, name)

    def _metric(self, key, reduction):
        values = [torch.as_tensor(info[key]).detach().reshape(-1)
                  for info in self.logger.ep_extras if key in info]
        if not values:
            return None
        values = torch.cat(values)
        if reduction == "max":
            return values.max().item()
        return (values.sum() if reduction == "sum" else values.mean()).item()

    def log(self, **kwargs):
        if self.logger.writer is None:
            return self.logger.log(**kwargs)

        # 原 Logger 会清空 ep_extras，所以先统计本轮的跌倒次数和机身倾斜。
        fallen = self._metric("Task/failures_per_step", "sum")
        roll = self._metric("Task/roll_abs_rad", "mean")
        pitch = self._metric("Task/pitch_abs_rad", "mean")
        height = self._metric("Task/height", "mean")
        speed = self._metric("Task/horizontal_speed_mps", "mean")
        spin = self._metric("Task/yaw_velocity_abs_radps", "mean")
        foot_support = self._metric("Contact/foot_support_fraction", "mean")
        raw_max = self._metric("Policy/raw_action_abs_max", "max")
        clip_fraction = self._metric("Policy/action_clip_fraction", "mean")
        action_excess = self._metric("Policy/action_limit_violation", "mean")
        components = self._reward_breakdown()
        reward = statistics.mean(self.logger.rewbuffer) if self.logger.rewbuffer else None
        episode_seconds = (statistics.mean(self.logger.lenbuffer) * self.control_dt
                           if self.logger.lenbuffer else None)

        # 仅将 log() 的长表写到文件；TensorBoard 写入仍执行，异常照常向外抛出。
        with self.details_path.open("a", encoding="utf-8") as details, redirect_stdout(details):
            self.logger.log(**kwargs)

        if fallen is not None:
            self.fallen_total += int(round(fallen))
        fallen_text = f"{int(round(fallen))} (total={self.fallen_total})" if fallen is not None else "--"
        reward_text = f"{reward:.3f}" if reward is not None else "--"
        episode_text = f"{episode_seconds:.2f}s" if episode_seconds is not None else "--"
        roll_text = f"{math.degrees(roll):.2f}deg" if roll is not None else "--"
        pitch_text = f"{math.degrees(pitch):.2f}deg" if pitch is not None else "--"
        height_text = f"{height:.3f}m" if height is not None else "--"
        speed_text = f"{speed:.3f}m/s" if speed is not None else "--"
        spin_text = f"{spin:.3f}rad/s" if spin is not None else "--"
        feet_text = f"{4 * foot_support:.2f}/4" if foot_support is not None else "--"
        raw_max_text = f"{raw_max:.2f}" if raw_max is not None else "--"
        clip_text = f"{clip_fraction:.1%}" if clip_fraction is not None else "--"
        excess_text = f"{action_excess:.3f}" if action_excess is not None else "--"
        current = kwargs["it"] - kwargs["start_it"] + 1
        total = kwargs["total_it"] - kwargs["start_it"]
        top = "--"
        if components:
            penalties = [(name, -value) for name, value in components.items() if value < 0]
            if penalties:
                name, value = max(penalties, key=lambda item: item[1])
                top = f"{name} ({value / sum(v for _, v in penalties):.1%})"
            else:
                top = "none"
        print(f"[{current}/{total}] fallen={fallen_text} | reward={reward_text} "
              f"| episode={episode_text} | roll={roll_text} | pitch={pitch_text} | height={height_text} "
              f"| speed={speed_text} | spin={spin_text} | feet={feet_text} "
              f"| raw_max={raw_max_text} | clip={clip_text} | excess={excess_text} | top={top}", flush=True)

    def _reward_breakdown(self):
        """按正式控制样本数加权，避免重置等待使每步活动环境数变化而偏置。"""
        infos = [info for info in self.logger.ep_extras if "Reward/active_samples" in info]
        if not infos:
            return {}
        counts = torch.stack([torch.as_tensor(info["Reward/active_samples"]).detach() for info in infos])
        total = counts.sum()
        if total.item() == 0:
            return {}
        keys = [key for key in infos[0] if key.startswith("Reward/") and key != "Reward/active_samples"]
        return {key.removeprefix("Reward/"): (
            torch.stack([torch.as_tensor(info[key]).detach() for info in infos]) * counts
        ).sum().div(total).item() for key in keys}
