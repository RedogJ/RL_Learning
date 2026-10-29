"""精简 RSL-RL 5.5.1 的终端输出，完整日志仍由原 Logger 负责。"""

import statistics
import math
import sys
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

    def _metric(self, key, reduction, as_tensor=False):
        values = [torch.as_tensor(info[key]).detach().reshape(-1)
                  for info in self.logger.ep_extras if key in info]
        if not values:
            return None
        values = torch.cat(values)
        result = (values.max() if reduction == "max" else
                  values.sum() if reduction == "sum" else values.mean())
        return result if as_tensor else result.item()

    def log(self, **kwargs):
        flush = getattr(self.logger, "flush_episodes", None)
        if flush is not None:
            flush()
        if self.logger.writer is None:
            return self.logger.log(**kwargs)

        # 原 Logger 会清空 ep_extras，所以先统计本轮的跌倒次数和机身倾斜。
        specs = (
            ("Task/failures_per_step", "sum"), ("Task/roll_abs_rad", "mean"),
            ("Task/pitch_abs_rad", "mean"), ("Task/height", "mean"),
            ("Task/horizontal_speed_mps", "mean"), ("Task/yaw_velocity_abs_radps", "mean"),
            ("Contact/foot_support_fraction", "mean"), ("Policy/raw_action_abs_max", "max"),
            ("Policy/action_clip_fraction", "mean"), ("Policy/action_limit_violation", "mean"),
            ("Task/upright_alignment", "mean"), ("Task/rear_support_count", "mean"),
            ("Task/front_support_count", "mean"),
        )
        metrics = [self._metric(key, reduction, as_tensor=True) for key, reduction in specs]
        present = [value for value in metrics if value is not None]
        # GPU 上完成全部归约，再一次传回 CPU；格式化前仍等待传输完成。
        host_values = iter(torch.stack(present).cpu().tolist() if present else [])
        (fallen, roll, pitch, height, speed, spin, foot_support, raw_max,
         clip_fraction, action_excess, upright, rear_support, front_support) = [
            next(host_values) if value is not None else None for value in metrics]
        upright_text = (f" | upright={upright:.3f} | rear={rear_support:.2f}/2 | front={front_support:.2f}/2"
                        if upright is not None and rear_support is not None and front_support is not None else "")
        components = self._reward_breakdown()
        force_balance = components.get("foot_force_balance")
        force_text = f" | force_balance={force_balance:.4f}" if force_balance is not None else ""
        foot_text = "".join(f" | {label}={components[key]:.4f}" for key, label in (
            ("foot_slide", "slide"), ("foot_time_variance", "time_balance"),
        ) if key in components)
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
        # 只在交互终端加粗，重定向到文件时保留纯文本。
        if sys.stdout.isatty():
            reward_text = f"\033[1m{reward_text}\033[22m"
            episode_text = f"\033[1m{episode_text}\033[22m"
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
              f"| raw_max={raw_max_text} | clip={clip_text} | excess={excess_text}{force_text}{foot_text}{upright_text} | top={top}", flush=True)

    def _reward_breakdown(self):
        """按正式控制样本数加权，避免重置等待使每步活动环境数变化而偏置。"""
        infos = [info for info in self.logger.ep_extras if "Reward/active_samples" in info
                 and any(key.startswith("Reward/") and key != "Reward/active_samples" for key in info)]
        if not infos:
            return {}
        counts = torch.stack([torch.as_tensor(info["Reward/active_samples"]).detach() for info in infos])
        total = counts.sum()
        keys = [key for key in infos[0] if key.startswith("Reward/") and key != "Reward/active_samples"]
        weighted = [(
            torch.stack([torch.as_tensor(info[key]).detach() for info in infos]) * counts
        ).sum().div(total.clamp_min(1)) for key in keys]
        # 同一批传输包含样本数，保留无有效样本时返回空字典的行为。
        host = torch.stack([total, *weighted]).cpu().tolist()
        if host[0] == 0:
            return {}
        return {key.removeprefix("Reward/"): value for key, value in zip(keys, host[1:])}
