"""模仿任务奖励与结束条件。"""
import torch

from .rewards import compute_reward, gravity_local


class MimicTask:
    def __init__(self, cfg):
        self.cfg = cfg

    def reward(self, *args, **kwargs):
        return compute_reward(self.cfg.reward, *args, **kwargs)

    def terminated(self, root_position, root_quat, ref_position, ref_quat):
        if not self.cfg.terminate_on_tracking_error:
            return root_position.new_zeros(root_position.shape[0], dtype=torch.bool)
        height = (root_position[:, 2] - ref_position[:, 2]).abs()
        tilt = (gravity_local(root_quat)[:, 2] - gravity_local(ref_quat)[:, 2]).abs()
        return (height > self.cfg.max_height_error) | (tilt > self.cfg.max_tilt_error)
