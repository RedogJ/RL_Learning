"""模仿任务奖励与结束条件。"""
from .rewards import compute_reward, gravity_local


class MimicTask:
    def __init__(self, cfg):
        self.cfg = cfg

    def reward(self, *args, **kwargs):
        return compute_reward(self.cfg.reward, *args, **kwargs)

    def terminated(self, root_position, root_quat, ref_position, ref_quat):
        height = (root_position[:, 2] - ref_position[:, 2]).abs()
        tilt = (gravity_local(root_quat)[:, 2] - gravity_local(ref_quat)[:, 2]).abs()
        return (height > self.cfg.max_height_error) | (tilt > self.cfg.max_tilt_error)
