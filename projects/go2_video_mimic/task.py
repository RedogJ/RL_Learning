import torch
from go2_mujoco_mimic.tasks.mimic.task import MimicTask
from .anchor import features, reanchor


class AnchorTask(MimicTask):
    def __init__(self, cfg, world_tracking=False):
        super().__init__(cfg)
        self.world_tracking = world_tracking

    def reward(self, actual, reference, action, previous_action, joint_pos, limits, dt, **kwargs):
        if self.world_tracking:
            target_pose = reanchor(actual, reference)
            tracked_pose = actual
        else:
            tracked_pose, target_pose = features(actual), features(reference)
        total, terms = super().reward(tracked_pose, target_pose, action,
                                     previous_action, joint_pos, limits, dt, **kwargs)
        # Differentiable height-gated foot slip proxy, used in real and imagined
        # transitions alike. This is not a measured contact-force label.
        ids = [4, 8, 12, 16]
        h = actual['pos'][:, ids, 2] - .022
        contact = torch.sigmoid((.015-h)/.01)
        slip = -.1 * (contact * actual['linear_vel'][:, ids, :2].square().sum(-1)).mean(-1)
        terms['foot_slip'] = slip * dt
        return total + terms['foot_slip'], terms
