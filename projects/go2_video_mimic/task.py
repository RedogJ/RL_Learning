import torch
from go2_mujoco_mimic.tasks.mimic.task import MimicTask
from .anchor import features


class AnchorTask(MimicTask):
    def reward(self, actual, reference, action, previous_action, joint_pos, limits, dt, **kwargs):
        total, terms = super().reward(features(actual), features(reference), action,
                                     previous_action, joint_pos, limits, dt, **kwargs)
        # Differentiable height-gated foot slip proxy, used in real and imagined
        # transitions alike. This is not a measured contact-force label.
        ids = [4, 8, 12, 16]
        h = actual['pos'][:, ids, 2] - .022
        contact = torch.sigmoid((.015-h)/.01)
        slip = -.1 * (contact * actual['linear_vel'][:, ids, :2].square().sum(-1)).mean(-1)
        terms['foot_slip'] = slip * dt
        return total + terms['foot_slip'], terms
