import unittest
import torch
from go2_learning.control.action_activity import ActionActivity


class ActivityTests(unittest.TestCase):
    def test_grouping_sign_reset_and_waiting(self):
        names = [f"{leg}_{kind}_joint" for leg in ("RR", "FL", "RL", "FR")
                 for kind in ("calf", "hip", "thigh")]
        tracker = ActionActivity(names, 2, "cpu", .1, .5)
        previous = torch.zeros(2, 12)
        actions = torch.tensor([[.2, -.2, .2] * 4] * 2)
        tracker.update(actions, previous, torch.tensor([True, False]))
        torch.testing.assert_close(tracker.penalty(), torch.zeros(2))
        self.assertEqual(tracker.values[1].sum(), 0.)
        actions[0, names.index("FL_calf_joint")] = 0.
        tracker.update(actions, previous, torch.ones(2, dtype=torch.bool))
        self.assertGreater(tracker.penalty()[0], 0.)
        saved = tracker.values[1].clone()
        tracker.reset(torch.tensor([0]))
        self.assertEqual(tracker.penalty()[0], 0.)
        torch.testing.assert_close(tracker.values[1], saved)
