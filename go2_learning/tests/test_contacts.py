"""CPU 接触检测检查：足部豁免、名称顺序、短暂碰撞及历史窗口。"""

from types import SimpleNamespace as NS
import unittest

import torch

from go2_learning.contacts import Go2Contacts, REQUIRED_BODY_NAMES, contact_counts, select_contact_bodies


class ContactTests(unittest.TestCase):
    def test_foot_exemption_threshold_and_body_count_with_shuffled_names(self):
        names = list(reversed(REQUIRED_BODY_NAMES))
        feet, undesired = select_contact_bodies(names)
        forces = torch.zeros(3, 4, len(names), 3)
        # 第一台只用四足支撑，不管足底力多大均不惩罚。
        for index in feet:
            forces[0, :, index, 2] = 1000.
        # 第二台短暂碰到两条小腿，当前帧已没有接触。
        forces[1, 2, names.index("FL_calf"), 0] = -2.
        forces[1, 1:, names.index("RR_calf"), 2] = 30.
        # 第三台边界力 =1 N 不计；(0.8, 0.8, 0) 的模长 >1 N 应计。
        forces[2, 0, names.index("base"), 2] = 1.
        forces[2, 1, names.index("FL_thigh"), :2] = .8
        counts = contact_counts(forces, feet, undesired, 1.)
        torch.testing.assert_close(counts["foot_contact_count"], torch.tensor([4., 0., 0.]))
        torch.testing.assert_close(counts["undesired_contact_count"], torch.tensor([0., 2., 1.]))

    def test_only_current_control_window_is_used(self):
        names = list(REQUIRED_BODY_NAMES)
        history = torch.zeros(1, 6, len(names), 3)
        history[0, 4:, names.index("FL_calf"), 2] = 100.
        sensor = NS(body_names=names, cfg=NS(history_length=6),
                    data=NS(net_normal_forces_w_history=NS(torch=history)))
        contacts = Go2Contacts(sensor, threshold=1., control_substeps=4)
        self.assertEqual(contacts.read()["undesired_contact_count"].item(), 0.)
        history[0, 3, names.index("FL_calf"), 2] = 2.
        self.assertEqual(contacts.read()["undesired_contact_count"].item(), 1.)

    def test_missing_foot_or_calf_and_short_history_fail_explicitly(self):
        for name in ("FL_foot", "FR_calf"):
            names = [n for n in REQUIRED_BODY_NAMES if n != name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                select_contact_bodies(names)
        with self.assertRaises(ValueError):
            Go2Contacts(NS(cfg=NS(history_length=3)), threshold=1., control_substeps=4)

    def test_support_distinguishes_continuous_three_feet_from_brief_fourth_contact(self):
        names = list(REQUIRED_BODY_NAMES)
        feet, undesired = select_contact_bodies(names)
        forces = torch.zeros(4, 4, len(names), 3)
        forces[0, :, feet, 2] = 30.  # 四足持续支撑。
        forces[1, :, feet[:3], 2] = 30.  # 三足持续支撑。
        forces[2, :, feet[:3], 2] = 30.
        forces[2, 0, feet[3], 2] = 30.  # 第四足只碰到一帧。
        forces[3, :, feet, 0] = 100.  # 水平碰撞力不是平地向上支撑。
        result = contact_counts(forces, feet, undesired, 1.)
        torch.testing.assert_close(result["foot_support_fraction"], torch.tensor([1., .75, .8125, 0.]))
        torch.testing.assert_close(result["all_feet_support_fraction"], torch.tensor([1., 0., .25, 0.]))
        self.assertEqual(result["foot_contact_count"][2], 4.)  # 峰值日志会计 4，支撑奖励不会给满分。


if __name__ == "__main__":
    unittest.main()
