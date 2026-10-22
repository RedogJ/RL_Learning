"""CPU 接触检测检查：足部豁免、名称顺序、短暂碰撞及历史窗口。"""

from types import SimpleNamespace as NS
import unittest
import math

import torch

from go2_learning.simulation.contacts import (
    Go2Contacts, HIP_NAMES, REQUIRED_BODY_NAMES, contact_counts, select_contact_bodies,
)


class ContactTests(unittest.TestCase):
    def test_all_airborne_counts_simultaneous_physics_frames(self):
        names = list(REQUIRED_BODY_NAMES)
        feet, undesired = select_contact_bodies(names)
        forces = torch.zeros(3, 4, len(names), 3)
        # 环境0每帧至少一脚支撑；环境1只有一帧四脚同时离地；环境2全程腾空。
        for frame in range(4):
            forces[0, frame, feet[frame], 2] = 10.
        forces[1, :3, feet[0], 2] = 10.
        result = contact_counts(forces, feet, undesired, 1.)
        torch.testing.assert_close(result["all_feet_airborne_fraction"], torch.tensor([0., .25, 1.]))

    def test_force_smoothing_uses_same_tau_chronology_and_partial_reset(self):
        names = list(reversed(REQUIRED_BODY_NAMES))
        history = torch.zeros(2, 4, len(names), 3)
        feet, _ = select_contact_bodies(names)
        for frame, force in enumerate((40., 30., 20., 10.)):
            history[:, frame, feet, 2] = force
        sensor = NS(body_names=names, cfg=NS(history_length=4),
                    data=NS(net_normal_forces_w_history=NS(torch=history)))
        c = Go2Contacts(sensor, 1., 4)
        c.update_air_time(.01, torch.tensor([True, False]), ratio_tau=.6)
        alpha = -math.expm1(-.01 / .6)
        expected = 0.
        for force in (10., 20., 30., 40.):
            expected += alpha * (force - expected)
        torch.testing.assert_close(c.mean_force_z[0], torch.full((4,), expected))
        self.assertEqual(c.mean_force_z[1].sum(), 0.)
        snapshot = c.mean_force_z.clone()
        c.read(); c.read()
        torch.testing.assert_close(c.mean_force_z, snapshot)
        c.reset(torch.tensor([1]))
        torch.testing.assert_close(c.mean_force_z[0], snapshot[0])
        c.reset(torch.tensor([0]))
        self.assertEqual(c.mean_force_z.sum(), 0.)

    def test_recent_ratio_does_not_reset_on_brief_touch(self):
        names = list(REQUIRED_BODY_NAMES)
        history = torch.zeros(2, 4, len(names), 3)
        sensor = NS(body_names=names, cfg=NS(history_length=4),
                    data=NS(net_normal_forces_w_history=NS(torch=history)))
        c = Go2Contacts(sensor, 1., 4)
        active = torch.tensor([True, False])
        for _ in range(100):
            c.update_air_time(.01, active)
        self.assertLess(c.contact_ratio[0, 0], .01)
        history[0, 0, names.index("FL_foot"), 2] = 10.
        c.update_air_time(.01, active)
        self.assertGreater(c.contact_ratio[0, 0], .01)
        self.assertLess(c.contact_ratio[0, 0], .05)
        torch.testing.assert_close(c.contact_ratio[1], torch.full((4,), 1.))
        c.reset(torch.tensor([0]))
        torch.testing.assert_close(c.contact_ratio[0], torch.full((4,), 1.))

    def test_air_time_short_landing_order_waiting_and_partial_reset(self):
        names = list(REQUIRED_BODY_NAMES)
        history = torch.zeros(2, 4, len(names), 3)
        sensor = NS(body_names=names, cfg=NS(history_length=4),
                    data=NS(net_normal_forces_w_history=NS(torch=history)))
        contacts = Go2Contacts(sensor, 1., 4)
        contacts.update_air_time(.01, torch.tensor([True, False]))
        torch.testing.assert_close(contacts.air_time[0], torch.full((4,), .04))
        self.assertEqual(contacts.air_time[1].sum(), 0.)
        # 第 2 帧落地，之后第 1、0 帧离地，应只累计 .02 秒。
        history[0, 2, names.index("FL_foot"), 2] = 2.
        contacts.update_air_time(.01, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(contacts.air_time[0], torch.tensor([.02, .08, .08, .08]))
        before = contacts.air_time.clone()
        contacts.read(); contacts.read()
        torch.testing.assert_close(contacts.air_time, before)
        contacts.reset(torch.tensor([0]))
        self.assertEqual(contacts.air_time[0].sum(), 0.)
        torch.testing.assert_close(contacts.air_time[1], before[1])

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

    def test_hip_contacts_are_counted_apart_from_other_undesired_bodies(self):
        names = list(reversed(REQUIRED_BODY_NAMES))
        feet, undesired = select_contact_bodies(names)
        hip_ids = [names.index(name) for name in HIP_NAMES]
        forces = torch.zeros(2, 4, len(names), 3)
        # 第一台同时碰到一个髋部刚体和机身；第二台只有机身接地。
        forces[0, 1, names.index("RL_hip"), 2] = 5.
        forces[0, 2, names.index("FR_hip"), :2] = .8  # 模长 > 1 N。
        forces[0, :, names.index("base"), 2] = 10.
        forces[1, :, names.index("base"), 2] = 10.
        counts = contact_counts(forces, feet, undesired, 1., hip_ids)
        torch.testing.assert_close(counts["hip_contact_count"], torch.tensor([2., 0.]))
        torch.testing.assert_close(counts["undesired_contact_count"], torch.tensor([3., 1.]))
        # 未传髋部索引时保持为零，不影响旧调用方。
        self.assertEqual(contact_counts(forces, feet, undesired, 1.)["hip_contact_count"].sum(), 0.)

    def test_both_rear_feet_airborne_requires_simultaneous_liftoff(self):
        names = list(REQUIRED_BODY_NAMES)
        feet, undesired = select_contact_bodies(names)
        rear = [feet.index(names.index("RL_foot")), feet.index(names.index("RR_foot"))]
        forces = torch.zeros(4, 4, len(names), 3)
        # 环境 0 两后脚全程离地；环境 1 两后脚全程支撑。
        forces[1, :, [names.index("RL_foot"), names.index("RR_foot")], 2] = 30.
        # 环境 2 左右后脚交替支撑，任意一帧都有一只脚在支撑。
        forces[2, :2, names.index("RL_foot"), 2] = 30.
        forces[2, 2:, names.index("RR_foot"), 2] = 30.
        # 环境 3 前两帧两后脚都离地，后两帧都支撑。
        forces[3, :2, [names.index("RL_foot"), names.index("RR_foot")], 2] = 30.
        result = contact_counts(forces, feet, undesired, 1., rear_ids=rear)
        torch.testing.assert_close(result["rear_feet_airborne_fraction"],
                                   torch.tensor([1., 0., 0., .5]))

    def test_rear_airborne_fraction_survives_shuffled_sensor_order(self):
        names = list(reversed(REQUIRED_BODY_NAMES))
        history = torch.zeros(1, 4, len(names), 3)
        history[:, :2, [names.index("RL_foot"), names.index("RR_foot")], 2] = 10.
        sensor = NS(body_names=names, cfg=NS(history_length=4),
                    data=NS(net_normal_forces_w_history=NS(torch=history)))
        self.assertEqual(Go2Contacts(sensor, 1., 4).read()["rear_feet_airborne_fraction"].item(), .5)

    def test_rear_knee_and_foot_support_requires_same_physics_frame(self):
        names = list(REQUIRED_BODY_NAMES)
        feet, undesired = select_contact_bodies(names)
        rear = [feet.index(names.index("RL_foot")), feet.index(names.index("RR_foot"))]
        knee = [names.index("RL_calf"), names.index("RR_calf")]
        feet_bodies = [names.index("RL_foot"), names.index("RR_foot")]
        forces = torch.zeros(3, 4, len(names), 3)
        # 环境 0：两后脚与两膝全程同时支撑。
        forces[0, :, feet_bodies, 2] = 30.
        forces[0, :, knee, 2] = 30.
        # 环境 1：前两帧脚掌、后两帧膝，任何一帧都不同时。
        forces[1, :2, feet_bodies, 2] = 30.
        forces[1, 2:, knee, 2] = 30.
        # 环境 2：脚掌全程支撑，膝只在前两帧，重叠 2/4。
        forces[2, :, feet_bodies, 2] = 30.
        forces[2, :2, knee, 2] = 30.
        result = contact_counts(forces, feet, undesired, 1., rear_ids=rear, knee_ids=knee)
        torch.testing.assert_close(result["rear_knee_and_foot_support_fraction"],
                                   torch.tensor([1., 0., .5]))
        torch.testing.assert_close(result["rear_knee_contact_count"], torch.tensor([2., 2., 2.]))
        torch.testing.assert_close(result["rear_knee_support_count"], torch.tensor([2., 2., 2.]))
        torch.testing.assert_close(result["rear_knee_support_fraction"],
                                   torch.tensor([1., .5, .5]))
        torch.testing.assert_close(result["rear_leg_supported_fraction"],
                                   torch.tensor([[1., 1.], [0., 0.], [.5, .5]]))
        torch.testing.assert_close(result["rear_both_unsupported_fraction"],
                                   torch.tensor([0., 1., .5]))

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
