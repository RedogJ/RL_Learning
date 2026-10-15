"""随机作用点的坐标转换、脉冲保持和逐环境重置；不依赖 Isaac Sim。"""

import math
import unittest

import torch

from go2_learning.randomization import DomainRandomizer, RandomizationCfg, push_positions_world
from go2_learning.tests.test_reset_settling import Robot


class RandomPointTests(unittest.TestCase):
    def make_randomizer(self, count=3, **kwargs):
        robot = Robot(count)
        recorded = []
        robot.instantaneous_wrench_composer.add_forces_and_torques_index = lambda **values: recorded.append(
            {key: value.clone() if torch.is_tensor(value) else value for key, value in values.items()}
        )
        randomizer = DomainRandomizer(robot, count, "cpu", 1 / 120, RandomizationCfg(**kwargs), 42)
        return randomizer, robot, recorded

    def test_rotated_offset_uses_world_com_and_produces_correct_moment_arm(self):
        com = torch.tensor([[[10., 20., 30.]]])
        quat = torch.tensor([[[0., 0., math.sqrt(.5), math.sqrt(.5)]]])
        point = push_positions_world(com, quat, torch.tensor([[[.1, 0., .03]]]))
        torch.testing.assert_close(point, torch.tensor([[[10., 20.1, 30.03]]]))
        torque = torch.cross(point - com, torch.tensor([[[10., 0., 0.]]]), dim=-1)
        torch.testing.assert_close(torque, torch.tensor([[[0., .3, -1.]]]), atol=1e-5, rtol=1e-5)

    def test_points_stay_on_body_during_pulse_and_forces_stay_world_horizontal(self):
        randomizer, robot, recorded = self.make_randomizer()
        randomizer.apply()
        offsets = randomizer.point_offsets_b.clone()
        original_force = recorded[-1]["forces"].clone()
        self.assertTrue((offsets.abs() <= randomizer.point_half_extent).all())
        magnitude = original_force.norm(dim=-1)
        self.assertTrue(((magnitude >= 40 - 1e-4) & (magnitude <= 100 + 1e-4)).all())
        self.assertEqual(torch.count_nonzero(original_force[..., 2]), 0)
        robot.data.body_com_pos_w.torch.add_(torch.tensor([1., 2., 3.]))
        robot.data.body_link_quat_w.torch[:] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
        randomizer.apply()
        torch.testing.assert_close(randomizer.point_offsets_b, offsets)
        torch.testing.assert_close(recorded[-1]["forces"], original_force)
        expected_offset = torch.stack((-offsets[..., 1], offsets[..., 0], offsets[..., 2]), dim=-1)
        torch.testing.assert_close(recorded[-1]["positions"], robot.data.body_com_pos_w.torch + expected_offset)
        self.assertTrue(recorded[-1]["is_global"])

    def test_partial_reset_clears_point_and_force_without_touching_other_environments(self):
        randomizer, _, recorded = self.make_randomizer()
        randomizer.apply()
        offsets = randomizer.point_offsets_b.clone()
        forces = randomizer.forces.clone()
        randomizer.reset_timers(torch.tensor([1]))
        self.assertEqual(torch.count_nonzero(randomizer.point_offsets_b[1]), 0)
        self.assertEqual(torch.count_nonzero(randomizer.forces[1]), 0)
        torch.testing.assert_close(randomizer.point_offsets_b[[0, 2]], offsets[[0, 2]])
        torch.testing.assert_close(randomizer.forces[[0, 2]], forces[[0, 2]])
        wait = randomizer.wait_steps[1].clone()
        randomizer.apply(torch.tensor([True, False, True]))
        self.assertEqual(torch.count_nonzero(recorded[-1]["forces"][1]), 0)
        self.assertEqual(randomizer.wait_steps[1], wait)

    def test_zero_extent_restores_com_application_and_pulse_stops_after_48_steps(self):
        randomizer, robot, recorded = self.make_randomizer(push_point_half_extent=(0., 0., 0.), push_duration=.4)
        for _ in range(48):
            randomizer.apply()
            self.assertGreater(torch.count_nonzero(recorded[-1]["forces"]), 0)
            torch.testing.assert_close(recorded[-1]["positions"], robot.data.body_com_pos_w.torch)
        self.assertEqual(torch.count_nonzero(randomizer.remaining_steps), 0)
        randomizer.apply()
        self.assertEqual(torch.count_nonzero(recorded[-1]["forces"]), 0)

    def test_point_range_validation(self):
        for value in ((1., 2.), (-.1, .1, .1), (.1, float("nan"), .1), (.1, .1, float("inf"))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RandomizationCfg(push_point_half_extent=value)


if __name__ == "__main__":
    unittest.main()
