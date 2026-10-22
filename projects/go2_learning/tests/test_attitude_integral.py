"""积分的时间尺度、衰减、限幅和多环境隔离检查，无需 Isaac Sim。"""

import math
import unittest

import torch

from go2_learning.control.attitude_integral import AttitudeIntegral
from go2_learning.tasks.standing.integral_cfg import IntegralCfg


class AttitudeIntegralTests(unittest.TestCase):
    def test_constant_bias_accumulates_with_simulation_time(self):
        dt, count = 0.1, 10
        integral = AttitudeIntegral(2, "cpu", dt, IntegralCfg())
        roll, pitch = torch.full((2,), 0.05), torch.full((2,), -0.03)
        height_error = torch.full((2,), -0.02)
        for _ in range(count):
            integral.update(roll, pitch, height_error)
        alpha = math.exp(-dt / integral.cfg.decay_seconds)
        factor = dt * (1 - alpha**count) / (1 - alpha)
        expected = torch.stack((roll, pitch, height_error), -1) * factor
        torch.testing.assert_close(integral.values, expected)

    def test_old_error_decays_after_leveling(self):
        integral = AttitudeIntegral(1, "cpu", 1 / 30, IntegralCfg())
        integral.values[:] = torch.tensor([[0.2, -0.1, 0.05]])
        before = integral.values.clone()
        integral.update(torch.zeros(1), torch.zeros(1), torch.zeros(1))
        torch.testing.assert_close(integral.values, before * integral.decay)

    def test_clamping_normalization_and_partial_reset(self):
        integral = AttitudeIntegral(4, "cpu", 1 / 30, IntegralCfg())
        for _ in range(100):
            integral.update(torch.full((4,), 10.0), torch.full((4,), -10.0), torch.full((4,), 10.0))
        torch.testing.assert_close(integral.values, torch.tensor([[0.5, -0.5, 0.2]]).repeat(4, 1))
        observation = integral.observations()
        torch.testing.assert_close(observation, torch.tensor([[1.0, -1.0, 1.0]]).repeat(4, 1))
        integral.reset(torch.tensor([1, 3]))
        self.assertFalse(integral.values[[1, 3]].any())
        torch.testing.assert_close(integral.values[[0, 2]], torch.tensor([[0.5, -0.5, 0.2]]).repeat(2, 1))
        self.assertTrue(observation[[1, 3]].any())  # 已返回的观测不能被重置反向修改。

    def test_invalid_limits_are_rejected(self):
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                IntegralCfg(limit=value)
            with self.assertRaises(ValueError):
                IntegralCfg(decay_seconds=value)
            with self.assertRaises(ValueError):
                IntegralCfg(height_limit=value)

    def test_height_limit_is_independent_of_attitude_limits(self):
        integral = AttitudeIntegral(2, "cpu", 1 / 30, IntegralCfg(limit=0.4, height_limit=0.1))
        for _ in range(60):
            integral.update(torch.ones(2), -torch.ones(2), torch.tensor([-1.0, 1.0]))
        torch.testing.assert_close(integral.values,
                                   torch.tensor([[0.4, -0.4, -0.1], [0.4, -0.4, 0.1]]))


if __name__ == "__main__":
    unittest.main()
