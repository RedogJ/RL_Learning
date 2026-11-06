"""独立关节奖励的环境隔离、梯度及旧奖励配置兼容性。"""
from dataclasses import asdict
import json
import math
import unittest

import torch

from go2_mujoco_mimic.tasks.mimic.backflip_reward_cfg import make_backflip_reward_cfg
from go2_mujoco_mimic.tasks.mimic.reward_cfg import RewardCfg, saved_reward_config
from go2_mujoco_mimic.tasks.mimic.rewards import compute_reward


def joint_reward(cfg, errors):
    n = errors.shape[0]
    positions = errors.new_zeros(n, 1, 3)
    quats = errors.new_zeros(n, 1, 4)
    quats[..., 0] = 1
    actual = dict(pos=positions, quat=quats, linear_vel=positions, angular_vel=positions)
    reference = dict(actual, joint_pos=torch.zeros_like(errors))
    limits = errors.new_tensor([-10., 10.]).expand(12, 2)
    _, terms = compute_reward(cfg, actual, reference, torch.zeros_like(errors),
                              torch.zeros_like(errors), errors, limits, 1/30)
    return terms['joint_position']


class JointRewardTests(unittest.TestCase):
    def test_independent_weights_preserve_environment_and_joint_identity(self):
        weights = (2.,) + (0.,) * 11
        sigmas = (.5,) + (1.,) * 11
        cfg = RewardCfg(joint_position_weights=weights, joint_sigmas=sigmas)
        errors = torch.zeros(2, 12)
        errors[0, 0] = .5
        errors[1, 1] = 1.0  # 另一个环境里零权重关节的误差不影响第一个关节。
        torch.testing.assert_close(joint_reward(cfg, errors), torch.tensor([2/math.e, 2.]))
        self.assertEqual(cfg.joint_position_max_reward, 2.)

    def test_sum_is_equivalent_to_old_twelve_times_mean_at_same_sigma(self):
        errors = torch.linspace(-1., 1., 24).reshape(2, 12)
        old = RewardCfg(joint_position_weight=12., joint_sigmas=(.9, 1., .85))
        new = RewardCfg(joint_position_weights=(1.,)*12, joint_sigmas=(.9, 1., .85))
        torch.testing.assert_close(joint_reward(new, errors), joint_reward(old, errors))

    def test_tighter_sigma_changes_reward_and_gradient_points_toward_target(self):
        old = RewardCfg(joint_position_weight=12., joint_sigmas=(.9, 1., .85))
        new = make_backflip_reward_cfg()
        new.torque_l2_weight = new.joint_acc_l2_weight = 0.
        errors = torch.zeros(2, 12)
        errors[0, 7] = math.radians(53)
        errors[1, 7] = -math.radians(53)
        errors.requires_grad_()
        reward = joint_reward(new, errors)
        self.assertTrue((reward < joint_reward(old, errors)).all())
        reward.sum().backward()
        self.assertLess(errors.grad[0, 7].item(), 0.)
        self.assertGreater(errors.grad[1, 7].item(), 0.)
        self.assertTrue(torch.isfinite(errors.grad).all())

    def test_json_roundtrip_and_legacy_config_keep_their_meaning(self):
        new = make_backflip_reward_cfg()
        restored = RewardCfg(**saved_reward_config(json.loads(json.dumps(asdict(new)))))
        self.assertEqual(asdict(new), asdict(restored))
        self.assertEqual(restored.joint_position_max_reward, 12.)
        old = RewardCfg(**saved_reward_config(dict(joint_position_weight=12.,
                                                   joint_sigmas=[.9, 1., .85])))
        self.assertIsNone(old.joint_position_weights)
        self.assertEqual(old.joint_position_mode, 'group_mse')
        self.assertEqual(old.joint_position_max_reward, 12.)
        torch.testing.assert_close(joint_reward(old, torch.zeros(2, 12)), torch.full((2,), 12.))

    def test_invalid_per_joint_configuration_fails(self):
        for values in ((1.,)*11, (float('nan'),)+(1.,)*11, (-1.,)+(1.,)*11):
            with self.assertRaises(ValueError):
                RewardCfg(joint_position_weights=values)
        with self.assertRaises(ValueError):
            RewardCfg(joint_position_weights=(1.,)*12, joint_position_mode='group_mse')
        with self.assertRaises(ValueError):
            RewardCfg(joint_sigmas=(1.,)*11)


if __name__ == '__main__':
    unittest.main()
