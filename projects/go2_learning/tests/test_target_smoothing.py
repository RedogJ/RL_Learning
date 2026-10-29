"""目标角平滑与原始输出超界应互补，且逐环境重置不能串入上一回合。"""

import unittest

import torch

from go2_learning.tasks import get_task
from go2_learning.tests.test_reset_settling import Scene, Sim, load_env_class


class TargetSmoothingTests(unittest.TestCase):
    def make_env(self):
        task = get_task("upright")
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        env.finish_initial_settling()
        return env

    def terms(self, env):
        return env.task.reward_components(env._read_state(), env.actions,
                                          torch.zeros(3, dtype=torch.bool), env.dt)

    def test_constant_clipped_target_has_no_smoothing_cost_but_raw_excess_cost(self):
        env = self.make_env()
        env._set_action(torch.full((3, 12), 2.))
        before = self.terms(env)["action_limit"]
        env._set_action(torch.full((3, 12), 8.))
        terms = self.terms(env)
        torch.testing.assert_close(terms["action_rate"], torch.zeros(3))
        self.assertTrue((terms["action_limit"] < before).all())

    def test_limit_reversal_penalizes_physical_target_change_without_raw_excess(self):
        env = self.make_env()
        env._set_action(torch.ones(3, 12))
        previous_target = env.action_center + env.action_half_range * env.actions
        env._set_action(-torch.ones(3, 12))
        target = env.action_center + env.action_half_range * env.actions
        terms = self.terms(env)
        expected = -env.reward_cfg.action_rate_weight * (target - previous_target).square().mean(-1) * env.dt
        torch.testing.assert_close(terms["action_rate"], expected)
        torch.testing.assert_close(terms["action_limit"], torch.zeros(3))

    def test_partial_reset_uses_waiting_target_and_keeps_other_env_history(self):
        env = self.make_env()
        env._set_action(torch.ones(3, 12))
        env.reset(torch.tensor([0]))
        # 等待中的环境 0 继续下发等待目标；环境 1/2 则由策略反向。
        env._set_action(-torch.ones(3, 12))
        delta = env._read_state()["joint_target_delta"]
        torch.testing.assert_close(delta[0], torch.zeros(12))
        torch.testing.assert_close(delta[1:], -2 * env.action_half_range[1:])
        env.reward_cfg.action_rate_weight = 0
        state = env._read_state()
        self.assertNotIn("joint_target_delta", state)
        torch.testing.assert_close(self.terms(env)["action_rate"], torch.zeros(3))

    def test_old_raw_difference_and_new_round_trip_are_preserved(self):
        env = self.make_env()
        saved = env.task.run_config(False)
        restored = get_task("upright")
        restored.load_playback_config(saved)
        self.assertEqual(restored.run_config(False)["reward_cfg"], saved["reward_cfg"])
        saved["reward_cfg"]["reward_schema_version"] = 9
        restored.load_playback_config(saved)
        env.task = restored
        env.reward_cfg = restored.cfg.rewards
        env._set_action(torch.full((3, 12), 2.))
        env._set_action(torch.full((3, 12), 8.))
        expected = torch.full((3,), -env.reward_cfg.action_rate_weight * 36 * env.dt)
        torch.testing.assert_close(self.terms(env)["action_rate"], expected)


if __name__ == "__main__":
    unittest.main()
