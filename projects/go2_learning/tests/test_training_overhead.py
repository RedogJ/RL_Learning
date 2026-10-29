"""验证采样优化保留完整回合、活动掩码及误差缓冲的语义。"""

from collections import deque
from types import SimpleNamespace as NS
import unittest

import torch

from go2_learning.training.settling import SettlingLogger
from go2_learning.training.reward_calibration import ErrorReplayBuffer
from go2_learning.tests.test_reset_settling import Scene, Sim, load_env_class


class TrainingOverheadTests(unittest.TestCase):
    def test_episode_batch_preserves_multiple_resets_order_and_waiting(self):
        for intrinsic in (False, True):
            logger = NS(writer=object(), ep_extras=[], rewbuffer=deque(maxlen=100),
                        lenbuffer=deque(maxlen=100), cur_reward_sum=torch.zeros(3),
                        cur_episode_length=torch.zeros(3), erewbuffer=deque(maxlen=100),
                        irewbuffer=deque(maxlen=100), cur_ereward_sum=torch.zeros(3),
                        cur_ireward_sum=torch.zeros(3))
            batched = SettlingLogger(logger)
            reference_reward = torch.zeros(3)
            reference_length = torch.zeros(3)
            expected = []
            # 同一环境可以一轮内结束多次；顺序必须按时间再按环境编号。
            for step in range(24):
                active = torch.tensor([True, step % 3 != 0, step % 4 != 0])
                rewards = torch.tensor([1., 2., 3.]) * active
                bonus = rewards * .25 if intrinsic else None
                done = torch.tensor([step % 2 == 1, step % 3 == 2, step % 4 == 3])
                reference_reward += rewards if bonus is None else rewards + bonus
                reference_length += active
                for index in done.nonzero().flatten().tolist():
                    expected.append((reference_reward[index].item(), reference_length[index].item()))
                reference_reward[done] = 0
                reference_length[done] = 0
                batched.process_env_step(rewards, done, {"policy_active": active, "log": {}}, bonus)
            self.assertEqual(list(logger.rewbuffer), [])  # flush 之前没有 CPU 回合传输。
            batched.flush_episodes()
            self.assertEqual(list(logger.rewbuffer), [row[0] for row in expected])
            self.assertEqual(list(logger.lenbuffer), [row[1] for row in expected])
            torch.testing.assert_close(logger.cur_episode_length, reference_length)
            torch.testing.assert_close(logger.cur_reward_sum, reference_reward)
            if intrinsic:
                self.assertEqual(list(logger.erewbuffer), [row[0] / 1.25 for row in expected])
                self.assertEqual(list(logger.irewbuffer), [row[0] / 5 for row in expected])
            batched.flush_episodes()
            self.assertEqual(len(logger.rewbuffer), len(expected))

    def test_cached_selection_is_consumed_once_and_reset_invalidates_it(self):
        scene = Scene()
        env = load_env_class()(Sim(scene), scene)
        env.finish_initial_settling()
        # 环境 1 下一步结束等待，但本步仍不可被计为有效样本。
        env.settle_steps_remaining[1] = 1
        active = env.policy_active
        ids = active.nonzero().flatten()
        original = env.task.reward_components
        sizes = []

        def reward(state, actions, failed, dt):
            sizes.append(actions.shape[0])
            self.assertIs(env._active_ids, ids)
            return original(state, actions, failed, dt)

        env.task.reward_components = reward
        env.cache_policy_selection(active, ids)
        _, _, _, extras = env.step(env.standing_actions)
        self.assertEqual(sizes, [2])
        torch.testing.assert_close(extras["policy_active"], torch.tensor([True, False, True]))
        self.assertTrue(env.policy_active.all())
        self.assertIsNone(env._pending_policy_selection)
        self.assertIsNone(env._step_active_ids)
        env.cache_policy_selection(active, ids)
        env.reset(torch.tensor([0]))
        self.assertIsNone(env._pending_policy_selection)

    def test_sparse_diagnostics_keep_all_failure_counts(self):
        scene = Scene()
        env = load_env_class()(Sim(scene), scene)
        env.diagnostics_interval = 24
        env.detailed_diagnostics = False
        state = env._read_state()
        active = torch.ones(3, dtype=torch.bool)
        failed = torch.tensor([True, False, True])
        env._compute_reward(state, failed, active)
        counts = []
        for step in range(48):
            log = env._make_extras(state, failed, torch.zeros(3, dtype=torch.bool), active)["log"]
            counts.append(log["Task/failures_per_step"])
            self.assertEqual("Task/height" in log, step % 24 == 0)
            self.assertNotIn("Task/hip_FL_rad", log)
            self.assertNotIn("Contact/rear_knee_contact_count", log)
        self.assertEqual(torch.stack(counts).sum().item(), 96)

    def test_error_sampling_matches_full_stack_before_selection(self):
        cfg = NS(capacity=9, samples_per_step=4)
        buffer = ErrorReplayBuffer(("a", "b"), cfg, "cpu", {})
        expected = []
        for step, n in enumerate((7, 3, 8, 5)):
            squared = {"a": torch.arange(n).float().square(),
                       "b": (torch.arange(n).float() + 10).square()}
            valid = {"b": torch.arange(n) % 2 == 0}
            ids = (torch.arange(min(n, 4)) * n // min(n, 4) + step) % n
            rows = torch.stack(list(squared.values()), -1)[ids].sqrt()
            rows[:, 1].masked_fill_(~valid["b"][ids], float("nan"))
            expected.extend(rows.unbind())
            buffer.observe(squared, valid)
        torch.testing.assert_close(buffer.state_dict()["errors"], torch.stack(expected[-9:]), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
