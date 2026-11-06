"""Critic 输入、旧权重迁移及 Actor/价值回归的梯度边界回归检查。"""
import unittest
from types import SimpleNamespace

import torch

from go2_mujoco_mimic.tasks.mimic.state_model import StateValue, critic_remaining_time
from go2_mujoco_mimic.tasks.mimic.state_train import fit_critic, reference_critic_mask


class CriticInputsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(120)
        torch.set_num_threads(1)
        self.s = torch.randn(4, 37)
        self.g = torch.randn(4, 190)
        self.ctx = torch.randn(4, 16)
        self.h = torch.randn(4, 10, 42)
        self.t = torch.linspace(.1, 1., 4)[:, None]

    def test_old_checkpoint_migration_preserves_values_and_source(self):
        old = StateValue()
        weights = old.state_dict()
        original = weights['net.0.weight'].clone()
        new = StateValue(history_length=10)
        self.assertTrue(new.load_for_resume(weights))
        torch.testing.assert_close(new(self.s, self.g, self.ctx, self.h, self.t),
                                   old(self.s, self.g, self.ctx), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(weights['net.0.weight'], original)
        self.assertEqual(weights['net.0.weight'].shape[1], 243)
        self.assertEqual(new.input_dim, 664)
        restored = StateValue(**new.config)
        self.assertFalse(restored.load_for_resume(new.state_dict()))
        torch.testing.assert_close(restored(self.s, self.g, self.ctx, self.h, self.t),
                                   new(self.s, self.g, self.ctx, self.h, self.t))

    def test_new_critic_does_not_silently_drop_missing_history(self):
        value = StateValue(history_length=10)
        with self.assertRaises(ValueError):
            value(self.s, self.g, self.ctx)
        with self.assertRaises(ValueError):
            value(self.s, self.g, self.ctx, self.h[:, :9], self.t)

    def test_frozen_value_still_backpropagates_through_actor_history(self):
        value = StateValue(history_length=10).requires_grad_(False)
        state = self.s.clone().requires_grad_()
        previous_action = torch.randn(4, 12, requires_grad=True)
        last_observation = torch.cat((self.h[:, -1, :30], previous_action), -1)
        history = torch.cat((self.h[:, :-1], last_observation[:, None]), 1)
        value(state, self.g, self.ctx, history, self.t).sum().backward()
        self.assertGreater(state.grad.abs().sum().item(), 0)
        self.assertGreater(previous_action.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in value.parameters()))

    def test_regression_fit_detaches_policy_and_fixed_targets(self):
        value = StateValue(history_length=10)
        opt = torch.optim.Adam(value.parameters(), lr=1e-3)
        state = self.s.clone().requires_grad_()
        history = self.h.clone().requires_grad_()
        targets = torch.randn(4, requires_grad=True)
        result = fit_critic(value, opt, state, self.g, self.ctx, targets, 2, 2,
                            histories=history, remaining_times=self.t)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in result))
        self.assertIsNone(state.grad)
        self.assertIsNone(history.grad)
        self.assertIsNone(targets.grad)

    def test_remaining_time_distinguishes_alias_and_preserves_end_mask(self):
        env = SimpleNamespace(ref=SimpleNamespace(frames=range(54)),
                              cfg=SimpleNamespace(task=SimpleNamespace(reference_cycles=1)))
        frame = torch.tensor([42, 52, 53, 54])
        limit = torch.full_like(frame, 53)
        remaining = critic_remaining_time(env, frame, limit)
        torch.testing.assert_close(remaining[:, 0], torch.tensor([11/53, 1/53, 0, 0]))
        self.assertEqual(reference_critic_mask(frame, limit).tolist(), [True, True, False, False])
        # 相同动作阶段、不同终点（连续拼接参考）也必须能区分。
        self.assertGreater(critic_remaining_time(env, frame[:1], limit[:1]+10).item(), remaining[0].item())


if __name__ == '__main__':
    unittest.main()
