"""使用真实 RSL-RL 验证等待样本过滤，物理侧使用 CPU 替身。"""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from go2_learning.tasks.standing.agent_cfg import make_agent_cfg
from go2_learning.training.settling import SettlingLogger, SettlingPPO
from go2_learning.training.log import CompactTrainingLogger
from go2_learning.tests.test_reset_settling import Scene, Sim, load_env_class

try:
    from tensordict import TensorDict
    from rsl_rl.runners import OnPolicyRunner
except ImportError:
    OnPolicyRunner = None


@unittest.skipIf(OnPolicyRunner is None, "需要 RSL-RL 5.5.1 和 TensorDict")
class SettlingTrainingTests(unittest.TestCase):
    def make_runner(self):
        env_cls = load_env_class()
        original_observations = env_cls.get_observations

        def observations(self):
            return TensorDict(original_observations(self), batch_size=[self.num_envs])

        env_cls.get_observations = observations
        scene = Scene()
        env = env_cls(Sim(scene), scene)
        env.finish_initial_settling()
        cfg = make_agent_cfg()
        cfg["actor"]["hidden_dims"] = [8]
        cfg["critic"]["hidden_dims"] = [8]
        cfg["algorithm"]["num_learning_epochs"] = 2
        runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
        runner.alg = SettlingPPO(runner.alg, env)
        return runner, scene

    def test_real_ppo_excludes_waiting_from_inference_normalization_and_updates(self):
        runner, scene = self.make_runner()
        env, adapter = runner.env, runner.alg
        actor_calls = []
        adapter.actor.register_forward_pre_hook(lambda module, inputs: actor_calls.append(inputs[0].batch_size[0]))
        logger = runner.logger
        logger.writer = object()  # process_env_step 不调用 writer 方法。
        filtered_logger = SettlingLogger(logger)
        obs = env.get_observations()
        expected_valid = []
        with torch.inference_mode():
            for step in range(24):
                active = env.policy_active.clone()
                calls_before = len(actor_calls)
                actions = adapter.act(obs)
                self.assertEqual(len(actor_calls) - calls_before, int(active.any()))
                if active.any():
                    self.assertEqual(actor_calls[-1], int(active.sum()))
                # 第一控制步令所有狗失败，随后整段等待中不应再进行网络推理。
                if step == 0:
                    scene.robot.data.projected_gravity_b.torch[:] = torch.tensor([1., 0., 0.])
                elif step == 16:
                    scene.robot.data.projected_gravity_b.torch[1] = torch.tensor([1., 0., 0.])
                obs, rewards, dones, extras = env.step(actions)
                scene.robot.data.projected_gravity_b.torch[:] = torch.tensor([0., 0., -1.])
                expected_valid.append(active)
                adapter.process_env_step(obs, rewards, dones, extras)
                filtered_logger.process_env_step(rewards, dones, extras)
            adapter.compute_returns(obs)
        expected_valid = torch.stack(expected_valid)
        torch.testing.assert_close(adapter.valid, expected_valid)
        # 第一步终止，收益不能包含下一回合的奖励或等待阶段的价值预测。
        st = adapter.storage
        torch.testing.assert_close(st.returns[0], st.rewards[0])
        filtered_logger.flush_episodes()
        self.assertEqual(logger.lenbuffer[0], 1.)
        self.assertEqual(logger.cur_episode_length[0], 14.)
        self.assertEqual(logger.cur_episode_length[1], 0.)
        normalization_counts = []
        for model in (adapter.actor, adapter.critic):
            original = model.update_normalization

            def update_normalization(obs, original=original):
                normalization_counts.append(obs.batch_size[0])
                return original(obs)

            model.update_normalization = update_normalization
        before = [p.detach().clone() for p in adapter.actor.parameters()]
        losses = adapter.update()
        count = int(expected_valid.sum())
        self.assertEqual(normalization_counts, [count, count])
        self.assertEqual(actor_calls[-2:], [count, count])
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, adapter.actor.parameters())))
        self.assertEqual(adapter.storage.step, 0)

    def test_unmodified_runner_can_train_with_adapter(self):
        runner, _ = self.make_runner()
        runner.learn(num_learning_iterations=2)
        self.assertEqual(runner.current_learning_iteration, 1)
        self.assertEqual(runner.alg.storage.step, 0)
        self.assertIn("optimizer_state_dict", runner.alg.save())

    def test_compact_logger_flushes_before_reading_completed_episodes(self):
        runner, _ = self.make_runner()
        logger = runner.logger
        logger.writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None)
        with TemporaryDirectory() as directory:
            logger.log_dir = directory
            batched = SettlingLogger(logger)
            compact = CompactTrainingLogger(batched, runner.env.dt)
            rewards = torch.tensor([2., 3., 4.])
            batched.process_env_step(rewards, torch.tensor([True, False, True]), {
                "policy_active": torch.tensor([True, False, True]),
                "log": {"Task/failures_per_step": torch.tensor(2.)},
            })
            output = StringIO()
            with redirect_stdout(output):
                compact.log(it=0, start_it=0, total_it=1, collect_time=.1, learn_time=.01,
                            loss_dict={"value": 0.}, learning_rate=3e-4,
                            action_std=torch.ones(12), rnd_weight=None)
            self.assertIn("reward=3.000", output.getvalue())
            self.assertIn("fallen=2 (total=2)", output.getvalue())
            self.assertEqual(list(logger.lenbuffer), [1., 1.])
            self.assertFalse(batched._pending_episodes)


if __name__ == "__main__":
    unittest.main()
