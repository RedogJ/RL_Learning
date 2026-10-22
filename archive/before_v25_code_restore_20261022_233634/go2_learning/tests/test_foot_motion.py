"""足部运动奖励：触地门控、无截顶方差、物理步计时和重置隔离。"""

from dataclasses import fields
from types import SimpleNamespace as NS
import unittest

import torch

from go2_learning.simulation.contacts import Go2Contacts, REQUIRED_BODY_NAMES
from go2_learning.tasks import get_task
from go2_learning.tasks.running.rewards import foot_slide_penalty, foot_time_variance_penalty
from go2_learning.tests.test_reset_settling import load_env_class, Scene, Sim


class FootMotionTests(unittest.TestCase):
    def test_slip_counts_only_contacting_feet_and_world_horizontal_speed(self):
        state = {
            "foot_velocity_w": torch.tensor([[[3., 4., 99.], [90., 0., 0.],
                                               [0., 0., 50.], [0., -2., 0.]]]),
            "foot_contact": torch.tensor([[True, False, True, True]]),
        }
        torch.testing.assert_close(foot_slide_penalty(state), torch.tensor([7.]))
        state["foot_contact"].zero_()
        self.assertEqual(foot_slide_penalty(state).item(), 0.)

    def test_completed_phase_variance_has_no_half_second_cap(self):
        state = {
            "foot_last_air_time": torch.tensor([[.2, .2, .2, .2], [.2, .2, .2, 1.2],
                                                [.2, .2, .2, 2.2]]),
            "foot_last_contact_time": torch.ones(3, 4),
        }
        torch.testing.assert_close(foot_time_variance_penalty(state), torch.tensor([0., .25, 1.]))
        state["foot_last_contact_time"][1, 0] += 1.
        self.assertAlmostEqual(foot_time_variance_penalty(state)[1].item(), .5)

    def test_physics_history_records_short_landing_waiting_and_partial_reset(self):
        names = list(reversed(REQUIRED_BODY_NAMES))
        history = torch.zeros(2, 4, len(names), 3)
        sensor = NS(body_names=names, cfg=NS(history_length=4),
                    data=NS(net_normal_forces_w_history=NS(torch=history)))
        contacts = Go2Contacts(sensor, 1., 4, track_phase_times=True)
        contacts.update_air_time(.01, torch.tensor([True, False]))
        # 两个物理帧触地，再离地；控制步末已离地，仍必须记住这次短接触。
        history[:, 2:, names.index("FL_foot"), 0] = 2.
        contacts.update_air_time(.01, torch.tensor([True, False]))
        index = contacts.foot_ids.index(names.index("FL_foot"))
        result = contacts.read()
        self.assertAlmostEqual(result["foot_last_air_time"][0, index].item(), .05)
        self.assertAlmostEqual(result["foot_last_contact_time"][0, index].item(), .03)
        self.assertTrue(result["foot_contact"][0, index])  # 窗口内曾触地。
        self.assertEqual(result["foot_last_air_time"][1].sum(), 0.)
        before = result["foot_last_air_time"].clone()
        contacts.read()
        torch.testing.assert_close(contacts.phase_times.last_air, before)
        contacts.reset(torch.tensor([1]))
        torch.testing.assert_close(contacts.phase_times.last_air[0], before[0])
        contacts.reset(torch.tensor([0]))
        for value in (contacts.phase_times.air, contacts.phase_times.contact,
                      contacts.phase_times.last_air, contacts.phase_times.last_contact):
            self.assertEqual(value.sum(), 0.)

    def test_running_reads_foot_names_correctly_and_excludes_settling(self):
        task = get_task("running")
        # 只有新增的奖励需要接触传感器时，也必须正常启用。
        for name in ("undesired_contact_weight", "foot_contact_ratio_weight", "foot_force_balance_weight",
                     "all_feet_airborne_weight", "foot_air_time_weight"):
            setattr(task.cfg.rewards, name, 0.)
        self.assertTrue(task.needs_contact_sensor)
        scene = Scene()
        scene.contact_forces.body_names.reverse()
        env = load_env_class()(Sim(scene), scene, task=task)
        # 数据与传感器使用相反顺序，每只脚速度按机器人自身名字设置。
        for leg, speed in zip(("FL", "FR", "RL", "RR"), (1., 2., 3., 4.)):
            robot_id = scene.robot.body_names.index(f"{leg}_foot")
            scene.robot.data.body_com_lin_vel_w.torch[:, robot_id, 0] = speed
            sensor_id = scene.contact_forces.body_names.index(f"{leg}_foot")
            scene.contact_forces.forces[:, sensor_id, 2] = 10.
        _, rewards, _, extras = env.step(env.standing_actions)
        self.assertEqual(rewards.sum(), 0.)
        self.assertNotIn("Reward/foot_slide", extras["log"])
        env.finish_initial_settling()
        self.assertEqual(env.contacts.phase_times.contact.sum(), 0.)
        _, rewards, _, extras = env.step(env.standing_actions)
        expected = -task.cfg.rewards.foot_slide_weight * 10 * env.dt
        self.assertAlmostEqual(extras["log"]["Reward/foot_slide"].item(), expected, places=5)
        self.assertIn("Reward/foot_time_variance", extras["log"])
        torch.testing.assert_close(rewards, sum(env._reward_components.values()))

    def test_new_constraints_survive_positive_floor_and_can_be_disabled(self):
        task = get_task("running")
        task.cfg.rewards.foot_slide_weight = 20.
        task.cfg.rewards.foot_time_variance_weight = 200.
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        state = env._read_state()
        state["foot_contact"] = torch.ones(3, 4, dtype=torch.bool)
        state["foot_velocity_w"] = torch.ones(3, 4, 3)
        state["foot_last_air_time"] = torch.tensor([[0., 0., 0., 2.]]).repeat(3, 1)
        # 常规奖励人为压成负值，验证新增惩罚仍在下限修正之后完整扣除。
        state["joint_offset"] = torch.full((3, 12), 100.)
        terms = task.reward_components(state, env.standing_actions, torch.zeros(3, dtype=torch.bool), env.dt)
        self.assertTrue((terms["positive_floor"] > 0).all())
        torch.testing.assert_close(terms["foot_slide"], torch.full((3,), -20 * 4 * 2**.5 * env.dt))
        torch.testing.assert_close(terms["foot_time_variance"], torch.full((3,), -200 * env.dt))
        task.cfg.rewards.foot_slide_weight = task.cfg.rewards.foot_time_variance_weight = 0.
        for name in ("foot_contact", "foot_velocity_w", "foot_last_air_time", "foot_last_contact_time"):
            del state[name]
        terms = task.reward_components(state, env.standing_actions, torch.zeros(3, dtype=torch.bool), env.dt)
        self.assertEqual(terms["foot_slide"].sum() + terms["foot_time_variance"].sum(), 0.)

    def test_legacy_playback_disables_new_rewards_and_new_runs_save_them(self):
        task = get_task("running")
        saved = task.run_config(False)
        self.assertEqual(saved["reward_cfg"]["foot_slide_weight"], task.cfg.rewards.foot_slide_weight)
        restored = get_task("running")
        restored.load_playback_config(saved)
        self.assertEqual(restored.run_config(False), saved)
        saved["reward_cfg"].pop("foot_slide_weight")
        saved["reward_cfg"].pop("foot_time_variance_weight")
        restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.rewards.foot_slide_weight, 0.)
        self.assertEqual(restored.cfg.rewards.foot_time_variance_weight, 0.)
        cfg_type = type(task.cfg.rewards)
        self.assertFalse(any("max_time" in field.name for field in fields(cfg_type)))
        for key in ("foot_slide_weight", "foot_time_variance_weight"):
            for value in (-1., float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    cfg_type(**{key: value})


if __name__ == "__main__":
    unittest.main()
