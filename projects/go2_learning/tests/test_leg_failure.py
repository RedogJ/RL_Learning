"""单腿断电：准确选关节、关闭真实驱动属性、切换时序与回放入口限制。"""

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from go2_learning.simulation.leg_failure import LegPowerCut
from go2_learning.tasks import TASKS, get_task


class Robot:
    def __init__(self):
        self.joint_names = list(reversed([f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf")
                                          for leg in ("FL", "FR", "RL", "RR")]))
        self.data = NS(joint_stiffness=NS(torch=torch.arange(12).float()[None] + 20),
                       joint_damping=NS(torch=torch.arange(12).float()[None] / 10 + .5),
                       joint_effort_limits=NS(torch=torch.full((1, 12), 23.7)),
                       joint_pos=NS(torch=torch.randn(1, 12)), joint_vel=NS(torch=torch.randn(1, 12)),
                       joint_viscous_friction_coeff=NS(torch=torch.full((1, 12), .15)),
                       joint_armature=NS(torch=torch.full((1, 12), .01)))
        self.writes = []

    def write_joint_stiffness_to_sim_index(self, *, stiffness, joint_ids):
        self.write("joint_stiffness", stiffness, joint_ids)

    def write_joint_damping_to_sim_index(self, *, damping, joint_ids):
        self.write("joint_damping", damping, joint_ids)

    def write_joint_effort_limit_to_sim_index(self, *, limits, joint_ids):
        self.write("joint_effort_limits", limits, joint_ids)

    def write(self, name, value, joint_ids):
        getattr(self.data, name).torch[:, joint_ids] = value
        self.writes.append((name, list(joint_ids)))


class LegFailureTests(unittest.TestCase):
    def test_power_cut_removes_drive_without_locking_or_changing_passive_parameters(self):
        robot = Robot()
        before = {name: value.torch.clone() for name, value in vars(robot.data).items()}
        fault = LegPowerCut(robot, "FL", 0., 1 / 30)
        with redirect_stdout(StringIO()):
            fault.before_control_step()
        self.assertEqual([robot.joint_names[i] for i in fault.joint_ids],
                         ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"])
        other_ids = [i for i in range(12) if i not in fault.joint_ids]
        for name, _, _ in LegPowerCut.PROPERTIES:
            value = getattr(robot.data, name).torch
            self.assertEqual(value[:, fault.joint_ids].abs().sum(), 0.)
            torch.testing.assert_close(value[:, other_ids], before[name][:, other_ids])
        for name in ("joint_pos", "joint_vel", "joint_viscous_friction_coeff", "joint_armature"):
            torch.testing.assert_close(getattr(robot.data, name).torch, before[name])
        # 位置目标即使继续变化，失效关节的隐式 PD 驱动力矩仍为 0。
        torque = robot.data.joint_stiffness.torch * 100 - robot.data.joint_damping.torch * 10
        torque = torque.clamp(-robot.data.joint_effort_limits.torch, robot.data.joint_effort_limits.torch)
        self.assertEqual(torque[:, fault.joint_ids].abs().sum(), 0.)
        self.assertTrue((torque[:, other_ids] > 0).all())
        for _ in range(100):
            fault.before_control_step()
        self.assertEqual(len(robot.writes), 3)  # 每步不反复写驱动属性。
        fault.close()
        for name, value in before.items():
            torch.testing.assert_close(getattr(robot.data, name).torch, value)

    def test_delay_and_episode_reset_restore_original_drives(self):
        robot = Robot()
        fault = LegPowerCut(robot, "FL", 3., 1 / 30)
        for _ in range(90):
            fault.before_control_step()
        self.assertFalse(fault.is_cut)
        self.assertEqual(robot.writes, [])
        output = StringIO()
        with redirect_stdout(output):
            fault.before_control_step()
        self.assertTrue(fault.is_cut)
        self.assertIn("3.00s", output.getvalue())
        fault.reset()
        self.assertFalse(fault.is_cut)
        self.assertEqual(fault.elapsed_steps, 0)
        self.assertEqual(len(robot.writes), 6)
        for _ in range(90):
            fault.before_control_step()
        self.assertEqual(len(robot.writes), 6)
        with redirect_stdout(StringIO()):
            fault.before_control_step()
        self.assertEqual(len(robot.writes), 9)
        fault.close()
        fault.close()
        self.assertEqual(len(robot.writes), 12)

    def test_non_integer_delay_does_not_trigger_early_and_invalid_arguments_fail(self):
        fault = LegPowerCut(Robot(), "RR", .035, .01)
        self.assertEqual(fault.delay_steps, 4)
        for delay, dt in ((-1., .01), (float("nan"), .01), (float("inf"), .01), (0., 0.)):
            with self.subTest(delay=delay, dt=dt), self.assertRaises(ValueError):
                LegPowerCut(Robot(), "FL", delay, dt)

    def test_cli_limits_fault_to_running_single_environment_play(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")

        def launcher_args(parser):
            parser.add_argument("--viz", dest="visualizer", type=lambda value: value.split(","))

        namespace = dict(argparse=argparse, TASKS=TASKS, get_task=get_task, __doc__="", add_launcher_args=launcher_args)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), namespace)
        base = ["go2", "--task", "running", "--control", "play", "--viz", "newton_gl", "--disable-leg", "FL"]
        with patch.object(sys, "argv", base):
            args = namespace["parse_args"]()
        self.assertEqual((args.disable_leg, args.disable_after, args.num_envs, args.keyboard), ("FL", 3., 1, True))
        for extra in (["--control", "train"], ["--control", "view"], ["--task", "standing"],
                      ["--task", "upright"], ["--num_envs", "2"], ["--disable-after", "-1"],
                      ["--disable-after", "nan"], ["--disable-after", "inf"]):
            with self.subTest(extra=extra), patch.object(sys, "argv", base + extra):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    namespace["parse_args"]()


if __name__ == "__main__":
    unittest.main()
