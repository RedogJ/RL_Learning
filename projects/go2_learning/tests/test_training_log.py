"""检查奖励分项的样本加权、排序，以及原 Logger 清空数据后的打印。"""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from go2_learning.training.log import CompactTrainingLogger


class TrainingLogTests(unittest.TestCase):
    def test_weighted_penalties_are_ranked_before_original_logger_clears_extras(self):
        with TemporaryDirectory() as directory:
            original = SimpleNamespace(
                log_dir=directory, writer=object(), rewbuffer=[], lenbuffer=[], ep_extras=[
                    {"Reward/active_samples": 1, "Reward/alive": .03,
                     "Task/horizontal_speed_mps": .3, "Task/yaw_velocity_abs_radps": .5,
                     "Contact/foot_support_fraction": .75,
                     "Task/upright_alignment": .7, "Task/rear_support_count": 2., "Task/front_support_count": .5,
                     "Reward/height": -.01, "Reward/action_limit": -.1, "Reward/failure": 0.,
                     "Reward/foot_slide": -.04, "Reward/foot_time_variance": -.02},
                    {"Reward/active_samples": 3, "Reward/alive": .03,
                     "Task/horizontal_speed_mps": .5, "Task/yaw_velocity_abs_radps": .7,
                     "Contact/foot_support_fraction": 1.,
                     "Task/upright_alignment": .9, "Task/rear_support_count": 2., "Task/front_support_count": 0.,
                     "Reward/height": -.01, "Reward/action_limit": -.2, "Reward/failure": -1.,
                     "Reward/foot_slide": -.08, "Reward/foot_time_variance": -.06},
                    {"Task/failures_per_step": 0.},  # 全部环境在等待，不进入分母。
                ],
            )

            def log(**kwargs):
                original.ep_extras.clear()
                print("full details")

            original.log = log
            logger = CompactTrainingLogger(original, 1 / 30)
            values = logger._reward_breakdown()
            self.assertAlmostEqual(values["action_limit"], -.175)
            self.assertAlmostEqual(values["failure"], -.75)
            output = StringIO()
            with redirect_stdout(output):
                logger.log(it=0, start_it=0, total_it=1)
            self.assertIn("top=failure (71.1%)", output.getvalue())
            self.assertIn("slide=-0.0700", output.getvalue())
            self.assertIn("time_balance=-0.0500", output.getvalue())
            self.assertIn("speed=0.400m/s", output.getvalue())
            self.assertIn("spin=0.600rad/s", output.getvalue())
            self.assertIn("feet=3.50/4", output.getvalue())
            self.assertIn("upright=0.800 | rear=2.00/2 | front=0.25/2", output.getvalue())
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            self.assertNotIn("penalties/step", output.getvalue())
            self.assertEqual((Path(directory) / "training_details.log").read_text(), "full details\n")
            self.assertEqual(logger._reward_breakdown(), {})


if __name__ == "__main__":
    unittest.main()
