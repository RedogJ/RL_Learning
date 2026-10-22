"""最佳模型不会被退化后的最终模型覆盖，最后状态仍可恢复。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from go2_learning.training.best_checkpoint import record_checkpoint


class BestCheckpointTests(unittest.TestCase):
    def test_improvement_and_regression_keep_best_and_last_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner = SimpleNamespace(_history_dir=root, current_learning_iteration=1,
                                     logger=SimpleNamespace(rewbuffer=[10., 12.]))
            first = root / 'model_1.pt'; first.write_bytes(b'good')
            record_checkpoint(runner, first)
            runner.current_learning_iteration = 2
            runner.logger.rewbuffer = [1., 2.]
            last = root / 'last.pt'; last.write_bytes(b'bad')
            record_checkpoint(runner, last)
            self.assertEqual((root / 'latest.pt').read_bytes(), b'good')
            self.assertEqual((root / 'best.pt').read_bytes(), b'good')
            self.assertEqual(last.read_bytes(), b'bad')
            self.assertEqual(json.loads((root / 'best.json').read_text())['score'], 11.)
            runner.logger.rewbuffer = [20.]
            newer = root / 'model_3.pt'; newer.write_bytes(b'better')
            record_checkpoint(runner, newer)
            self.assertEqual((root / 'latest.pt').read_bytes(), b'better')
            self.assertEqual(last.read_bytes(), b'better')
