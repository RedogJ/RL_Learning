"""同一次训练中按最近完成回合的平均回报保留最佳已保存检查点。"""

import json
import math
from pathlib import Path
import shutil
import statistics


def _copy_atomic(source, destination):
    temporary = destination.with_name(destination.name + '.tmp')
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def update_best_checkpoint(directory, checkpoint, score, iteration):
    directory, checkpoint = Path(directory), Path(checkpoint)
    if score is None or not math.isfinite(score):
        return False
    metadata = directory / 'best.json'
    previous = json.loads(metadata.read_text()) if metadata.exists() else None
    if previous is not None and score <= previous['score']:
        return False
    _copy_atomic(checkpoint, directory / 'best.pt')
    _copy_atomic(checkpoint, directory / 'latest.pt')
    info = {'metric': 'mean_completed_episode_return', 'score': score,
            'iteration': iteration, 'source_checkpoint': checkpoint.name,
            'scope': 'within_run_saved_checkpoints',
            'note': '训练最近完成回合平均回报；含探索噪声，不代表独立评估最优。'}
    temporary = metadata.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    temporary.replace(metadata)
    return True


def record_checkpoint(runner, path):
    directory, path = Path(runner._history_dir), Path(path)
    # last 始终保留最后保存的权重；latest/best 只在评分提升时替换。
    if path.name != 'last.pt':
        _copy_atomic(path, directory / 'last.pt')
    buffer = getattr(getattr(runner, 'logger', None), 'rewbuffer', None)
    score = statistics.mean(buffer) if buffer is not None and len(buffer) else None
    update_best_checkpoint(directory, path, score, runner.current_learning_iteration)
    if not (directory / 'latest.pt').exists():
        _copy_atomic(path, directory / 'latest.pt')
