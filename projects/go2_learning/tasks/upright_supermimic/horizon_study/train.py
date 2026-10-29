"""Resume one checkpoint with a different actor imagination horizon.

All training logic is inherited from the paper-aligned entry. Only the
resume guard and output directory are changed for this experiment.
"""
from pathlib import Path

from go2_learning.tasks.upright_supermimic import paper_aligned_train


source = Path(paper_aligned_train.__file__).read_text()
old_guard = "('num_envs','replay_steps','wm_horizon','horizon','collect_steps')"
new_guard = "('num_envs','replay_steps','wm_horizon','collect_steps')"
old_path = "LOG_ROOT/'upright_supermimic'/'paper_aligned'/name"
new_path = "LOG_ROOT/'upright_supermimic'/'horizon_study'/name"
for old, new in ((old_guard, new_guard), (old_path, new_path)):
    if source.count(old) != 1:
        raise RuntimeError(f'实验入口的预期代码未找到或不唯一: {old}')
    source = source.replace(old, new)
exec(compile(source, str(paper_aligned_train.__file__), 'exec'), {'__name__': '__main__'})
