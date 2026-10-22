"""固定模型副本、SHA256 校验及运行配置校验。"""
import hashlib
import json
from pathlib import Path
import shutil
from go2_learning.paths import LOG_ROOT

ROOT = LOG_ROOT / 'mimic'
MANIFEST = Path(__file__).with_name('experts.json')
UPRIGHT = 'upright/2026-10-22_21-21-35_342854_upright_restart_v25/latest.pt'
# 用户确认使用连续训练线的最新编号，不选日期较新的从零训练分支。
RUNNING = 'running/2026-10-17_15-07-50_097333_running_18/latest.pt'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin_experts():
    if MANIFEST.exists():
        return load_experts()  # 专家固定后不静默替换
    sources = {'upright': LOG_ROOT / UPRIGHT,
               'running': LOG_ROOT / RUNNING}
    entries = {}
    for name, path in sources.items():
        cfg = json.loads(path.with_name('run.json').read_text())
        if cfg['task_name'] != name:
            raise ValueError(f'{path} 不是 {name} 专家')
        folder = ROOT / 'experts' / name
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, folder / 'expert.pt')
        shutil.copy2(path.with_name('run.json'), folder / 'run.json')
        entries[name] = {'checkpoint': str((folder / 'expert.pt').relative_to(LOG_ROOT)),
                         'source': str(path.relative_to(LOG_ROOT)),
                         'sha256': sha256(path), 'config_sha256': sha256(folder / 'run.json')}
    # 观测和动作维数一致仍不代表关节映射一致，显式核对。
    configs = [json.loads((ROOT / 'experts' / n / 'run.json').read_text()) for n in sources]
    for key in ('joint_names', 'default_joint_pos', 'action_joint_limits', 'env_cfg'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'两个专家的 {key} 不一致，不能共用学生')
    MANIFEST.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    return load_experts()


def load_experts():
    entries = json.loads(MANIFEST.read_text())
    for name, entry in entries.items():
        path = LOG_ROOT / entry['checkpoint']
        if sha256(path) != entry['sha256'] or sha256(path.with_name('run.json')) != entry['config_sha256']:
            raise ValueError(f'{name} 专家文件被修改，停止运行')
    return entries


def expert_path(name):
    return LOG_ROOT / load_experts()[name]['checkpoint']


def validate_runtime(env, saved):
    """检查专家实际看到的参考角、关节顺序和动作映射，而不只比较两份 JSON。"""
    import torch
    if list(env.robot.joint_names) != saved['joint_names']:
        raise ValueError('当前仿真关节顺序与专家不一致')
    for key, actual in (('default_joint_pos', env.default_joint_pos[0]),
                        ('action_joint_limits', env.action_joint_limits[0])):
        expected = torch.tensor(saved[key], device=actual.device, dtype=actual.dtype)
        if expected.shape != actual.shape or not torch.allclose(expected, actual, atol=1e-6, rtol=0.):
            raise ValueError(f'当前 {key} 与专家不一致，禁止继续采集')
    expected_dt = saved['env_cfg']['physics_dt'] * saved['env_cfg']['decimation']
    if abs(env.dt-expected_dt)>1e-8:
        raise ValueError('当前控制周期与专家不一致')
