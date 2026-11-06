"""导出保留仓库相对路径的轻量SuperMimic分享包；不会移动或删除原工程文件。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('projects/go2_mujoco_mimic')


def ignored(directory, names):
    excluded = {'logs', '.cache', '__pycache__', '.pytest_cache', 'diagnostics', 'history',
                'checkpoints', 'exports'}
    here = Path(directory)
    if here == ROOT / PROJECT:
        excluded |= {'projects', 'assets'}  # 早期路径错误产生的空目录，不属于工程资源。
    if here.name == 'videos':
        excluded |= {name for name in names if not (
            name.startswith('sideflip_latest_cycle5800_20261106.') or
            name.startswith('backflip_latest_cycle5250_20261106.'))}
    return [name for name in names if name in excluded or name.endswith(('.pyc', '.pyo', '.rgb'))]


def export(output, with_demo_policies=False, archive=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'输出目录已存在；指定一个新目录：{output}')
    if output.is_relative_to(ROOT / PROJECT) or output.is_relative_to(ROOT / 'assets'):
        raise ValueError('输出目录不能放在被复制的工程或assets内部')
    demos = None
    if with_demo_policies:
        checkpoint_dir = ROOT / PROJECT / 'checkpoints'
        demos = json.loads((checkpoint_dir / 'demo_policies.json').read_text())
        for info in demos['policies'].values():
            name = info['file']
            if Path(name).name != name:
                raise ValueError('演示manifest中的文件名必须是单个文件名')
            path = checkpoint_dir / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
                raise ValueError(f'演示策略校验失败：{name}')
    output.mkdir(parents=True)
    shutil.copytree(ROOT / PROJECT, output / PROJECT, ignore=ignored)
    asset = Path('assets/unitree_model/Go2/mjcf')
    shutil.copytree(ROOT / asset, output / asset)
    for relative in ('scripts/export_supermimic.py', 'tests/test_supermimic_sharing.py'):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    if demos:
        target = output / PROJECT / 'checkpoints'
        target.mkdir()
        for info in demos['policies'].values():
            shutil.copy2(ROOT / PROJECT / 'checkpoints' / info['file'], target / info['file'])
        shutil.copy2(ROOT / PROJECT / 'checkpoints/demo_policies.json', target / 'demo_policies.json')
    (output / 'run.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == supermimic ]]; then shift; fi
exec bash "$root/projects/go2_mujoco_mimic/run_gpu.sh" "$@"
''')
    (output / '.gitignore').write_text('__pycache__/\n*.pyc\n.cache/\nlogs/\ncheckpoints/\n')
    (output / 'README.md').write_text('''# Go2 SuperMimic · 组内分享

本包包含Go2 MuJoCo Warp世界模型训练代码、参考轨迹、Go2 MJCF模型和演示视频。
请先阅读[工程说明](projects/go2_mujoco_mimic/README.md)，再看[架构](projects/go2_mujoco_mimic/docs/architecture.md)与[环境](projects/go2_mujoco_mimic/docs/environment.md)。

所有命令从本包根目录执行，目录布局与原仓库一致。`bash run.sh supermimic ...` 与原入口相同。

```bash
# 有演示策略的分享包可直接回放；需要已准备好的Docker GPU环境
bash run.sh supermimic play --checkpoint /workspace/go2_mujoco_mimic/checkpoints/sideflip_policy_5800.pt --no-noise

# 从当前配置开始新的训练
bash projects/go2_mujoco_mimic/scripts/train_sideflip.sh --cycles 1000
```

本包不携带完整训练日志、优化器检查点、Warp缓存、Isaac Lab源码或历史诊断。
演示策略（如包含）只能回放；完整续训检查点需要单独传递。
文件校验记录在SHARE_MANIFEST.json，第三方来源记录在工程的THIRD_PARTY_NOTICES.md。
''')
    entries = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            entries.append({'path': path.relative_to(output).as_posix(), 'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'format': 'go2_supermimic_share_v1', 'with_demo_policies': with_demo_policies,
                'files': entries, 'total_bytes': sum(x['bytes'] for x in entries)}
    (output / 'SHARE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    archive_path = shutil.make_archive(str(output), 'gztar', root_dir=output.parent, base_dir=output.name) if archive else None
    return output, manifest, archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-demo-policies', action='store_true')
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    output, manifest, archive_path = export(args.output, args.with_demo_policies, args.archive)
    print(f'{output}: {len(manifest["files"])} files, {manifest["total_bytes"] / 1024**2:.1f} MiB')
    if archive_path:
        print(archive_path)


if __name__ == '__main__':
    main()
