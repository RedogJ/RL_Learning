"""从完整训练检查点导出Actor和回放配置；不会改写原检查点。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'sideflip': 'logs/sideflip_v5_continue_2026-11-05_15-48-02_035736/model_5800.pt',
    'backflip': 'logs/backflip_v23_landing_refine_2026-11-05_12-26-04_609466/model_5250.pt',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'checkpoints')
    for task in DEFAULTS:
        parser.add_argument(f'--{task}', type=Path, default=ROOT / DEFAULTS[task])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'go2_demo_policies_v1', 'purpose': 'playback_only', 'policies': {}}
    for task in DEFAULTS:
        source = getattr(args, task)
        saved = torch.load(source, map_location='cpu', weights_only=False)
        if saved.get('format') != 'go2_core_normalized_v3':
            raise ValueError(f'需要当前完整训练格式：{source}')
        output = args.output_dir / f'{task}_policy_{saved["cycle"]}.pt'
        if output.exists():
            raise FileExistsError(f'输出已存在，使用新的 --output-dir：{output}')
        payload = {'format': 'go2_policy_export_v1', 'cycle': saved['cycle'],
                   'env': saved['env'], 'motion': saved['motion'], 'actor': saved['actor'],
                   'task': task, 'purpose': 'playback_only'}
        torch.save(payload, output)
        manifest['policies'][task] = {'file': output.name, 'cycle': saved['cycle'],
            'source_checkpoint': str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
            'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
        print(f'{task}: {output.name} ({output.stat().st_size / 1024**2:.1f} MiB)', flush=True)
        del saved, payload
    (args.output_dir / 'demo_policies.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
