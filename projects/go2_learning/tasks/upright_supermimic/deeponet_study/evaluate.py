"""用独立采集的数据集对已训练的两种世界模型做相同轨迹评估。"""
import argparse
import json
from pathlib import Path

import torch
from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel
from go2_learning.tasks.upright_supermimic.deeponet_study.model import TimeDeepONet
from go2_learning.tasks.upright_supermimic.deeponet_study.train import result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', required=True, help='容器内独立测试数据集路径')
    p.add_argument('--eval-windows', type=int, default=512)
    p.add_argument('--seed', type=int, default=9251)
    p.add_argument('--viz', default='none', choices=['none'])
    args = p.parse_args()
    if args.eval_windows < 1:
        p.error('测试窗口数必须大于 0')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    data = torch.load(args.dataset, map_location='cpu', weights_only=False)
    if data.get('world_model_format') != 'deeponet_dataset_v1':
        raise ValueError('测试数据集格式不正确')
    if args.dataset in (saved['source'], saved['args'].get('validation_checkpoint')):
        raise ValueError('测试数据必须与训练和验证数据不同')
    train_seed = saved.get('training_collection_seed')
    val_seed = saved.get('validation_collection_seed')
    source_checkpoint = saved.get('collection_source_checkpoint')
    if train_seed is None or source_checkpoint is None:
        train_data = torch.load(saved['source'], map_location='cpu', weights_only=False)
        train_seed = train_data.get('collection_args', {}).get('seed')
        source_checkpoint = train_data.get('source_checkpoint')
        del train_data
    if val_seed is None and saved['args'].get('validation_checkpoint'):
        val_data = torch.load(saved['args']['validation_checkpoint'], map_location='cpu', weights_only=False)
        val_seed = val_data.get('collection_args', {}).get('seed')
        del val_data
    if data['collection_args']['seed'] in (train_seed, val_seed):
        raise ValueError('测试采集随机种子不能与训练或验证相同')
    if data['source_checkpoint'] != source_checkpoint:
        raise ValueError('测试数据需要来自相同源策略检查点')
    for name in ('input_mean', 'input_std', 'delta_mean', 'delta_std'):
        if not torch.equal(saved['mlp'][name], data['model'][name]) or not torch.equal(saved['deeponet'][name], data['model'][name]):
            raise ValueError('模型和测试数据归一化不同')
    device = torch.device('cuda:0')
    buf = PaperBuffer(data['args']['replay_steps'], data['args']['num_envs'], 90, 12, device)
    for key, value in data['buffer'].items():
        if torch.is_tensor(value):
            getattr(buf, key).copy_(value.to(device))
        else:
            setattr(buf, key, value)
    windows = buf.windows(args.eval_windows * 4, 32, torch.arange(buf.num_envs, device=device))
    chosen = (~windows[5].any(1)).nonzero().flatten()[:args.eval_windows]
    if len(chosen) < args.eval_windows:
        raise ValueError('非终止测试窗口不足')
    windows = tuple(v[chosen] for v in windows)
    mlp, deep = PaperWorldModel(90, 12).to(device), TimeDeepONet().to(device)
    mlp.load_state_dict(saved['mlp']); mlp.eval()
    deep.load_state_dict(saved['deeponet']); deep.eval()
    output = {'checkpoint': args.checkpoint, 'dataset': args.dataset, 'dataset_seed': data['collection_args']['seed'],
              'sample_seed': args.seed, 'windows': len(chosen), 'cycle': saved['cycle'],
              'note': 'fixed-action nonterminal windows; samples with replacement',
              'mlp': result(mlp, windows[0], windows[1], windows[2], windows[3], False),
              'deeponet': result(deep, windows[0], windows[1], windows[2], windows[3], True)}
    out = Path(args.checkpoint).with_name(f'independent_test_seed{output["dataset_seed"]}.json')
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print('独立测试结果：', out, flush=True)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
