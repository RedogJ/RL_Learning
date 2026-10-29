"""冻结 replay，比较 MLP 与时间条件 DeepONet；不会更新 Actor 或原任务。"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel
from go2_learning.tasks.upright_supermimic.deeponet_study.model import TimeDeepONet


def result(model, states, actions, contexts, actual, is_deeponet):
    predicted = states[:, 0]
    rows = {}
    with torch.no_grad():
        for k in range(actions.shape[1]):
            if is_deeponet:
                if k % 8 == 0:
                    start = predicted
                    block_actions = actions[:, k:k + 8]
                predicted, _ = model(start, block_actions, contexts[:, k], k % 8 + 1)
            else:
                predicted, _ = model(predicted, actions[:, k], contexts[:, k])
            if k + 1 in (1, 8, 16, 32):
                err = predicted - actual[:, k]
                norm = err / model.delta_std
                rows[str(k + 1)] = {
                    'joint_rmse_rad': float(err[:, 9:21].square().mean().sqrt()),
                    'gravity_rmse': float(err[:, 6:9].square().mean().sqrt()),
                    'normalized_state_mse': float(torch.cat((norm[:, :80], norm[:, 88:]), -1).square().mean()),
                }
    return rows


def deeponet_update(model, optimizer, windows):
    states, actions, contexts, targets, fail, _, _ = windows
    # 每个时间点分别反传，再统一做一次参数更新；等价于八个损失的均值。
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for step in range(1, 9):
        pred, logits = model(states[:, 0], actions, contexts[:, 0], step)
        diff = (pred - targets[:, step - 1]) / model.delta_std
        dynamics = torch.cat((diff[:, :80], diff[:, 88:]), -1).square().mean()
        terminal = F.binary_cross_entropy_with_logits(logits, fail[:, step - 1].float())
        loss = (dynamics + terminal) / 8
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total


def mlp_update(model, optimizer, windows):
    states, actions, contexts, targets, fail, _, _ = windows
    predicted = states[:, 0]
    loss = 0.0
    for step in range(8):
        predicted, logits = model(predicted, actions[:, step], contexts[:, step])
        diff = (predicted - targets[:, step]) / model.delta_std
        loss = loss + (torch.cat((diff[:, :80], diff[:, 88:]), -1).square().mean()
                       + F.binary_cross_entropy_with_logits(logits, fail[:, step].float())) / 8
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--validation-checkpoint', default=None, help='独立采集的数据集容器路径')
    p.add_argument('--cycles', type=int, default=10)
    p.add_argument('--groups-per-cycle', type=int, default=10)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--holdout-envs', type=int, default=8)
    p.add_argument('--eval-windows', type=int, default=512)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--viz', default='none', choices=['none'])
    args = p.parse_args()
    if min(args.cycles, args.groups_per_cycle, args.batch, args.holdout_envs, args.eval_windows, args.save_interval) < 1:
        p.error('数值必须大于 0')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved.get('world_model_format') not in ('paper_context_v3', 'deeponet_dataset_v1'):
        raise ValueError('需要 paper_context_v3 检查点或 deeponet_dataset_v1 数据集')
    n, capacity = saved['args']['num_envs'], saved['args']['replay_steps']
    if not args.validation_checkpoint and args.holdout_envs >= n:
        p.error('必须留有训练环境')
    device = torch.device('cuda:0')
    buf = PaperBuffer(capacity, n, 90, 12, device)
    for key, value in saved['buffer'].items():
        if torch.is_tensor(value):
            getattr(buf, key).copy_(value.to(device))
        else:
            setattr(buf, key, value)
    if args.validation_checkpoint:
        if saved.get('world_model_format') != 'deeponet_dataset_v1':
            raise ValueError('使用独立验证时，训练数据也必须是 deeponet_dataset_v1')
        validation = torch.load(args.validation_checkpoint, map_location='cpu', weights_only=False)
        if validation.get('world_model_format') != 'deeponet_dataset_v1':
            raise ValueError('独立验证需要 deeponet_dataset_v1 数据集')
        if validation.get('source_checkpoint') != saved.get('source_checkpoint'):
            raise ValueError('训练与验证数据集必须使用同一源策略检查点')
        if validation['collection_args']['seed'] == saved['collection_args']['seed']:
            raise ValueError('训练和验证采集 seed 必须不同')
        for name in ('input_mean', 'input_std', 'delta_mean', 'delta_std'):
            if not torch.equal(validation['model'][name], saved['model'][name]):
                raise ValueError('训练与验证的归一化尺度不一致')
        vn, vc = validation['args']['num_envs'], validation['args']['replay_steps']
        val_buf = PaperBuffer(vc, vn, 90, 12, device)
        for key, value in validation['buffer'].items():
            if torch.is_tensor(value):
                getattr(val_buf, key).copy_(value.to(device))
            else:
                setattr(val_buf, key, value)
        train_ids = torch.arange(n, device=device)
        val_ids = torch.arange(vn, device=device)
        validation_description = 'separate Isaac rollout and seed; fixed nonterminal windows sampled with replacement'
    else:
        ids = torch.randperm(n, device=device)
        val_ids, train_ids = ids[:args.holdout_envs], ids[args.holdout_envs:]
        val_buf = buf
        validation_description = 'held-out env IDs within same saved replay; no new simulation'
    candidates = val_buf.windows(args.eval_windows * 4, 32, val_ids)
    valid = (~candidates[5].any(1)).nonzero().flatten()[:args.eval_windows]
    if len(valid) < args.eval_windows:
        raise RuntimeError(f'留出环境不足 {args.eval_windows} 个无终止的采样窗口')
    val = tuple(v[valid] for v in candidates)
    mlp, deep = PaperWorldModel(90, 12).to(device), TimeDeepONet().to(device)
    for model in (mlp, deep):
        for name in ('input_mean', 'input_std', 'delta_mean', 'delta_std'):
            getattr(model, name).copy_(saved['model'][name].to(device))
    opt_m, opt_d = torch.optim.Adam(mlp.parameters(), lr=2e-4), torch.optim.Adam(deep.parameters(), lr=2e-4)
    run = LOG_ROOT / 'upright_supermimic' / 'deeponet_study' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    config = {'args': vars(args), 'source_cycle': saved['cycle'], 'train_env_ids': train_ids.tolist(),
              'holdout_env_ids': val_ids.tolist(), 'mlp_params': sum(p.numel() for p in mlp.parameters()),
              'deeponet_params': sum(p.numel() for p in deep.parameters()),
              'sampling': f'two {args.batch}-window batches sampled together, separate sequential optimizer steps',
              'validation': validation_description,
              'note': 'time-query DeepONet directly predicts 1..8 steps, not integrated into online actor training'}
    (run / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print('DeepONet 对照输出：', run, flush=True)
    print('参数量：', config['mlp_params'], config['deeponet_params'], flush=True)
    for cycle in range(args.cycles):
        sums = [0.0, 0.0]
        for _ in range(args.groups_per_cycle):
            grouped = buf.windows(args.batch * 2, 8, train_ids)
            for batch in range(2):
                sl = slice(batch * args.batch, (batch + 1) * args.batch)
                windows = tuple(v[sl] for v in grouped)
                sums[0] += mlp_update(mlp, opt_m, windows)
                sums[1] += deeponet_update(deep, opt_d, windows)
        metrics = {'cycle': cycle + 1, 'updates_per_model': (cycle + 1) * args.groups_per_cycle * 2,
                   'mlp_train_loss': sums[0] / (args.groups_per_cycle * 2),
                   'deeponet_train_loss': sums[1] / (args.groups_per_cycle * 2),
                   'holdout_mlp': result(mlp, val[0], val[1], val[2], val[3], False),
                   'holdout_deeponet': result(deep, val[0], val[1], val[2], val[3], True)}
        with (run / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if (cycle + 1) % args.save_interval == 0 or cycle + 1 == args.cycles:
            tmp = run / 'checkpoint.tmp'
            torch.save({'mlp': mlp.state_dict(), 'deeponet': deep.state_dict(),
                        'optimizer_mlp': opt_m.state_dict(), 'optimizer_deeponet': opt_d.state_dict(),
                        'args': vars(args), 'cycle': cycle + 1, 'metrics': metrics,
                        'source': args.checkpoint, 'rng': torch.get_rng_state(),
                        'training_collection_seed': saved.get('collection_args', {}).get('seed'),
                        'validation_collection_seed': validation.get('collection_args', {}).get('seed') if args.validation_checkpoint else None,
                        'collection_source_checkpoint': saved.get('source_checkpoint'),
                        'cuda_rng': torch.cuda.get_rng_state_all()}, tmp)
            os.replace(tmp, run / 'latest.pt')


if __name__ == '__main__':
    main()
