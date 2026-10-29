"""同一独立数据集上从头训练十帧历史 MLP 与 DeepONet。"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.history_study.model import HISTORY, HORIZON, HistoryMLP, HistoryDeepONet
from go2_learning.tasks.upright_supermimic.history_study.sampler import WindowSampler


def load_dataset(path, device):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if data.get('world_model_format') != 'deeponet_dataset_v1':
        raise ValueError('需要独立采集的 deeponet_dataset_v1 数据集')
    buf = PaperBuffer(data['args']['replay_steps'], data['args']['num_envs'], 90, 12, device)
    for key, value in data['buffer'].items():
        if torch.is_tensor(value):
            getattr(buf, key).copy_(value.to(device))
        else:
            setattr(buf, key, value)
    return data, buf


def unpack(windows):
    states, actions, contexts, targets, fail, _, _ = windows
    return (states[:, :HISTORY], actions[:, HISTORY-1:], contexts[:, HISTORY-1:],
            targets[:, HISTORY-1:], fail[:, HISTORY-1:])


def append_history(history, state):
    return torch.cat((history[:, 1:], state[:, None]), dim=1)


def dynamics_loss(predicted, target, model):
    diff = (predicted - target) / model.delta_std
    return torch.cat((diff[:, :80], diff[:, 88:]), -1).square().mean()


def update_mlp(model, optimizer, windows):
    history, actions, contexts, targets, fail = unpack(windows)
    loss = 0.0
    for k in range(HORIZON):
        predicted, logits = model(history, actions[:, k], contexts[:, k])
        history = append_history(history, predicted)
        loss = loss + (dynamics_loss(predicted, targets[:, k], model)
                       + F.binary_cross_entropy_with_logits(logits, fail[:, k].float())) / HORIZON
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def update_deep(model, optimizer, windows):
    history, actions, contexts, targets, fail = unpack(windows)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for k in range(HORIZON):
        predicted, logits = model(history, actions, contexts[:, 0], k + 1)
        loss = (dynamics_loss(predicted, targets[:, k], model)
                + F.binary_cross_entropy_with_logits(logits, fail[:, k].float())) / HORIZON
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total


def evaluate(model, windows, is_deep, uses_history=True):
    history, actions, contexts, targets, _ = unpack(windows)
    rows = {}
    with torch.no_grad():
        for k in range(actions.shape[1]):
            if is_deep:
                if k % HORIZON == 0:
                    block_history = history
                    block_actions = actions[:, k:k + HORIZON]
                model_input = block_history if uses_history else block_history[:, -1]
                predicted, _ = model(model_input, block_actions, contexts[:, k], k % HORIZON + 1)
            else:
                model_input = history if uses_history else history[:, -1]
                predicted, _ = model(model_input, actions[:, k], contexts[:, k])
            history = append_history(history, predicted)
            if k + 1 in (1, 8, 16, 32):
                err = predicted - targets[:, k]
                norm = err / model.delta_std
                rows[str(k + 1)] = {'joint_rmse_rad': float(err[:, 9:21].square().mean().sqrt()),
                                    'gravity_rmse': float(err[:, 6:9].square().mean().sqrt()),
                                    'normalized_state_mse': float(torch.cat((norm[:, :80], norm[:, 88:]), -1).square().mean())}
    return rows


def verify_pair(a, b):
    if a['source_checkpoint'] != b['source_checkpoint']:
        raise ValueError('数据应来自同一策略检查点')
    if a['collection_args']['seed'] == b['collection_args']['seed']:
        raise ValueError('采集数据随机种子必须不同')
    if abs(a.get('control_dt', -1) - b.get('control_dt', -2)) > 1e-9:
        raise ValueError('控制周期不一致')
    for name in ('input_mean', 'input_std', 'delta_mean', 'delta_std'):
        if not torch.equal(a['model'][name], b['model'][name]):
            raise ValueError('归一化不一致：' + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='训练数据集')
    p.add_argument('--validation-checkpoint', required=True, help='独立验证数据集，容器路径')
    p.add_argument('--cycles', type=int, default=100)
    p.add_argument('--groups-per-cycle', type=int, default=10)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--eval-windows', type=int, default=512)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--viz', choices=['none'], default='none')
    args = p.parse_args()
    if min(args.cycles, args.groups_per_cycle, args.batch, args.eval_windows, args.save_interval) < 1:
        p.error('数量必须为正')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda:0')
    data, buf = load_dataset(args.checkpoint, device)
    validation, val_buf = load_dataset(args.validation_checkpoint, device)
    verify_pair(data, validation)
    train_sampler = WindowSampler(buf, HISTORY - 1 + HORIZON, torch.arange(buf.num_envs, device=device))
    val_sampler = WindowSampler(val_buf, HISTORY - 1 + 32, torch.arange(val_buf.num_envs, device=device), allow_last_done=False)
    val = val_sampler.sample(args.eval_windows)
    mlp, deep = HistoryMLP().to(device), HistoryDeepONet().to(device)
    for model in (mlp, deep):
        for name in ('input_mean', 'input_std', 'delta_mean', 'delta_std'):
            getattr(model, name).copy_(data['model'][name].to(device))
    opt_m, opt_d = torch.optim.Adam(mlp.parameters(), lr=2e-4), torch.optim.Adam(deep.parameters(), lr=2e-4)
    run = LOG_ROOT / 'upright_supermimic' / 'history_study' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    config = {'args': vars(args), 'history_frames': HISTORY, 'control_dt': data['control_dt'],
              'training_data_seed': data['collection_args']['seed'], 'validation_data_seed': validation['collection_args']['seed'],
              'collection_source_checkpoint': data['source_checkpoint'], 'train_distinct_windows': len(train_sampler.valid),
              'validation_distinct_windows': len(val_sampler.valid),
              'mlp_params': sum(p.numel() for p in mlp.parameters()), 'deeponet_params': sum(p.numel() for p in deep.parameters()),
              'sampling': 'two batches sampled together; sequential optimizer updates; sampling with replacement',
              'validation': 'independent simulator rollout; fixed nonterminal windows sampled with replacement',
              'note': 'offline replay; DeepONet queries action prefix directly; not integrated into online training'}
    (run / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print('十帧模型对照输出：', run, flush=True)
    print('数据及模型：', json.dumps({k: config[k] for k in ('train_distinct_windows','validation_distinct_windows','mlp_params','deeponet_params')}), flush=True)
    for cycle in range(args.cycles):
        sums = [0.0, 0.0]
        for _ in range(args.groups_per_cycle):
            grouped = train_sampler.sample(args.batch * 2)
            for batch in range(2):
                sl = slice(batch * args.batch, (batch + 1) * args.batch)
                windows = tuple(v[sl] for v in grouped)
                sums[0] += update_mlp(mlp, opt_m, windows)
                sums[1] += update_deep(deep, opt_d, windows)
        metrics = {'cycle': cycle + 1, 'updates_per_model': (cycle + 1) * args.groups_per_cycle * 2,
                   'mlp_train_loss': sums[0] / (args.groups_per_cycle * 2),
                   'deeponet_train_loss': sums[1] / (args.groups_per_cycle * 2),
                   'holdout_mlp': evaluate(mlp, val, False), 'holdout_deeponet': evaluate(deep, val, True)}
        with (run / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if (cycle + 1) % args.save_interval == 0 or cycle + 1 == args.cycles:
            tmp = run / 'checkpoint.tmp'
            torch.save({'model_format':'history_world_model_v1', 'mlp':mlp.state_dict(), 'deeponet':deep.state_dict(),
                        'optimizer_mlp':opt_m.state_dict(), 'optimizer_deeponet':opt_d.state_dict(),
                        'args':vars(args), 'cycle':cycle + 1, 'metrics':metrics,
                        'training_data_seed':data['collection_args']['seed'], 'validation_data_seed':validation['collection_args']['seed'],
                        'collection_source_checkpoint':data['source_checkpoint'], 'control_dt':data['control_dt'],
                        'rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all()}, tmp)
            os.replace(tmp, run / 'latest.pt')


if __name__ == '__main__':
    main()
