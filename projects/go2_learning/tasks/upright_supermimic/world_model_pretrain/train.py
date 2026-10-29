"""Pretrain the existing MLP world model without updating Actor or Critic."""
import argparse
import json
import os
from datetime import datetime

import torch
from torch.nn import functional as F

from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.history_study.sampler import WindowSampler
from go2_learning.tasks.upright_supermimic.history_study.train import load_dataset, verify_pair
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel
from go2_learning.tasks.upright_supermimic.recursive_study.train import evaluate


def train_batch(model, optimizer, windows):
    states, actions, contexts, targets, fail, _, _ = windows
    predicted = states[:, 0]
    dyn = predicted.new_zeros(())
    terminal = predicted.new_zeros(())
    for k in range(actions.shape[1]):
        predicted, logits = model(predicted, actions[:, k], contexts[:, k])
        difference = (predicted - targets[:, k]) / model.delta_std
        dyn = dyn + torch.cat((difference[:, :80], difference[:, 88:]), -1).square().mean() / actions.shape[1]
        terminal = terminal + F.binary_cross_entropy_with_logits(logits, fail[:, k].float()) / actions.shape[1]
    optimizer.zero_grad(set_to_none=True)
    (dyn + terminal).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return dyn.detach(), terminal.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='训练数据集')
    p.add_argument('--validation-checkpoint', required=True, help='验证数据集容器路径')
    p.add_argument('--resume', help='本预训练实验的 latest.pt')
    p.add_argument('--init', choices=['scratch', 'source'], default='scratch',
                   help='scratch 为新世界模型；source 使用采集来源中的世界模型权重')
    p.add_argument('--cycles', type=int, default=100, help='追加的轮数')
    p.add_argument('--groups-per-cycle', type=int, default=10)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--eval-windows', type=int, default=512)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--viz', choices=['none'], default='none')
    args = p.parse_args()
    if min(args.cycles, args.groups_per_cycle, args.batch, args.eval_windows, args.save_interval) < 1:
        p.error('所有数量参数必须大于零')

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda:0')
    data, buf = load_dataset(args.checkpoint, device)
    validation, val_buf = load_dataset(args.validation_checkpoint, device)
    verify_pair(data, validation)
    mode = data['collection_args'].get('policy')
    if mode not in ('expert', 'actor') or validation['collection_args'].get('policy') != mode:
        raise ValueError('需要由相同单一策略模式采集的训练和验证数据，旧的混合数据不可使用')
    source = torch.load(data['source_checkpoint'], map_location='cpu', weights_only=False)
    if source.get('world_model_format') != 'paper_context_v3':
        raise ValueError('数据源必须是 paper_context_v3 世界模型检查点')
    for name in ('input_mean', 'input_std', 'delta_mean', 'delta_std'):
        if not torch.equal(source['model'][name], data['model'][name]):
            raise ValueError('数据源归一化参数不匹配：' + name)

    ids = torch.arange(buf.num_envs, device=device)
    validation_ids = torch.arange(val_buf.num_envs, device=device)
    train_sampler = WindowSampler(buf, 8, ids)
    train_eval_sampler = WindowSampler(buf, 16, ids, allow_last_done=False)
    val_sampler = WindowSampler(val_buf, 16, validation_ids, allow_last_done=False)
    # Fixed probes allow comparisons across updates; all are sampled with replacement.
    train_probe = train_eval_sampler.sample(args.eval_windows)
    validation_probe = val_sampler.sample(args.eval_windows)

    model = PaperWorldModel(90, 12).to(device)
    if args.init == 'source':
        model.load_state_dict(source['model'])
    else:
        # Keep the existing normalization unchanged while initializing network weights anew.
        for name in ('input_mean', 'input_std', 'delta_mean', 'delta_std'):
            getattr(model, name).copy_(data['model'][name].to(device))
    # Start fresh optimizer statistics when switching to the much larger dataset.
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('format') != 'world_model_pretrain_v1':
            raise ValueError('续训需要本入口保存的预训练检查点')
        for key in ('checkpoint', 'validation_checkpoint', 'groups_per_cycle', 'batch', 'eval_windows', 'seed', 'init'):
            if getattr(args, key) != saved['args'][key]:
                raise ValueError('续训参数须保持一致：' + key)
        if saved['source_checkpoint'] != data['source_checkpoint']:
            raise ValueError('续训数据源不匹配')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start = saved['cycle']

    run = LOG_ROOT / 'upright_supermimic' / 'world_model_pretrain' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    info = {'args': vars(args), 'source_checkpoint': data['source_checkpoint'],
            'source_cycle': source['cycle'], 'start_cycle': start,
            'train_seed': data['collection_args']['seed'], 'validation_seed': validation['collection_args']['seed'],
            'train_valid_8_windows': len(train_sampler.valid),
            'validation_valid_16_windows': len(val_sampler.valid), 'policy': mode,
            'sampling': 'two batches sampled together; two separate optimizer updates; sampling with replacement',
            'validation': 'fixed samples from separate collection; 16-step windows exclude all terminal steps',
            'note': 'offline world model only; executed actions replayed; no actor, critic or simulator updates'}
    (run / 'config.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print('世界模型预训练输出：', run, flush=True)
    print('数据：', json.dumps({k: info[k] for k in ('train_valid_8_windows', 'validation_valid_16_windows', 'policy')}), flush=True)

    def metrics(cycle, dyn=None, terminal=None):
        result = {'cycle': cycle, 'updates': cycle * args.groups_per_cycle * 2,
                  'train_dynamics_loss': dyn, 'train_terminal_bce': terminal,
                  'train_probe': evaluate(model, train_probe, False),
                  'validation_policy': evaluate(model, validation_probe, False)}
        with (run / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    metrics(start)
    for cycle in range(start, start + args.cycles):
        dyn_total = torch.zeros((), device=device)
        terminal_total = torch.zeros((), device=device)
        for _ in range(args.groups_per_cycle):
            grouped = train_sampler.sample(args.batch * 2)
            for b in range(2):
                sl = slice(b * args.batch, (b + 1) * args.batch)
                dyn, terminal = train_batch(model, optimizer, tuple(v[sl] for v in grouped))
                dyn_total += dyn
                terminal_total += terminal
        count = args.groups_per_cycle * 2
        result = metrics(cycle + 1, float(dyn_total / count), float(terminal_total / count))
        if (cycle + 1) % args.save_interval == 0 or cycle + 1 == start + args.cycles:
            temporary = run / 'checkpoint.tmp'
            torch.save({'format': 'world_model_pretrain_v1', 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'cycle': cycle + 1, 'args': vars(args),
                        'metrics': result, 'source_checkpoint': data['source_checkpoint'],
                        'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temporary)
            os.replace(temporary, run / 'latest.pt')
    print('预训练完成：', run, flush=True)


if __name__ == '__main__':
    main()
