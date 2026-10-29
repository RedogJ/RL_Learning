"""在已有轨迹上训练直接时间查询 DeepONet；不更新策略。"""
import argparse
import json
import os
from datetime import datetime

import torch

from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.continuous_world_model import DirectTimeDeepONet
from go2_learning.tasks.upright_supermimic.history_study.sampler import WindowSampler
from go2_learning.tasks.upright_supermimic.history_study.train import load_dataset, verify_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--validation-checkpoint', required=True)
    p.add_argument('--cycles', type=int, default=250)
    p.add_argument('--groups-per-cycle', type=int, default=10)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--eval-windows', type=int, default=512)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--viz', choices=['none'], default='none')
    args = p.parse_args()
    if min(args.cycles, args.groups_per_cycle, args.batch, args.eval_windows, args.save_interval) < 1:
        p.error('数量参数必须大于零')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda:0')
    data, buf = load_dataset(args.checkpoint, device)
    val_data, val_buf = load_dataset(args.validation_checkpoint, device)
    verify_pair(data, val_data)
    if data['collection_args'].get('policy') != 'expert' or val_data['collection_args'].get('policy') != 'expert':
        raise ValueError('训练和验证数据必须来自单一专家策略')
    train = WindowSampler(buf, 16, torch.arange(buf.num_envs, device=device))
    val = WindowSampler(val_buf, 16, torch.arange(val_buf.num_envs, device=device), allow_last_done=False)
    probe = val.sample(args.eval_windows)
    model = DirectTimeDeepONet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    run = LOG_ROOT/'upright_supermimic'/'direct_time_deeponet'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({'args': vars(args), 'state': '原状态前55维；不预测接触',
        'source': data['source_checkpoint'], 'horizon': 16, 'training': '每次更新均匀随机选择一个离散查询步',
        'coordinate': '沿用原数据坐标', 'future_actions': '查询时刻之后遮蔽'}, ensure_ascii=False, indent=2))
    print(f'直接时间查询 DeepONet | 输出：{run} | 训练窗口 {len(train.valid)} | 验证窗口 {len(val.valid)}', flush=True)
    best = float('inf')
    for cycle in range(1, args.cycles+1):
        total = torch.zeros((), device=device)
        for _ in range(args.groups_per_cycle):
            grouped = train.sample(args.batch*2)
            for b in range(2):
                s, a, ctx, target, *_ = (x[b*args.batch:(b+1)*args.batch] for x in grouped)
                model.update_statistics(s[..., :55], a, ctx, target[..., :55])
                step = int(torch.randint(1, 17, (), device='cpu'))
                pred = model(s[:, 0, :55], a, ctx[:, 0], step)
                diff = (pred-target[:, step-1, :55])/model.out_norm.std().clamp_min(1e-3)
                loss = diff.square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('世界模型损失非有限，已停止')
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                total += loss.detach()
        s,a,ctx,target,*_ = probe
        with torch.no_grad():
            values = {}
            for step in (1,8,16):
                pred = model(s[:,0,:55], a, ctx[:,0], step)
                diff = (pred-target[:,step-1,:55])/model.out_norm.std().clamp_min(1e-3)
                values[f'val_{step}'] = float(diff.square().mean())
        row = {'cycle': cycle, 'updates': cycle*args.groups_per_cycle*2,
               'train_loss': float(total/(args.groups_per_cycle*2)), **values,
               'normalization_samples': int(model.in_norm.count.item())}
        with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
        improved = row['val_16'] < best
        if improved or cycle % args.save_interval == 0 or cycle == args.cycles:
            payload = {'format':'direct_time_deeponet_v1', 'model':model.state_dict(),
                       'optimizer':optim.state_dict(), 'args':vars(args), 'cycle':cycle, 'metrics':row}
            if improved:
                best = row['val_16']
                torch.save(payload, run/'best.tmp')
                os.replace(run/'best.tmp', run/'best.pt')
            if cycle % args.save_interval == 0 or cycle == args.cycles:
                torch.save(payload, run/'checkpoint.tmp')
                os.replace(run/'checkpoint.tmp', run/'latest.pt')
    print(f'完成：{run}', flush=True)


if __name__ == '__main__':
    main()
