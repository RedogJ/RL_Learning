"""在已有单策略轨迹上离线训练连续状态世界模型，不训练策略。"""
import argparse
import json
import os
from datetime import datetime

import torch

from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.continuous_world_model import ContinuousWorldModel, ContinuousDeepONet
from go2_learning.tasks.upright_supermimic.history_study.sampler import WindowSampler
from go2_learning.tasks.upright_supermimic.history_study.train import load_dataset, verify_pair


def run_window(model, windows):
    states, actions, contexts, targets, *_ = windows
    pred = states[:, 0, :55]
    losses = []
    for k in range(actions.shape[1]):
        pred = model(pred, actions[:, k], contexts[:, k])
        # 执行动作是已知值，沿用旧世界模型的确定性覆盖。
        pred = torch.cat((pred[:, :33], actions[:, k], pred[:, 45:]), -1)
        diff = (pred - targets[:, k, :55]) / model.out_norm.std().clamp_min(1e-3)
        losses.append(diff.square().mean())
    return torch.stack(losses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('mlp', 'deeponet'), default='mlp')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--validation-checkpoint', required=True)
    p.add_argument('--cycles', type=int, default=250)
    p.add_argument('--groups-per-cycle', type=int, default=10)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--eval-windows', type=int, default=512)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--viz', default='none', choices=['none'])
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
    train = WindowSampler(buf, 8, torch.arange(buf.num_envs, device=device))
    val = WindowSampler(val_buf, 16, torch.arange(val_buf.num_envs, device=device), allow_last_done=False)
    probe = val.sample(args.eval_windows)
    model = (ContinuousDeepONet(55, 12) if args.model == 'deeponet' else ContinuousWorldModel(55, 12)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    folder = 'continuous_deeponet' if args.model == 'deeponet' else 'continuous_pretrain'
    run = LOG_ROOT / 'upright_supermimic' / folder / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'args': vars(args), 'state': '原状态前55维；不预测接触',
        'source': data['source_checkpoint'], 'normalization': '对抽样真实训练状态及增量更新在线统计',
        'coordinate': '沿用原数据坐标；没有引入局部旋转表示'}, ensure_ascii=False, indent=2))
    print(f'模型：{args.model} | 输出：{run} | 连续状态 55 维 | 训练窗口 {len(train.valid)} | 验证窗口 {len(val.valid)}', flush=True)
    for cycle in range(1, args.cycles + 1):
        loss_sum = torch.zeros((), device=device)
        for _ in range(args.groups_per_cycle):
            grouped = train.sample(args.batch * 2)
            for b in range(2):
                w = tuple(x[b*args.batch:(b+1)*args.batch] for x in grouped)
                s, a, ctx, target, *_ = w
                model.update_statistics(s[..., :55], a, ctx, target[..., :55])
                loss = run_window(model, w).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('世界模型损失非有限，已停止')
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                loss_sum += loss.detach()
        with torch.no_grad():
            vals = run_window(model, probe)
        row = {'cycle': cycle, 'updates': cycle*args.groups_per_cycle*2,
               'train_loss': float(loss_sum/(args.groups_per_cycle*2)),
               'val_1': float(vals[0]), 'val_8': float(vals[7]), 'val_16': float(vals[15]),
               'normalization_samples': int(model.in_norm.count.item())}
        with (run/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if cycle % args.save_interval == 0 or cycle == args.cycles:
            tmp = run/'checkpoint.tmp'
            torch.save({'format': 'continuous_pretrain_v1', 'model': model.state_dict(),
                        'optimizer': optim.state_dict(), 'args': vars(args), 'cycle': cycle, 'metrics': row}, tmp)
            os.replace(tmp, run/'latest.pt')
    print(f'完成：{run}', flush=True)


if __name__ == '__main__':
    main()
