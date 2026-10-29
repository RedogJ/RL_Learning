"""MLP 与单步查询 DeepONet 的相同自回归训练。"""
import argparse
import json
import os
from datetime import datetime

import torch
from torch.nn import functional as F

from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel
from go2_learning.tasks.upright_supermimic.deeponet_study.model import TimeDeepONet
from go2_learning.tasks.upright_supermimic.history_study.sampler import WindowSampler
from go2_learning.tasks.upright_supermimic.history_study.train import load_dataset, verify_pair


def next_step(model, state, action, context, is_deep):
    """两种模型均只查询下一步；DeepONet 的其余动作槽被遮罩。"""
    if not is_deep:
        return model(state, action, context)
    zeros = action.new_zeros((len(action), model.horizon - 1, action.shape[-1]))
    return model(state, torch.cat((action[:, None], zeros), dim=1), context, 1)


def update(model, optimizer, windows, is_deep):
    states, actions, contexts, targets, fail, _, _ = windows
    predicted = states[:, 0]
    loss = 0.0
    for k in range(actions.shape[1]):
        predicted, logits = next_step(model, predicted, actions[:, k], contexts[:, k], is_deep)
        diff = (predicted - targets[:, k]) / model.delta_std
        loss = loss + (torch.cat((diff[:, :80], diff[:, 88:]), -1).square().mean()
                       + F.binary_cross_entropy_with_logits(logits, fail[:, k].float())) / actions.shape[1]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def evaluate(model, windows, is_deep):
    states, actions, contexts, targets, _, _, _ = windows
    predicted = states[:, 0]
    rows = {}
    with torch.no_grad():
        for k in range(actions.shape[1]):
            predicted, _ = next_step(model, predicted, actions[:, k], contexts[:, k], is_deep)
            if k + 1 in (1, 8, 16, 32):
                err = predicted - targets[:, k]
                norm = err / model.delta_std
                rows[str(k + 1)] = {'joint_rmse_rad': float(err[:, 9:21].square().mean().sqrt()),
                                    'gravity_rmse': float(err[:, 6:9].square().mean().sqrt()),
                                    'normalized_state_mse': float(torch.cat((norm[:, :80], norm[:, 88:]), -1).square().mean())}
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='训练数据集')
    p.add_argument('--validation-checkpoint', required=True, help='验证数据集容器路径')
    p.add_argument('--resume', default=None, help='本实验检查点；cycles 为追加轮数')
    p.add_argument('--cycles', type=int, default=250)
    p.add_argument('--groups-per-cycle', type=int, default=10)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--eval-windows', type=int, default=512)
    p.add_argument('--save-interval', type=int, default=25)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--viz', choices=['none'], default='none')
    args = p.parse_args()
    if min(args.cycles,args.groups_per_cycle,args.batch,args.eval_windows,args.save_interval)<1:
        p.error('数量必须为正')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda:0')
    data, buf = load_dataset(args.checkpoint, device)
    validation, val_buf = load_dataset(args.validation_checkpoint, device)
    verify_pair(data, validation)
    train_sampler = WindowSampler(buf, 8, torch.arange(buf.num_envs, device=device))
    val_sampler = WindowSampler(val_buf, 32, torch.arange(val_buf.num_envs, device=device), allow_last_done=False)
    val = val_sampler.sample(args.eval_windows)
    mlp, deep = PaperWorldModel(90, 12).to(device), TimeDeepONet().to(device)
    for model in (mlp, deep):
        for name in ('input_mean','input_std','delta_mean','delta_std'):
            getattr(model,name).copy_(data['model'][name].to(device))
    opt_m, opt_d = torch.optim.Adam(mlp.parameters(), lr=2e-4), torch.optim.Adam(deep.parameters(), lr=2e-4)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('model_format') != 'recursive_world_model_v1':
            raise ValueError('续训需要本实验检查点')
        for key in ('checkpoint','validation_checkpoint','groups_per_cycle','batch','eval_windows','seed'):
            if getattr(args,key) != saved['args'][key]:
                raise ValueError('续训参数须保持一致：' + key)
        mlp.load_state_dict(saved['mlp']); deep.load_state_dict(saved['deeponet'])
        opt_m.load_state_dict(saved['optimizer_mlp']); opt_d.load_state_dict(saved['optimizer_deeponet'])
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start = saved['cycle']
    run = LOG_ROOT / 'upright_supermimic' / 'recursive_study' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    config = {'args':vars(args), 'start_cycle':start, 'control_dt':data['control_dt'],
              'training_data_seed':data['collection_args']['seed'], 'validation_data_seed':validation['collection_args']['seed'],
              'collection_source_checkpoint':data['source_checkpoint'], 'train_distinct_windows':len(train_sampler.valid),
              'validation_distinct_windows':len(val_sampler.valid),
              'mlp_params':sum(x.numel() for x in mlp.parameters()), 'deeponet_params':sum(x.numel() for x in deep.parameters()),
              'sampling':'two 2048-window batches sampled together; sequential updates; with replacement',
              'note':'both models use identical 8-step recursive training; DeepONet always queries one step'}
    config['sampling'] = f'two {args.batch}-window batches sampled together; sequential updates; with replacement'
    (run/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    print('单步递推对照输出：',run,flush=True)
    print('数据及模型：',json.dumps({k:config[k] for k in ('train_distinct_windows','validation_distinct_windows','mlp_params','deeponet_params')}),flush=True)
    for cycle in range(start,start+args.cycles):
        sums=[0.0,0.0]
        for _ in range(args.groups_per_cycle):
            grouped=train_sampler.sample(args.batch*2)
            for batch in range(2):
                sl=slice(batch*args.batch,(batch+1)*args.batch)
                windows=tuple(v[sl] for v in grouped)
                sums[0]+=update(mlp,opt_m,windows,False)
                sums[1]+=update(deep,opt_d,windows,True)
        metrics={'cycle':cycle+1,'updates_per_model':(cycle+1)*args.groups_per_cycle*2,
                 'mlp_train_loss':sums[0]/(args.groups_per_cycle*2),
                 'deeponet_train_loss':sums[1]/(args.groups_per_cycle*2),
                 'holdout_mlp':evaluate(mlp,val,False),'holdout_deeponet':evaluate(deep,val,True)}
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics,ensure_ascii=False),flush=True)
        if (cycle+1)%args.save_interval==0 or cycle+1==start+args.cycles:
            tmp=run/'checkpoint.tmp'
            torch.save({'model_format':'recursive_world_model_v1','mlp':mlp.state_dict(),'deeponet':deep.state_dict(),
                        'optimizer_mlp':opt_m.state_dict(),'optimizer_deeponet':opt_d.state_dict(),
                        'args':vars(args),'cycle':cycle+1,'metrics':metrics,
                        'training_data_seed':data['collection_args']['seed'],
                        'validation_data_seed':validation['collection_args']['seed'],
                        'collection_source_checkpoint':data['source_checkpoint'],'control_dt':data['control_dt'],
                        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},tmp)
            os.replace(tmp,run/'latest.pt')
            if (cycle+1) in (100,200) or cycle+1==start+args.cycles:
                milestone=run/f'cycle_{cycle+1}.pt'
                os.link(run/'latest.pt',milestone)


if __name__=='__main__':
    main()
