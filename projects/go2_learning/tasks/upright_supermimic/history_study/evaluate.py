"""在相同独立 10 帧历史与动作轨迹上比较历史版及单帧版模型。"""
import argparse
import json
from pathlib import Path
import torch

from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel
from go2_learning.tasks.upright_supermimic.deeponet_study.model import TimeDeepONet
from go2_learning.tasks.upright_supermimic.history_study.model import HISTORY, HistoryMLP, HistoryDeepONet
from go2_learning.tasks.upright_supermimic.history_study.sampler import WindowSampler
from go2_learning.tasks.upright_supermimic.history_study.train import append_history, evaluate, load_dataset, unpack


def distribution(model, windows, is_deep, uses_history):
    """32 步时每个窗口的误差分布，避免仅由少量极值决定判断。"""
    history, actions, contexts, targets, _ = unpack(windows)
    with torch.no_grad():
        for k in range(32):
            if is_deep:
                if k % 8 == 0:
                    block_history = history
                    block_actions = actions[:, k:k+8]
                model_input = block_history if uses_history else block_history[:, -1]
                predicted, _ = model(model_input, block_actions, contexts[:, k], k % 8 + 1)
            else:
                model_input = history if uses_history else history[:, -1]
                predicted, _ = model(model_input, actions[:, k], contexts[:, k])
            history = append_history(history, predicted)
        err = predicted - targets[:, 31]
        joint = err[:, 9:21].square().mean(-1).sqrt()
        gravity = err[:, 6:9].square().mean(-1).sqrt()
        return {name:{'median':float(torch.quantile(x,.5)), 'p90':float(torch.quantile(x,.9)),
                      'p99':float(torch.quantile(x,.99)), 'max':float(x.max())}
                for name,x in (('joint_rmse_rad_per_window',joint), ('gravity_rmse_per_window',gravity))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='十帧模型检查点')
    p.add_argument('--dataset', required=True, help='测试数据集容器路径')
    p.add_argument('--single-checkpoint', required=True, help='单帧对照检查点容器路径')
    p.add_argument('--eval-windows', type=int, default=1024)
    p.add_argument('--seed', type=int, default=9251)
    p.add_argument('--viz', choices=['none'], default='none')
    args = p.parse_args()
    if args.eval_windows < 1:
        p.error('测试窗口数量必须为正')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    single = torch.load(args.single_checkpoint, map_location='cpu', weights_only=False)
    if saved.get('model_format') != 'history_world_model_v1':
        raise ValueError('不是十帧模型检查点')
    device = torch.device('cuda:0')
    data, buf = load_dataset(args.dataset, device)
    if data['collection_args']['seed'] in (saved['training_data_seed'], saved['validation_data_seed']):
        raise ValueError('测试数据随机种子不能与训练和验证相同')
    if data['source_checkpoint'] != saved['collection_source_checkpoint']:
        raise ValueError('测试数据的源策略检查点不同')
    for name in ('input_mean', 'input_std', 'delta_mean', 'delta_std'):
        for component in (saved['mlp'], saved['deeponet'], single['mlp'], single['deeponet']):
            if not torch.equal(component[name], data['model'][name]):
                raise ValueError('模型和数据归一化不同：' + name)
    sampler = WindowSampler(buf, HISTORY - 1 + 32, torch.arange(buf.num_envs, device=device), allow_last_done=False)
    windows = sampler.sample(args.eval_windows)
    models = [('history_mlp', HistoryMLP(), saved['mlp'], False, True),
              ('history_deeponet', HistoryDeepONet(), saved['deeponet'], True, True),
              ('single_mlp', PaperWorldModel(90, 12), single['mlp'], False, False),
              ('single_deeponet', TimeDeepONet(), single['deeponet'], True, False)]
    output = {'history_checkpoint':args.checkpoint, 'single_checkpoint':args.single_checkpoint, 'dataset':args.dataset,
              'dataset_seed':data['collection_args']['seed'], 'sample_seed':args.seed,
              'distinct_valid_windows':len(sampler.valid), 'sampled_windows':args.eval_windows,
              'history_frames':HISTORY, 'note':'same nonterminal trajectories and recorded actions; sampling with replacement',
              'models':{}, 'distribution_at_32':{}}
    for name, model, state, is_deep, uses_history in models:
        model = model.to(device); model.load_state_dict(state); model.eval()
        output['models'][name] = evaluate(model, windows, is_deep, uses_history)
        output['distribution_at_32'][name] = distribution(model, windows, is_deep, uses_history)
        del model
    out = Path(args.checkpoint).with_name(f'independent_test_history_seed{output["dataset_seed"]}_sample{args.seed}.json')
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print('同轨迹独立测试：', out, flush=True)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
