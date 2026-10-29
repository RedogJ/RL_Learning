"""对独立测试集的全部有效 32 步窗口评估四种模型。"""
import argparse
import json
from pathlib import Path

import torch
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel
from go2_learning.tasks.upright_supermimic.deeponet_study.model import TimeDeepONet
from go2_learning.tasks.upright_supermimic.history_study.sampler import WindowSampler
from go2_learning.tasks.upright_supermimic.history_study.train import load_dataset
from go2_learning.tasks.upright_supermimic.recursive_study.train import next_step


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',required=True,help='独立测试数据集容器路径')
    p.add_argument('--baseline',required=True,help='旧直接查询模型检查点容器路径')
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--viz',choices=['none'],default='none')
    args=p.parse_args()
    if args.batch<1:p.error('batch 必须为正')
    torch.backends.cuda.matmul.allow_tf32=True
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    baseline=torch.load(args.baseline,map_location='cpu',weights_only=False)
    if saved.get('model_format')!='recursive_world_model_v1':raise ValueError('递推模型格式不符')
    if baseline['source']!=saved['args']['checkpoint']:
        raise ValueError('基线和新模型并非使用相同训练数据')
    device=torch.device('cuda:0')
    data,buf=load_dataset(args.dataset,device)
    if data['collection_args']['seed'] in (saved['training_data_seed'],saved['validation_data_seed']):
        raise ValueError('测试数据 seed 与训练或验证相同')
    if data['source_checkpoint']!=saved['collection_source_checkpoint']:
        raise ValueError('测试数据的源策略不一致')
    for name in ('input_mean','input_std','delta_mean','delta_std'):
        for comp in (saved['mlp'],saved['deeponet'],baseline['mlp'],baseline['deeponet']):
            if not torch.equal(comp[name],data['model'][name]):raise ValueError('归一化不一致：'+name)
    sampler=WindowSampler(buf,32,torch.arange(buf.num_envs,device=device),allow_last_done=False)
    entries=[('recursive_mlp',PaperWorldModel(90,12),saved['mlp'],'mlp'),
             ('recursive_deeponet',TimeDeepONet(),saved['deeponet'],'deep'),
             ('old_mlp',PaperWorldModel(90,12),baseline['mlp'],'mlp'),
             ('direct_deeponet',TimeDeepONet(),baseline['deeponet'],'direct')]
    output={'test_seed':data['collection_args']['seed'],'distinct_windows':len(sampler.valid),
            'note':'all eligible nonterminal 32-step windows, identical recorded actions for all models; overlapping windows are correlated',
            'training_cycle':saved['cycle'],'baseline_cycle':baseline['cycle'],'models':{}}
    for label,model,weights,kind in entries:
        model=model.to(device);model.load_state_dict(weights);model.eval()
        vals={str(k):{'joint':[],'gravity':[],'normalized':[]} for k in (1,8,16,32)}
        with torch.no_grad():
            for picks in sampler.valid.split(args.batch):
                time=(sampler.starts[picks[:,0],None]+torch.arange(32,device=device))%buf.capacity
                env=sampler.env_ids[picks[:,1],None].expand_as(time)
                states=buf.state[time,env];actions=buf.action[time,env];contexts=buf.context[time,env]
                targets=buf.next_state[time,env];predicted=states[:,0]
                for k in range(32):
                    if kind=='direct':
                        if k%8==0:block_start=predicted;block_actions=actions[:,k:k+8]
                        predicted,_=model(block_start,block_actions,contexts[:,k],k%8+1)
                    else:
                        predicted,_=next_step(model,predicted,actions[:,k],contexts[:,k],kind=='deep')
                    if k+1 in (1,8,16,32):
                        err=predicted-targets[:,k];norm=err/model.delta_std;v=vals[str(k+1)]
                        v['joint'].append(err[:,9:21].square().mean(-1).cpu())
                        v['gravity'].append(err[:,6:9].square().mean(-1).cpu())
                        v['normalized'].append(torch.cat((norm[:,:80],norm[:,88:]),-1).square().mean(-1).cpu())
        rows={}
        for step,v in vals.items():
            q=torch.cat(v['joint']);g=torch.cat(v['gravity']);n=torch.cat(v['normalized'])
            rows[step]={'joint_rmse_rad':float(q.mean().sqrt()),'gravity_rmse':float(g.mean().sqrt()),
                        'normalized_state_mse':float(n.mean()),
                        'joint_window_median_rad':float(torch.quantile(q.sqrt(),.5)),
                        'joint_window_p99_rad':float(torch.quantile(q.sqrt(),.99))}
        output['models'][label]=rows
        del model
        print('已评估：',label,flush=True)
    out=Path(args.checkpoint).with_name(f'independent_test_all_seed{output["test_seed"]}_cycle{output["training_cycle"]}.json')
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print('独立测试：',out,flush=True)
    print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
