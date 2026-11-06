import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
p=root/'independent_adam'
s=json.loads((p/'summary.json').read_text())
step=json.loads((p/'step_size_check.json').read_text())
assert step['exact_loss_match']
ends=[r for r in s['events'] if r['phase']=='after_wm']
keys=('root_position_rms_cm','root_rotation_rms_deg','joint_position_rmse_deg','linear_velocity_rmse_m_s','angular_velocity_rmse_rad_s','joint_velocity_rmse_rad_s')
aggregate={}
for branch in ends[0]['branches']:
 aggregate[branch]={}
 for probe in ('old_policy_3270','recent_policy_3330'):
  values={}
  for key in keys:
   data=[r['branches'][branch][probe]['physical']['16'][key] for r in ends]
   values[key]={'mean':statistics.mean(data),'std':statistics.pstdev(data),'max':max(data),'min':min(data)}
  values['bce8_mean']=statistics.mean(r['branches'][branch][probe]['bce8'] for r in ends)
  aggregate[branch][probe]=values
initial=s['events'][0]['branches']['joint_lr2e-4']
# Every branch shares the exact initial model/normalization and physical probes.
assert all(v==initial for v in s['events'][0]['branches'].values())
for probe in initial:
 assert len({r['branches'][name][probe]['physical']['16']['samples'] for r in ends for name in aggregate})==1
for cycle in range(1,10):
 before=next(r for r in s['events'] if r['cycle']==cycle and r['phase']=='before_wm')['branches']['joint_lr2e-4']
 previous=next(r for r in s['events'] if r['cycle']==cycle-1 and r['phase']=='after_wm')['branches']['joint_lr2e-4']
 assert before==previous
checks=s['gradient_checks']
grad={'sampled_steps':len(checks),'negative_cosines':sum(r['cosine']<0 for r in checks),
 'state_loss_increased_on_same_batch':sum(r['state_loss_after']>r['state_loss_before'] for r in checks),
 'actual_adam_step_dot_state_gradient_positive':sum(r['adam_step_dot_state_gradient']>0 for r in checks),
 'bce_to_state_gradient_norm_ratio_median':statistics.median(r['bce_to_state_norm_ratio'] for r in checks),
 'bce_to_state_gradient_norm_ratio_max':max(r['bce_to_state_norm_ratio'] for r in checks)}
result={'authoritative_experiment':'independent_adam','aggregation':'Arithmetic mean of fixed-probe RMS at 10 cycle ends, not pooled individual-sample RMS and not final checkpoint only.',
 'aggregate':aggregate,'gradient_summary':grad,'checks':{'initial_predictions_identical':True,'fixed_probe_sample_counts_identical':True,
 'probe_before_next_cycle_identical_to_previous_end':True,'bad_step_replayed_exactly':True,**s['checks']},
 'scope':'Actor/Critic frozen, fresh replay prefill, single collected stream, not exact historical RL continuation; no buffer-capacity A/B or policy-performance evaluation.'}
(root/'analysis.json').write_text(json.dumps(result,indent=2))
lines=['# WM 更新回退归因诊断','','## 最终采用的实验','','- 最终数值使用 `independent_adam/`。初测复用了 checkpoint optimizer 中 CPU step 计数对象；复验对每组 optimizer 状态作独立深拷贝，避免组间共享。初测记录保留作追溯，不作为最终定量结果。','- 从生产 checkpoint 3320 恢复；冻结 Actor 和 Critic，保持原归一化、奖励和物理参数。','- 2048 环境真实采集一次，含原 RSI、观测噪声和 0.1 动作噪声；采集前 320 步作为缓冲预填充，再采集 10×32 步。','- 训练时模拟原 320 步容量，每轮只用当前最近 320 步；60 次更新、batch512、窗口8。','- 三组使用完全相同的数据和 minibatch；保留原相位优先采样，由基线的优先级决定所有组共享的采样序列。','- 终止头学习率各组都是 2e-4。BCE detach 仅用于诊断，不修改生产实现。','- 探针是前次固定的旧策略3270/新策略3330独立验证轨迹，未用于本次训练。','', '## 十轮末固定探针 16 步误差的算术平均','','| 更新方式 | 旧轨迹位置 cm | 旧轨迹姿态 deg | 旧轨迹关节 deg | 新轨迹位置 cm | 新轨迹姿态 deg | 新轨迹关节 deg |','|---|---:|---:|---:|---:|---:|---:|']
labels={'joint_lr2e-4':'原联合更新，WM lr=2e-4','detached_bce_lr2e-4':'BCE 对 WM 停止梯度，lr=2e-4','joint_lr1e-4':'保留联合更新，WM lr=1e-4'}
for branch,values in aggregate.items():
 vals=[values[probe][k]['mean'] for probe in ('old_policy_3270','recent_policy_3330') for k in keys[:3]]
 lines.append('| '+labels[branch]+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
lines += ['', '## 已定位的现象','','1. Actor 不更新时，WM 更新阶段内部仍会出现固定探针误差回升。第一轮原基线的旧/新轨迹姿态 RMS：初始 8.33°/11.20°，第10次 7.62°/10.58°，第20次 11.97°/15.70°，第40次 12.56°/17.55°，第60次 8.91°/9.80°。这些更新间没有采集或淘汰 replay 数据。','2. 采集/切换可用数据阶段本身不改变 WM；下一轮更新前的固定探针指标与上轮更新后完全相同。','3. 保持 BCE 联合目标、只把 WM lr 降为1e-4，旧/新轨迹位置及姿态的十轮均值均改善，但新轨迹关节角均值略变差；不能声称所有指标都改善。','4. 阻断 BCE 到 WM 的梯度，六项物理 RMS 均值在这两组探针上均改善，但终止头验证 BCE 变差，表明动力学精度与终止预测存在权衡。','', '## 梯度与单步幅度检查','',f'- 抽查 {grad["sampled_steps"]} 次更新：{grad["negative_cosines"]} 次 BCE/状态梯度余弦为负；BCE 梯度范数与状态梯度之比中位数 {grad["bce_to_state_gradient_norm_ratio_median"]:.3f}。不能归纳为 BCE 持续反向拉扯。',f'- {grad["state_loss_increased_on_same_batch"]}/{grad["sampled_steps"]} 次更新后同批状态损失上升；抽查中实际 Adam 参数位移对状态梯度的点积均为负，说明一阶方向本身并未全部反向。','- 用保存的数据和 sample schedule 精确重放第一轮第20次更新，重放前后损失与记录完全一致。','', '| 实际 Adam 更新比例 | 同批 BF16 状态损失 | 同批 float32 状态损失（TF32允许） | 同批 BCE |','|---:|---:|---:|---:|']
for r in step['fractions']:
 lines.append(f'| {r["fraction"]:.2f} | {r["bf16_state_loss"]:.8f} | {r["float32_state_loss"]:.8f} | {r["bf16_bce"]:.8f} |')
lines += ['', '- 同一次更新取小步时状态拟合改善，完整步则状态损失上升，同时 BCE 下降；联合总损失仍可能下降。支持有限步长与多目标权衡的问题，不能只用“梯度方向错了”解释。','- 高精度前向也出现相同趋势，说明这一次的状态损失过冲并非仅由 BF16 前向舍入产生；本实验并未验证整个 FP32 训练分支。','- 该步的探针姿态与位置并非都在小步时最好，因此不能把一批训练 MSE 的升降等同于全局物理精度的升降。','', '## 判断及建议','','- 已证明：原更新方案自身能在固定数据/Actor条件下造成预测波动；无需 Actor 改变或一次 buffer 淘汰即可发生。','- 支持优先处理 WM 更新幅度及 BCE/动力学目标权衡。没有做 buffer 大小的因果对比，仍不能断言容量毫无作用。','- 建议下一轮在线验证先保留论文的 BCE 联合接法，将 WM 学习率试为1e-4，并持续记录固定物理探针；不直接把 BCE detach 当正式修复。','- 前次单片段3e-4加训的结论不能直接推广到跨阶段的数据流。','- 原始3320训练时的 replay/RNG没有保存，且本次冻结Actor；不能声称精确复现历史退化或已经证明最终策略会改善。','- 正式代码、奖励配置、生产日志、模型、latest、replay容量均未修改。','', '## 文件','','- `independent_adam/summary.json`、`events.jsonl`：最终三组对照。','- `independent_adam/gradients.jsonl`、`step_size_check.json`：梯度与精确重放。','- `fixed_stream.pt`：真实数据流，约470MiB，复验直接复用。','- `independent_adam/sample_schedule.pt`：三组共享的采样序列。','- `analysis.json`：聚合指标及一致性检查。','- `experiment.py`、`step_size_check.py`：复现脚本。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
