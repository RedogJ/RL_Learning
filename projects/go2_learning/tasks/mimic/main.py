"""独立复现入口：pin → collect → train-vae → dagger → latent collect → train-diffusion → play。"""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import torch
from go2_learning.paths import LOG_ROOT
from .experts import ROOT, pin_experts


def parse_args():
    from .config import GuidanceCfg
    defaults=GuidanceCfg()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('pin','collect','evaluate','train-vae','dagger','train-diffusion','play','relabel'))
    parser.add_argument('--expert',choices=('running','upright'),default='running')
    parser.add_argument('--controller',choices=('expert','vae','diffusion'),default='expert')
    parser.add_argument('--num_envs',type=int,default=16)
    parser.add_argument('--steps',type=int,default=900)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--batch_size',type=int,default=256)
    simulation_command = len(sys.argv)>1 and sys.argv[1] in ('collect','play','evaluate')
    if not simulation_command:
        parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data',default=str(ROOT/'data'))
    parser.add_argument('--output',default=None)
    parser.add_argument('--vae',default=str(ROOT/'models/vae.pt'))
    parser.add_argument('--model',default=str(ROOT/'models/diffusion.pt'))
    parser.add_argument('--resume',default=None)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--noise',type=float,default=0.)
    parser.add_argument('--teacher_mix',type=float,default=0.)
    parser.add_argument('--command_profile',choices=('episode','transitions'),default='episode')
    parser.add_argument('--transition_fraction',type=float,default=0.,help='diffusion 各技能批次内切换窗口采样比例')
    parser.add_argument('--guidance',type=float,default=.05)
    parser.add_argument('--guidance_mode',choices=('output','input'),default='output',
                        help='output 为原输出修正；input 为经去噪网络回传的实验路径')
    parser.add_argument('--cost_mode',choices=('legacy','separate'),default=defaults.mode)
    parser.add_argument('--velocity_guidance_weight',type=float,default=defaults.velocity)
    parser.add_argument('--orientation_guidance_weight',type=float,default=defaults.orientation)
    parser.add_argument('--height_guidance_weight',type=float,default=defaults.height)
    parser.add_argument('--yaw_guidance_weight',type=float,default=defaults.yaw)
    parser.add_argument('--future_keyframe_prob',type=float,default=0.,help='训练时随机可见未来状态的概率')
    parser.add_argument('--keyframes',default=None,help='播放时使用的关键帧库.pt')
    parser.add_argument('--keyframe_entry',type=float,default=.5,help='切upright后首关键帧的时间，秒')
    parser.add_argument('--keyframe_interval',type=float,default=.2,help='关键帧间隔，秒')
    parser.add_argument('--command_conditioned',action='store_true',help='训练六维当前指令条件扩散；输出需指定独立路径')
    parser.add_argument('--emphasis',type=float,default=6.,help='新 diffusion 的根特征强调；0 可训练旧结构')
    parser.add_argument('--eager',action='store_true',help='关闭扩散 CUDA Graph，供数值和性能对照')
    parser.add_argument('--skill_switch',action='store_true',help='播放同一 diffusion 时用 Tab/1/2 切换技能目标')
    parser.add_argument('--keyboard',action=argparse.BooleanOptionalAction,default=None,
                        help='方向键速度控制；技能切换模式默认开启')
    parser.add_argument('--vx',type=float,default=0.)
    parser.add_argument('--vy',type=float,default=0.)
    parser.add_argument('--rounds',type=int,default=3)
    # 离线命令不依赖 Isaac Sim，只有仿真子命令加载启动器。
    if simulation_command:
        from isaaclab.app import add_launcher_args
        add_launcher_args(parser)
    args=parser.parse_args()
    try:
        GuidanceCfg(args.cost_mode,args.velocity_guidance_weight,args.orientation_guidance_weight,
                    args.height_guidance_weight,args.yaw_guidance_weight)
    except ValueError as error:parser.error(str(error))
    if not 0<=args.future_keyframe_prob<=1:parser.error('future_keyframe_prob 必须在[0,1]')
    if args.future_keyframe_prob and (args.command!='train-diffusion' or args.output is None):
        parser.error('关键帧训练请使用train-diffusion并指定独立output路径')
    if args.keyframes and (args.command not in ('play','evaluate') or args.controller!='diffusion'):
        parser.error('keyframes 仅用于 diffusion play/evaluate')
    if not math.isfinite(args.keyframe_entry) or args.keyframe_entry<=0:parser.error('keyframe_entry 必须有限且>0')
    if not math.isfinite(args.keyframe_interval) or args.keyframe_interval<=0:parser.error('keyframe_interval 必须有限且>0')
    if args.command_conditioned and args.command!='train-diffusion':
        parser.error('command_conditioned 只用于 train-diffusion；播放自动读取模型结构')
    if not math.isfinite(args.emphasis) or args.emphasis<0:parser.error('emphasis 必须有限且非负')
    if not 0<=args.transition_fraction<=1:parser.error('transition_fraction 必须在 [0,1]')
    if args.command_profile=='transitions':
        if args.command not in ('collect','evaluate','play','dagger') or (args.command!='dagger' and args.expert!='running'):
            parser.error('transitions 指令仅用于 running 的 collect/evaluate/play')
        if args.skill_switch or args.keyboard:parser.error('自动切换指令不能与键盘控制同时使用')
    if args.keyboard is None:args.keyboard=args.skill_switch
    if args.skill_switch or args.keyboard:
        if args.command!='play' or args.controller not in ('vae','diffusion'):
            parser.error('方向键控制仅用于 play --controller vae 或 diffusion')
        if args.skill_switch and args.controller!='diffusion':
            parser.error('技能切换仅用于 play --controller diffusion')
        visualizers=getattr(args,'visualizer',None) or []
        if isinstance(visualizers,str):visualizers=visualizers.split(',')
        if 'newton_gl' not in visualizers:
            parser.error('键盘控制需要 --viz newton_gl')
    if args.steps<1 or args.num_envs<1 or args.updates<1 or args.batch_size<2 or args.rounds<1:
        parser.error('steps/num_envs/updates/rounds 必须为正，batch_size 至少为 2')
    if not 0<=args.teacher_mix<=1 or args.noise<0 or args.guidance<0:
        parser.error('teacher_mix 在 [0,1]，noise/guidance 非负')
    if args.command=='collect' and args.controller=='diffusion':
        parser.error('扩散训练数据应由固定 VAE 采集')
    if args.output is None:
        args.output=str(ROOT/'models/vae.pt' if args.command=='train-vae' else
                        ROOT/('models/diffusion_command_v1.pt' if args.command_conditioned else
                              'models/diffusion_emphasis.pt' if args.emphasis else 'models/diffusion.pt')
                        if args.command=='train-diffusion' else
                        ROOT/'evaluation' if args.command=='evaluate' else ROOT/'data')
    return args


def main():
    args=parse_args()
    if args.command=='pin':
        print(pin_experts());return
    if args.command=='relabel':
        from .supervision import relabel_running
        relabel_running(args)
    elif args.command=='train-vae':
        from .train import train_vae
        train_vae(args)
    elif args.command=='train-diffusion':
        from .train import train_diffusion
        train_diffusion(args)
    elif args.command=='dagger':
        # 每轮先在学生实际访问到的状态上查询两个专家，再混合全部历史数据训练。
        for round_id in range(args.rounds):
            for expert in ('running','upright'):
                command=[sys.executable,'-B','-m','go2_learning.tasks.mimic.main','collect',
                    '--expert',expert,'--controller','vae','--vae',args.vae,'--output',args.data,
                    '--steps',str(args.steps),'--num_envs',str(args.num_envs),'--viz','none',
                    '--command_profile',args.command_profile if expert=='running' else 'episode',
                    '--teacher_mix',str(max(0.,.5*(1-round_id/args.rounds))),
                    '--seed',str(args.seed+round_id)]
                subprocess.run(command,check=True)
            from .supervision import relabel_running
            relabel_running(args)
            from .train import train_vae
            args.resume=args.vae;args.output=args.vae
            train_vae(args)
    else:
        from .simulation import run
        run(args)


if __name__=='__main__':main()
