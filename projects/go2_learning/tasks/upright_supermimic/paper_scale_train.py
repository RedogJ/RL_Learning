"""独立试验：论文的 k=8、H=32 和 2048×3 世界模型，缩小并行批量。"""
from pathlib import Path

source = Path(__file__).with_name('train.py').read_text()


def change(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f'训练入口已变化，无法安全应用论文实验配置：{old!r}')
    source = source.replace(old, new, 1)


change('def main():', '''class PaperDynamics(Dynamics):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, 2048), nn.SiLU(),
                                 nn.Linear(2048, 2048), nn.SiLU(),
                                 nn.Linear(2048, 2048), nn.SiLU(),
                                 nn.Linear(2048, obs_dim + 2))
        with torch.no_grad():
            self.net[-1].bias[-1] = -4.0


def main():''')
change("    p.add_argument('--num-envs',", "    p.add_argument('--wm-horizon', type=int, default=8)\n    p.add_argument('--wm-batch-size', type=int, default=128)\n    p.add_argument('--disable-push', action='store_true', help='仅将随机推力大小设为 0；保留关节初始化随机化')\n    p.add_argument('--num-envs',")
change("    torch.manual_seed(args.seed)", "    if args.wm_horizon < 1 or args.wm_batch_size < 1 or args.replay_steps < args.wm_horizon:\n        p.error('世界模型窗口和批量必须有效')\n    torch.backends.cuda.matmul.allow_tf32 = True\n    torch.manual_seed(args.seed)")
change("('num_envs','replay_steps','horizon','randomization','reward_scale','gamma')", "('num_envs','replay_steps','horizon','wm_horizon','wm_batch_size','randomization','reward_scale','gamma')")
change('    task=task_from_snapshot(saved_cfg)', '''    task=task_from_snapshot(saved_cfg)
    if resume_data and args.disable_push != resume_data['args'].get('disable_push', False):
        raise ValueError('续训时 --disable-push 必须与检查点一致')
    if args.disable_push:
        if not args.randomization:
            p.error('--disable-push 需要 --randomization，以保留其他随机化')
        task.cfg.randomization.push_force_range=(0.0,0.0)
        saved_cfg['randomization_cfg']['push_force_range']=[0.0,0.0]
        print('随机推力幅度设为 0 N；保留初始化随机化和推力计时逻辑。',flush=True)''')
change('model=Dynamics(dim,env.num_actions).to(sim.device)', 'model=PaperDynamics(dim,env.num_actions).to(sim.device)')
change('opt_m=torch.optim.Adam(model.parameters(),lr=3e-4)', 'opt_m=torch.optim.Adam(model.parameters(),lr=2e-4)')
change('opt_a=torch.optim.Adam(actor.parameters(),lr=args.actor_lr)', 'opt_a=torch.optim.Adam(actor.parameters(),lr=args.actor_lr,betas=(0.9,0.95))')
change('opt_c=torch.optim.Adam(critic.parameters(),lr=3e-4)', 'opt_c=torch.optim.Adam(critic.parameters(),lr=2e-4)')
change("run=LOG_ROOT/'upright_supermimic'/('train_'", "run=LOG_ROOT/'upright_supermimic'/'paper_trials'/('train_'")
change('args.batch_size,args.horizon,args.warmup_updates if cycle==0', 'args.wm_batch_size,args.wm_horizon,args.warmup_updates if cycle==0')
change('max(args.batch_size,2048),args.horizon,failure_penalty', 'max(args.wm_batch_size,2048),args.wm_horizon,failure_penalty')
change("nn.utils.clip_grad_norm_(actor.parameters(),.5);opt_a.step()", "nn.utils.clip_grad_norm_(actor.parameters(),1.0);\n                        for parameter in actor.parameters():\n                            if parameter.grad is not None: parameter.grad.clamp_(-0.1,0.1)\n                        opt_a.step()")
exec(compile(source, str(Path(__file__).with_name('train.py')), 'exec'), {'__name__': '__main__'})
