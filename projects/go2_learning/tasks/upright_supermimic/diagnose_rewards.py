"""用与 evaluate 相同的评估流程，按专家/学生累计各项奖励。"""
from pathlib import Path

source = Path(__file__).with_name('evaluate.py').read_text()


def change(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f'评估入口已变化，无法安全添加奖励诊断：{old!r}')
    source = source.replace(old, new, 1)


change("obs=env.get_observations(); totals=torch.zeros(2,6,device=sim.device)",
       "obs=env.get_observations(); totals=torch.zeros(2,6,device=sim.device); components=[{},{}]")
change("                    obs,r,done,extra=env.step(a)", '''                    obs,r,done,extra=env.step(a)
                    ids=env._active_ids
                    for group in (0,1):
                        selected=(ids < args.pairs) if group==0 else (ids >= args.pairs)
                        for name, values in env._reward_components.items():
                            total=values[selected].sum()
                            components[group][name]=components[group].get(name, torch.zeros((),device=sim.device))+total''')
change("                print(json.dumps(out,ensure_ascii=False),flush=True)", '''                print(json.dumps(out,ensure_ascii=False),flush=True)
                seconds=max(t[0]*env.dt,1e-9)
                detail={name: float(value.cpu())/seconds for name,value in components[group].items()}
                print('REWARD_COMPONENTS '+json.dumps({'policy':label,'per_active_second':detail},ensure_ascii=False),flush=True)''')
exec(compile(source, str(Path(__file__).with_name('evaluate.py')), 'exec'), {'__name__': '__main__'})
