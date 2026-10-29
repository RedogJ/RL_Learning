"""一次性的同步分段计时；运行 train.py 的相同训练逻辑，不修改原训练入口。"""
from pathlib import Path

source = Path(__file__).with_name("train.py").read_text()


def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"训练代码已变化，计时插桩无法安全应用: {old!r}")
    source = source.replace(old, new, 1)


replace_once("import argparse, json, os", "import argparse, json, os, time")
replace_once("def main():", "def stamp():\n    torch.cuda.synchronize()\n    return time.perf_counter()\n\n\ndef main():")
replace_once("run=LOG_ROOT/'upright_supermimic'/('train_'", "run=LOG_ROOT/'upright_supermimic'/'benchmarks'/('sm_'")
replace_once("                falls=0; active_count=0; reward_sum=0.", "                timing={}; start_cycle=stamp(); t=start_cycle\n                falls=0; active_count=0; reward_sum=0.")
replace_once("                wm_loss=wm_updates(", "                timing['collect_and_calibrate']=stamp()-t; t=stamp()\n                wm_loss=wm_updates(")
replace_once("                stats=validation(", "                timing['world_model']=stamp()-t; t=stamp()\n                stats=validation(")
replace_once("                actor_loss=None; critic_loss=None", "                timing['validation_and_gate']=stamp()-t; t=stamp()\n                actor_loss=None; critic_loss=None\n                timing['actor']=0.; timing['critic']=0.")
replace_once("                    for q in model.parameters():q.requires_grad_(True)", "                    timing['actor']=stamp()-t; t=stamp()\n                    for q in model.parameters():q.requires_grad_(True)")
replace_once("                data={'cycle':cycle+1", "                    timing['critic']=stamp()-t; t=stamp()\n                data={'cycle':cycle+1")
replace_once("            print(f'训练文件：{run}',flush=True)", "                timing['log_and_save']=stamp()-t\n                timing['total']=stamp()-start_cycle\n                timing['cycle']=cycle+1\n                with (run/'timings.jsonl').open('a') as f: f.write(json.dumps(timing)+'\\n')\n            print(f'训练文件：{run}',flush=True)")
exec(compile(source, str(Path(__file__).with_name("train.py")), "exec"), {"__name__": "__main__"})
