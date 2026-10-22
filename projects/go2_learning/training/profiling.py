"""可选短程 CPU/CUDA 时间线；只在诊断运行中包装方法，不逐阶段强制同步。"""
from contextlib import contextmanager
from pathlib import Path
from functools import wraps

import torch


@contextmanager
def profile_training(env, algorithm, log_dir):
    output = Path(log_dir) / "profile"
    output.mkdir(parents=True, exist_ok=True)
    patched = []
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def save_trace(prof):
        prof.export_chrome_trace(str(output / "trace.json"))
        averages = prof.key_averages()
        (output / "cpu.txt").write_text(averages.table(sort_by="self_cpu_time_total", row_limit=100))
        if torch.cuda.is_available():
            (output / "cuda.txt").write_text(averages.table(sort_by="self_device_time_total", row_limit=100))

    # 以控制步推进：等待 24 步、预热 24 步、记录 24 步，只记录一次。
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=24, warmup=24, active=24, repeat=1),
        on_trace_ready=save_trace,
        record_shapes=False, profile_memory=False, with_stack=False,
    ) as prof:
        def patch(obj, name, label, tick=False):
            original = getattr(obj, name)
            owned = name in vars(obj)
            saved = vars(obj).get(name)
            @wraps(original)
            def wrapped(*args, **kwargs):
                with torch.profiler.record_function(label):
                    result = original(*args, **kwargs)
                if tick:
                    prof.step()
                return result
            setattr(obj, name, wrapped)
            patched.append((obj, name, owned, saved))
        try:
            for name in ("_set_action", "_advance_simulation", "_read_state", "_compute_reward",
                         "_make_extras", "_reset_finished_envs", "reset", "get_observations"):
                patch(env, name, "env/" + name)
            patch(env.sim, "step", "physics/step")
            patch(env.scene, "write_data_to_sim", "physics/write")
            patch(env.scene, "update", "physics/update")
            if env.randomization is not None:
                patch(env.randomization, "apply", "physics/random_push")
            for name in ("act", "process_env_step", "compute_returns", "update"):
                patch(algorithm, name, "ppo/" + name)
            patch(env, "step", "env/control_step", tick=True)
            yield
        finally:
            for obj, name, owned, saved in reversed(patched):
                if owned:
                    setattr(obj, name, saved)
                else:
                    delattr(obj, name)
    print(f"性能剖析已写入：{output}；诊断运行的耗时不能直接作为正常训练吞吐。")
