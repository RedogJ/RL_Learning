"""同一扩散策略的交互目标切换，不切模型、不重置物理状态。"""
from go2_learning.control.attitude_integral import AttitudeIntegral


def apply_keyboard_commands(env, axes, cfg, skill):
    env.commands.values.zero_()
    if skill == 'running':
        forward, turn = axes
        env.commands.values[:, 0] = forward * (cfg.keyboard_forward if forward >= 0 else cfg.keyboard_backward)
        env.commands.values[:, 2] = turn * cfg.keyboard_turn


class SwitchingTask:
    def __init__(self, tasks, initial):
        self.tasks, self.skill = tasks, initial
        # 共享物理场景和失败后的重置姿态，始终取启动时的任务。
        self.initial = tasks[initial]

    def __getattr__(self, name):
        return getattr(self.initial, name)

    @property
    def target_height(self):
        return self.tasks[self.skill].target_height

    def failures(self, state):
        # 接受两个任务有效姿态区域的并集，避免尚未到达新姿态就被终止。
        return self.tasks['running'].failures(state) & self.tasks['upright'].failures(state)

    def tilt_failures(self, state):
        return self.tasks['running'].tilt_failures(state) & self.tasks['upright'].tilt_failures(state)

    def select(self, skill, env):
        if skill not in self.tasks:
            raise ValueError(f'未知技能：{skill}')
        if skill == self.skill:
            return False
        self.skill = skill
        # running 教师使用积分，upright 教师使用零占位；仅重建这三个软件观测。
        # 不调用 env.reset，不改根状态、关节状态、动作、回合计时或 diffusion 历史。
        env.attitude_integral = AttitudeIntegral(
            env.num_envs, env.device, env.dt, self.tasks[skill].cfg.integral)
        return True


class SkillKeyboard:
    def __init__(self, window, initial):
        from pyglet.window import key
        from pyglet.event import EVENT_HANDLED
        self.window, self.handled = window, EVENT_HANDLED
        self.keys = {key._1: 'running', key._2: 'upright', key.TAB: 'toggle'}
        self.requested_skill, self.pressed = initial, set()
        window.push_handlers(self)
        print('[技能切换] 点击窗口取得焦点；Tab 切换，1 running，2 upright。'
              '切换不重置；失败/超时仍回到启动姿态。', flush=True)

    def on_key_press(self, symbol, modifiers):
        choice = self.keys.get(symbol)
        if choice is not None:
            if symbol not in self.pressed:
                self.pressed.add(symbol)
                self.requested_skill = (('upright' if self.requested_skill == 'running' else 'running')
                                        if choice == 'toggle' else choice)
            return self.handled

    def on_key_release(self, symbol, modifiers):
        if symbol in self.keys:
            self.pressed.discard(symbol)
            return self.handled

    def on_deactivate(self):
        self.pressed.clear()

    def on_close(self):
        self.pressed.clear()

    def close(self):
        self.pressed.clear()
        self.window.remove_handlers(self)


def attach_skill_keyboard(sim, initial):
    for visualizer in sim.visualizers:
        if visualizer.cfg.visualizer_type == 'newton_gl':
            viewer = getattr(visualizer, '_viewer', None)
            if viewer is not None and not getattr(visualizer, '_runtime_headless', False):
                return SkillKeyboard(viewer.renderer.window, initial)
    raise RuntimeError('技能切换需要已打开的 newton_gl 窗口')
