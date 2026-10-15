"""GL 窗口方向键输入检测；独立于环境、奖励和策略网络。"""


class KeyboardInput:
    def __init__(self, window, test_only=True):
        from pyglet.event import EVENT_HANDLED
        from pyglet.window import key

        self.window = window
        self.handled = EVENT_HANDLED
        self.key_names = {key.UP: "UP", key.DOWN: "DOWN", key.LEFT: "LEFT", key.RIGHT: "RIGHT"}
        self.pressed = set()
        self.label = "键盘检测" if test_only else "键盘控制"
        self.window.push_handlers(self)
        help_text = "当前仅检测输入" if test_only else "上下控制前后速度，左右控制转向；松键为零速度指令，策略尝试停稳"
        print(f"[{self.label}] 请点击仿真窗口取得焦点；{help_text}。", flush=True)
        self._report("就绪")

    @property
    def axes(self):
        """方向信号 (上下, 左右)，每轴为 -1/0/+1；此处尚未定义移动速度。"""
        return (
            int("UP" in self.pressed) - int("DOWN" in self.pressed),
            int("LEFT" in self.pressed) - int("RIGHT" in self.pressed),
        )

    def _report(self, event):
        held = "+".join(name for name in self.key_names.values() if name in self.pressed) or "无"
        up_down, left_right = self.axes
        print(f"[{self.label}] {event} | 按住: {held} | 上下={up_down:+d} 左右={left_right:+d}", flush=True)

    def on_key_press(self, symbol, modifiers):
        name = self.key_names.get(symbol)
        if name is not None:
            if name not in self.pressed:
                self.pressed.add(name)
                self._report(f"按下 {name}")
            return self.handled  # 方向键由本模块接收，其余键继续交给 Viewer。

    def on_key_release(self, symbol, modifiers):
        name = self.key_names.get(symbol)
        if name is not None:
            if name in self.pressed:
                self.pressed.remove(name)
                self._report(f"松开 {name}")
            return self.handled

    def clear(self):
        if self.pressed:
            self.pressed.clear()
            self._report("清空按键")

    def on_deactivate(self):
        # 切到终端后可能收不到松键事件；失去焦点时主动归零。
        # 不返回 EVENT_HANDLED，让鼠标模块也收到失焦事件并清除拉力。
        self.clear()

    def on_close(self):
        self.clear()

    def close(self):
        self.clear()
        self.window.remove_handlers(self)


def attach_keyboard_input(sim, test_only=True):
    """连接已创建的 newton_gl 窗口，不创建全局键盘监听。"""
    for visualizer in sim.visualizers:
        if visualizer.cfg.visualizer_type == "newton_gl":
            viewer = getattr(visualizer, "_viewer", None)
            if viewer is not None and not getattr(visualizer, "_runtime_headless", False):
                return KeyboardInput(viewer.renderer.window, test_only)
    raise RuntimeError("键盘检测需要 newton_gl 窗口：请使用 view --viz newton_gl --keyboard_test")
