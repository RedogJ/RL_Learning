"""Newton GL 高度读数：仅显示环境 0，与奖励使用同一个根节点高度定义。"""

import time


class HeightDisplay:
    def __init__(self, env, visualizer):
        self.env = env
        self.height = None
        self.next_refresh = 0.0
        # panel 位于左侧栏顶部，不依赖 Simulation 折叠项是否展开。
        visualizer.register_ui_callback(self.draw, position="panel")

    def draw(self, imgui):
        if self.env is None:
            return
        now = time.monotonic()
        if now >= self.next_refresh:
            # 先在设备上作差，再取一个标量；不传回全部环境状态。
            # UI 读数会等待这一个 CUDA 标量，因此限频，且不接入训练循环。
            env = self.env
            self.height = (env.robot.data.root_link_pos_w.torch[0, 2]
                           - env.scene.env_origins[0, 2]).item()
            self.next_refresh = now + 0.1
        imgui.text("Go2 height | env 0")
        imgui.text(f"Height: {self.height:.3f} m")
        imgui.separator()

    def close(self):
        # Viewer 随 SimulationContext 释放；提前停止回调对环境的访问。
        self.env = None


def attach_height_display(env):
    for visualizer in env.sim.visualizers:
        if (visualizer.cfg.visualizer_type == "newton_gl"
                and getattr(visualizer, "_viewer", None) is not None
                and not getattr(visualizer, "_runtime_headless", False)):
            return HeightDisplay(env, visualizer)
    return None
