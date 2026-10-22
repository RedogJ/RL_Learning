"""复用 Newton Viewer 官方 Picking，将鼠标拉力接到 PhysX。"""

import numpy as np
import torch
import time


class MousePush:
    # 使用官方的弹簧阻尼交互方式，增益针对本工程设定。
    stiffness = 200.0  # N/m，鼠标目标与实际点击点之间的弹簧。
    damping = 10.0  # N·s/m，抑制被拖动点的振荡。
    max_force = 150.0
    arrow_meters_per_newton = 0.01  # 50 N 显示为 0.5 m 长的箭头。
    arrow_name = "go2_mouse_force"
    arrow_hz = 30.0

    def __init__(self, env, viewer):
        from pyglet.event import EVENT_HANDLED
        from pyglet.window import mouse

        self.handled, self.right_button = EVENT_HANDLED, mouse.RIGHT
        self.env, self.viewer = env, viewer
        self.window = viewer.renderer.window
        from isaaclab_newton.physics import NewtonManager

        # Viewer 已创建 Picking，即使 PhysX 模式关闭了它的默认力回调。
        self.picking = viewer.picking
        # Newton 的 body_label 有时指向关节 prim，先按官方映射还原刚体路径。
        self.body_paths = NewtonManager._resolve_scene_data_body_paths(
            list(viewer.model.body_label), env.scene.stage,
        )
        self.native_body = -1
        self.body_ids = []
        self.picked_env = 0
        self.local_point = None
        self.env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        self.force = torch.zeros((1, 1, 3), device=env.device)
        self.target = torch.zeros((1, 3), device=env.device)
        self.anchor = None
        self.arrow_visible = False
        self.pending_mouse = None
        self.last_arrow_time = float("-inf")
        self.arrow_offset = np.zeros(3, dtype=np.float32)
        import warp as wp

        # 重用显存，不为每次箭头更新重新分配 Warp 数组。
        self.arrow_starts = wp.zeros(1, dtype=wp.vec3, device=viewer.device)
        self.arrow_ends = wp.zeros(1, dtype=wp.vec3, device=viewer.device)
        self.arrow_colors = wp.array([[1.0, 0.25, 0.05]], dtype=wp.vec3, device=viewer.device)
        self.window.push_handlers(self)
        print("[交互] 官方 Viewer 拾取：在 Go2 上右键拖拽，弹簧力作用于点击点（最大 150 N）。")

    def on_mouse_press(self, x, y, button, modifiers):
        if button == self.right_button:
            self.clear()
            gui = getattr(self.viewer, "gui", None)
            if gui is not None and gui.should_ignore_mouse_input():
                return
            if not self.pick(x, y):
                return self.handled
            self.anchor = (x, y)
            self.force.zero_()
            self.hide_arrow()
            return self.handled

    def screen_ray(self, x, y):
        # 与官方 ViewerGui.start_picking_from_screen 完全相同的坐标入口。
        return self.viewer.camera.get_world_ray(*self.viewer._to_framebuffer_coords(x, y))

    def pick(self, x, y):
        """直接使用官方拾取器，对窗口最近显示的模型状态做射线求交。"""
        state = self.viewer._last_state
        if state is None:
            return False
        self.picking.pick(state, *self.screen_ray(x, y))
        if not self.picking.is_picking():
            return False
        self.native_body = int(self.picking.pick_body.numpy()[0])
        body_path = self.body_paths[self.native_body]
        if not body_path:
            self.picking.release()
            return False
        body_name = body_path.rsplit("/", 1)[-1]
        if body_name not in self.env.robot.body_names:
            self.picking.release()
            return False  # 地面等非机器人对象不施力。
        for env_id, env_path in enumerate(self.env.scene.env_prim_paths):
            if body_path.startswith(env_path + "/Robot/"):
                self.picked_env = env_id
                self.env_ids[0] = env_id
                self.body_ids = [self.env.robot.body_names.index(body_name)]
                pick_state = self.picking.pick_state.numpy()[0]
                self.target.copy_(torch.as_tensor(pick_state["picking_target_world"].copy(), device=self.env.device))
                # Isaac Lab 用该刚体 link 的 transform 同步 Newton 渲染模型；
                # 直接保留官方拾取器算出的局部点，不混用下一物理帧的位姿。
                self.local_point = torch.tensor(
                    pick_state["picked_point_local"].copy(), device=self.env.device,
                ).reshape(1, 3)
                self.arrow_offset = self.display_offset()
                return True
        self.picking.release()
        return False

    def body_pose(self):
        data = self.env.robot.data
        return (
            data.body_link_pos_w.torch[self.picked_env, self.body_ids],
            data.body_link_quat_w.torch[self.picked_env, self.body_ids],
        )

    def application_point(self):
        from isaaclab.utils.math import quat_apply

        pos, quat = self.body_pose()
        # 固定在该刚体表面的点；刚体平移或旋转时，世界坐标随之变化。
        return pos + quat_apply(quat, self.local_point)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.anchor is not None and buttons & self.right_button:
            # 一帧可能收到很多鼠标事件，只保留最后一个位置，下一物理步处理。
            self.pending_mouse = (x, y)
            return self.handled

    def on_mouse_release(self, x, y, button, modifiers):
        if button == self.right_button:
            self.clear()
            return self.handled

    def on_deactivate(self):
        self.clear()

    def on_mouse_leave(self, x, y):
        self.clear()

    def clear(self):
        self.anchor = None
        self.local_point = None
        self.pending_mouse = None
        self.last_arrow_time = float("-inf")
        self.picking.release()
        self.force.zero_()
        self.hide_arrow()

    def hide_arrow(self):
        if self.arrow_visible and not self.viewer.renderer.has_exit():
            self.viewer.log_arrows(self.arrow_name, None, None, None)
        self.arrow_visible = False

    def update_force(self):
        point = self.application_point()
        data = self.env.robot.data
        com = data.body_com_pos_w.torch[self.picked_env, self.body_ids]
        linear = data.body_com_lin_vel_w.torch[self.picked_env, self.body_ids]
        angular = data.body_com_ang_vel_w.torch[self.picked_env, self.body_ids]
        point_velocity = linear + torch.linalg.cross(angular, point - com)
        force = self.stiffness * (self.target - point) - self.damping * point_velocity
        scale = (self.max_force / force.norm().clamp_min(1e-6)).clamp(max=1.0)
        self.force[0].copy_(force * scale)
        return point

    def display_offset(self):
        offsets = self.viewer.world_offsets
        worlds = self.viewer.model.body_world
        if offsets is not None and offsets.shape[0] and worlds is not None:
            world = int(worlds.numpy()[self.native_body])
            if 0 <= world < offsets.shape[0]:
                return offsets.numpy()[world]
        return np.zeros(3, dtype=np.float32)

    def update_arrow(self, point):
        if self.viewer.renderer.has_exit():
            return
        now = time.perf_counter()
        if now - self.last_arrow_time < 1.0 / self.arrow_hz:
            return
        self.last_arrow_time = now
        # 合并成一次 GPU→CPU 读回，且只按显示频率执行。
        packed = torch.cat((point, self.force[0]), dim=0).detach().cpu().numpy()
        start = packed[:1] + self.arrow_offset
        force = packed[1:]
        if self.anchor is None or np.linalg.norm(force) < 1e-6:
            self.hide_arrow()
            return
        # 箭头与实际外力共用同一个随刚体运动的点击点。
        end = start + self.arrow_meters_per_newton * force
        self.arrow_starts.assign(start)
        self.arrow_ends.assign(end)
        self.viewer.log_arrows(
            self.arrow_name,
            self.arrow_starts, self.arrow_ends, self.arrow_colors,
        )
        self.arrow_visible = True

    def apply(self):
        if not self.env.policy_active[self.picked_env]:
            self.force.zero_()
            self.hide_arrow()
            return
        if self.anchor is not None:
            if self.pending_mouse is not None:
                # 保留官方目标更新算法，将同一帧的鼠标事件合并。
                self.picking.update(*self.screen_ray(*self.pending_mouse))
                target = self.picking.pick_state.numpy()[0]["picking_target_world"].copy()
                self.target.copy_(torch.as_tensor(target, device=self.env.device))
                self.pending_mouse = None
            # 物体靠近鼠标目标后拉力减小，物体速度进入阻尼项。
            point = self.update_force()
            # 瞬时外力每个物理步都重新写入，释放后下一步不再施加。
            self.env.robot.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=self.force, positions=point.unsqueeze(0),
                body_ids=self.body_ids, env_ids=self.env_ids, is_global=True,
            )
            # positions 使 PhysX 同时计入相对质心的力矩 r × F。
            self.update_arrow(point)

    def close(self):
        self.clear()
        self.window.remove_handlers(self)


def attach_mouse_push(env):
    """只挂接已初始化的 GL 窗口；无窗口模式不导入 pyglet。"""
    for visualizer in env.sim.visualizers:
        if visualizer.cfg.visualizer_type == "newton_gl":
            # Isaac Lab 暂未公开 Viewer 对象；版本相关访问集中在此处。
            viewer = getattr(visualizer, "_viewer", None)
            if viewer is not None and not getattr(visualizer, "_runtime_headless", False):
                return MousePush(env, viewer)
    return None
