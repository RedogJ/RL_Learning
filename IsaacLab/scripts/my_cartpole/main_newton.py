import argparse
import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_newton.physics import NewtonCfg, NewtonManager

# 读取命令行参数，例如 --viz none
parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()

# 配置物理时间步和物理引擎
sim_cfg = SimulationCfg(
    dt=1 / 120,
    physics=NewtonCfg(),
)

# 准备运行所需的仿真环境
with launch_simulation(sim_cfg, args):
    sim = SimulationContext(sim_cfg)

    try:
        # 创建物理场景的构建器
        builder = sim.physics_manager.create_builder()

        # 添加一个可自由运动的刚体，并给它球形几何
        body = builder.add_body(
            mass=1.0,
            xform=wp.transform(
                (0.0, 0.0, 2.0),
                wp.quat_identity(),
        ),
)
        builder.add_shape_sphere(body=body, radius=0.1)

        # 将构建好的场景交给 Newton
        builder.add_ground_plane()
        NewtonManager.set_builder(builder)

        sim.reset()
        print("仿真已初始化")

        for step in range(600):
            sim.step(render=False)
            if (step + 1) % 120 == 0:
                sim_time = (step + 1) * sim_cfg.dt

                state = NewtonManager.get_state()
                body_poses = state.body_q.numpy()
                position = body_poses[body, :3]

                print(
                    f"时间：{sim_time:.1f} 秒，"
                    f"位置：x={position[0]:.3f}，"
                    f"y={position[1]:.3f}，"
                    f"z={position[2]:.3f}"
                )

        print("仿真运行结束")

    finally:
        SimulationContext.clear_instance()