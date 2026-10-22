"""本地 Go2 USD、静态地面和关节 PD 配置。"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from go2_learning.paths import USD_PATH, GROUND_PATH


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    # 官方棋盘地面：本地 USD 和贴图，碰撞平面位于 z=0。
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            usd_path=str(GROUND_PATH),
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                # 四条腿采用相同参考角，与髋角趋零奖励保持一致。
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.6,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                stiffness=25.0,
                damping=0.5,
                # 入门场景采用统一的保守力矩上限，非真实电机的完整模型。
                joint_effort_limit=23.7,
            ),
        },
    )
    # 更新周期在 make_env 中设为物理 dt；4 帧覆盖 30 Hz 控制之间的全部物理步。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]+",
        history_length=4,
    )
