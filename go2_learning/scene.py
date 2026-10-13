"""本地 Go2 USD、静态地面和关节 PD 配置。"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

USD_PATH = Path(__file__).resolve().parents[1] / "assets/unitree_model/Go2/usd/go2.usd"
GROUND_PATH = Path(__file__).resolve().parents[1] / "assets/isaaclab/ground_plane/default_ground_plane.usda"


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
            rigid_props=PhysxRigidBodyCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
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
