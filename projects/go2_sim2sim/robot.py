"""加载官方 MJCF，保留被动关节参数，显式对齐关节映射、力矩限幅和接触摩擦。"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import FRICTION, MODEL_ROOT, TORQUE_LIMIT


def load_model(policy, physics_hz):
    source = MODEL_ROOT / "go2.xml"
    if not source.is_file():
        raise FileNotFoundError("缺少 Go2 MJCF；先执行 bash run.sh go2-sim2sim fetch-model")
    root = ET.parse(source).getroot()
    root.find("compiler").set("meshdir", str(MODEL_ROOT / "assets"))
    option = root.find("option")
    option.set("timestep", str(1 / physics_hz))
    option.set("integrator", "implicitfast")
    # 官方默认场景含障碍物；这里在机器人模型中直接建立独立的纯平地。
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker", width="256", height="256",
                  rgb1="0.22 0.30 0.37", rgb2="0.40 0.48 0.55")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="10 10",
                  texuniform="true", reflectance="0.05")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .1", material="floor_mat",
                  friction=f"{FRICTION} .005 .0001", contype="2", conaffinity="1")
    ET.SubElement(world, "light", directional="true", pos="0 0 3", dir="-.5 -.3 -1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in policy.joint_names])
    if (joint_ids < 0).any() or len(set(joint_ids)) != 12:
        raise ValueError("MJCF 缺少检查点中的关节或关节重复")
    if not np.allclose(model.jnt_range[joint_ids], policy.limits, atol=2e-5, rtol=0):
        raise ValueError("MJCF 与检查点的机械关节限位不一致")
    # 保留 MJCF 自带的 damping、frictionloss 和 armature，不在加载后覆盖。
    actuator_ids = []
    for joint in joint_ids:
        ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(ids) != 1:
            raise ValueError(f"关节 {joint} 必须对应一个电机")
        actuator_ids.append(ids[0])
    actuator_ids = np.array(actuator_ids)
    if not np.allclose(model.actuator_gear[actuator_ids, 0], 1.):
        raise ValueError("当前 PD 控制要求电机 gear=1")
    model.actuator_ctrlrange[actuator_ids] = [-TORQUE_LIMIT, TORQUE_LIMIT]
    model.actuator_ctrllimited[actuator_ids] = True
    # 与 enabled_self_collisions=False 对应：机器人只与地面碰撞。
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != 0 and (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            model.geom_contype[geom], model.geom_conaffinity[geom] = 1, 2
            model.geom_friction[geom] = [FRICTION, .005, .0001]
    return model, joint_ids, actuator_ids
