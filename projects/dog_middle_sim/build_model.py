"""将 middle 原始 URDF 转换成独立 MJCF；原文件始终只读。"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/dog_middle"
SOURCE = next(ASSETS.glob("*/result_round1/urdf_with_motors/*__with_motors.urdf"))
OUTPUT = ASSETS / "simulation/middle.xml"
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = tuple(f"{leg}_{suffix}" for leg in LEGS
               for suffix in ("UP_joint1", "UP_joint2", "LOW_joint"))
# 仅为仿真试验值，须按实际脚掌尺寸校准。
FOOT_RADIUS = 0.02


def fmt(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def xyz(element):
    if element is None:
        return np.zeros(3)
    if not np.allclose(np.fromstring(element.get("rpy", "0 0 0"), sep=" "), 0):
        raise ValueError("此转换器仅适配当前 middle 的零 rpy 结构，不能静默处理其他模型")
    return np.fromstring(element.get("xyz", "0 0 0"), sep=" ")


def stl_vertices(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                      ("attribute", "<u2")])
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"预期二进制 STL：{path}")
    return np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)


def placeholder(link):
    """仅识别此导出器给无几何中间/脚端节点填的 1 kg、单位惯性。"""
    if link.find('visual') is not None or link.find('collision') is not None:
        return False
    inertial=link.find('inertial')
    if inertial is None:return False
    inertia=inertial.find('inertia')
    return (float(inertial.find('mass').get('value'))==1.0
            and np.allclose(xyz(inertial.find('origin')),0)
            and all(float(inertia.get(k))==1.0 for k in ('ixx','iyy','izz'))
            and all(float(inertia.get(k))==0.0 for k in ('ixy','ixz','iyz')))


def build():
    source = ET.parse(SOURCE).getroot()
    links = {e.get("name"): e for e in source.findall("link")}
    joints = {e.get("name"): e for e in source.findall("joint")}
    ignored=[name for name,link in links.items() if placeholder(link)]
    doc = ET.Element("mujoco", model="dog_middle")
    ET.SubElement(doc, "compiler", angle="radian", autolimits="true", inertiafromgeom="false")
    ET.SubElement(doc, "option", timestep="0.002", integrator="implicitfast",
                  gravity="0 0 -9.81", cone="elliptic", iterations="50")
    ET.SubElement(doc, "size", njmax="512", nconmax="128")
    defaults = ET.SubElement(doc, "default")
    ET.SubElement(defaults, "joint", damping="0.3", armature="0.002", frictionloss="0.02")
    ET.SubElement(defaults, "geom", friction="0.8 0.005 0.0001", condim="3")
    assets = ET.SubElement(doc, "asset")
    ET.SubElement(assets, "texture", name="checker", type="2d", builtin="checker",
                  rgb1="0.2 0.24 0.28", rgb2="0.7 0.74 0.78", width="512", height="512")
    ET.SubElement(assets, "material", name="ground", texture="checker", texrepeat="4 4",
                  texuniform="true", reflectance="0")
    world = ET.SubElement(doc, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 0.1",
                  material="ground", contype="2", conaffinity="1")
    body = ET.SubElement(world, "body", name="BODY", pos="0 0 0.3")
    ET.SubElement(body, "freejoint", name="root")
    toe_points = {}

    def add_link(parent, name):
        link = links[name]
        inertia = link.find("inertial")
        if inertia is None:
            raise ValueError(f"实体 link 缺少惯性：{name}")
        i = inertia.find("inertia")
        ET.SubElement(parent, "inertial", pos=fmt(xyz(inertia.find("origin"))),
                      mass=inertia.find("mass").get("value"),
                      fullinertia=" ".join(i.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        # 复用原网格与偏移；碰撞由 MuJoCo 对该网格的凸包计算。
        for tag, group, collision in (("visual", "2", False), ("collision", "3", True)):
            element = link.find(tag)
            mesh = element.find("geometry/mesh")
            mesh_path = SOURCE.parent / mesh.get("filename")
            mesh_name = name
            if tag == "visual":
                import os
                ET.SubElement(assets, "mesh", name=mesh_name,
                              file=os.path.relpath(mesh_path, OUTPUT.parent))
            ET.SubElement(parent, "geom", name=f"{name}_{tag}", type="mesh", mesh=mesh_name,
                          pos=fmt(xyz(element.find("origin"))), group=group,
                          contype="1" if collision else "0",
                          conaffinity="2" if collision else "0",
                          rgba=("0.18 0.20 0.23 1" if '_motor_' in name else "0.68 0.47 0.28 1")
                          if not collision else "1 0.3 0.1 0.25")
        if name.endswith("_LOW"):
            # 根据脚底网格选一个用于初始化、诊断及放置试验脚掌的 site。
            element = link.find("collision")
            vertices = stl_vertices(SOURCE.parent / element.find("geometry/mesh").get("filename"))
            vertices = vertices + xyz(element.find("origin"))
            lowest = vertices[vertices[:, 2] < vertices[:, 2].min() + 0.004]
            point = lowest.mean(axis=0)
            point[2] = vertices[:, 2].min()
            toe_points[name[:2]] = point
            ET.SubElement(parent, "site", name=f"{name[:2]}_sole", pos=fmt(point),
                          size="0.006", rgba="0 0.8 0.9 1", group="4")
            # 在局部 Z 方向抬高半径的一半，减小球面向下凸出的距离。
            center = point + np.array([0., 0., FOOT_RADIUS / 2])
            ET.SubElement(parent, "geom", name=f"{name[:2]}_foot_sphere", type="sphere",
                          pos=fmt(center), size=str(FOOT_RADIUS), group="2",
                          contype="1", conaffinity="2", rgba="0.08 0.08 0.08 1")
        # 电机外壳为固定实体：保留完整安装旋转、网格、碰撞和声明惯性。
        for joint in joints.values():
            child=joint.find('child').get('link')
            if joint.get('type')=='fixed' and joint.find('parent').get('link')==name and '_motor_' in child:
                origin=joint.find('origin')
                position=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
                quat=Rotation.from_euler('xyz',rpy).as_quat()[[3,0,1,2]]
                motor_body=ET.SubElement(parent,'body',name=child,pos=fmt(position),quat=fmt(quat))
                add_link(motor_body,child)

    add_link(body, "BODY")
    for leg in LEGS:
        first = joints[f"{leg}_UP_joint1"]
        second = joints[f"{leg}_UP_joint2"]
        knee = joints[f"{leg}_LOW_joint"]
        intermediate = links[first.find("child").get("link")]
        if ((list(intermediate) and not placeholder(intermediate))
                or second.find("parent").get("link") != intermediate.get("name")):
            raise ValueError("仅能折叠当前无实体、无惯性的中间 link")
        offset = xyz(second.find("origin"))
        upper = ET.SubElement(body, "body", name=f"{leg}_UP",
                              pos=fmt(xyz(first.find("origin")) + offset))
        # 同一 body 上的两个 hinge 保留顺序、轴向和不共点的转轴；不改为 ball joint。
        for joint, position in ((first, -offset), (second, np.zeros(3))):
            limit = joint.find("limit")
            ET.SubElement(upper, "joint", name=joint.get("name"), type="hinge",
                          pos=fmt(position), axis=joint.find("axis").get("xyz"),
                          range=f"{limit.get('lower')} {limit.get('upper')}")
        add_link(upper, f"{leg}_UP")
        lower = ET.SubElement(upper, "body", name=f"{leg}_LOW", pos=fmt(xyz(knee.find("origin"))))
        limit = knee.find("limit")
        ET.SubElement(lower, "joint", name=knee.get("name"), type="hinge",
                      axis=knee.find("axis").get("xyz"),
                      range=f"{limit.get('lower')} {limit.get('upper')}")
        add_link(lower, f"{leg}_LOW")
        for fixed in source.findall("joint"):
            if fixed.get("type") == "fixed" and fixed.find("parent").get("link") == f"{leg}_LOW":
                if '_motor_' in fixed.find('child').get('link'):continue
                child_link=links[fixed.find('child').get('link')]
                if child_link.find('inertial') is not None and not placeholder(child_link):
                    raise ValueError('脚端新增了真实惯性，需重新评估转换规则')
                ET.SubElement(lower, "site", name=fixed.find("child").get("link"),
                              pos=fmt(xyz(fixed.find("origin"))), size="0.004", group="4")
    actuators = ET.SubElement(doc, "actuator")
    for name in JOINTS:
        effort = joints[name].find("limit").get("effort")
        ET.SubElement(actuators, "motor", name=f"{name}_motor", joint=name,
                      gear="1", ctrllimited="true", ctrlrange=f"-{effort} {effort}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(doc)
    ET.ElementTree(doc).write(OUTPUT, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    data = mujoco.MjData(model)
    qids = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
    site_ids = [model.site(f"{leg}_sole").id for leg in LEGS]
    # 仅是仿真站姿初始化：机身水平、四个脚底到地面、脚下支撑区域覆盖质心。
    targets = np.array([[0.11, -0.36, 0], [-0.11, -0.36, 0],
                        [0.11, 0.10, 0], [-0.11, 0.10, 0]])
    def residual(q):
        data.qpos[:3] = (0, 0, 0.275)
        data.qpos[3:7] = (1, 0, 0, 0)
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        return np.r_[((data.site_xpos[site_ids] - targets) * 20).ravel(), 0.03 * q]
    result = least_squares(residual, np.tile([0, -0.5, 0.8], 4), bounds=(-1.55, 1.55))
    residual(result.x)
    # 平移整个机器人，确保所有碰撞网格在初始帧不穿地。
    collision_min = []
    for leg_or_body,link in links.items():
        if link.find('collision') is None:continue
        geom_id = model.geom(f"{leg_or_body}_collision").id
        col = link.find("collision")
        vertices = stl_vertices(SOURCE.parent / col.find("geometry/mesh").get("filename"))
        vertices = vertices + xyz(col.find("origin"))
        bid = model.body(leg_or_body).id
        world_vertices = vertices @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
        collision_min.append(world_vertices[:, 2].min())
    data.qpos[2] += 0.001 - min(collision_min)
    reset = data.qpos.copy()
    key = ET.SubElement(doc, "keyframe")
    ET.SubElement(key, "key", name="stand", qpos=fmt(reset), qvel=fmt(np.zeros(model.nv)),
                  ctrl=fmt(np.zeros(model.nu)))
    ET.indent(doc)
    ET.ElementTree(doc).write(OUTPUT, encoding="unicode")
    report = {"source": str(SOURCE.relative_to(ROOT)), "joint_names": JOINTS,
              "motor_body_count":sum('_motor_' in name for name in links),
              "excluded_placeholder_links":ignored,
              "mass_kg": float(model.body_mass.sum()), "initial_height_m": float(reset[2]),
              "stand_angles_rad": dict(zip(JOINTS, result.x.tolist())),
              "notes": ["9个躯干/腿实体与16个电机外壳采用源文件声明质量和惯性；尚未经实机核验",
                        "排除8个无几何节点的1 kg/单位惯性占位参数，不代表确认实机连接件无质量",
                        "4个无质量中间link转换为同一body上的顺序双hinge；不额外填质量",
                        "原始STL用作外观和凸包碰撞；初期关闭机器人自碰撞",
                        "阻尼/摩擦/armature和PD为仿真初始化参数，不代表实机"]}
    OUTPUT.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"模型已生成：{OUTPUT}\n12 joints, 12 motors, mass={report['mass_kg']:.6f} kg")
    return OUTPUT


if __name__ == "__main__":
    build()
