"""检查双髋轴转换的运动学等价性，以及浮动基座和驱动器配置。"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .build_model import SOURCE, OUTPUT, JOINTS, placeholder


def main():
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    data = mujoco.MjData(model)
    source = ET.parse(SOURCE).getroot()
    assert (model.nq, model.nv, model.nu) == (19, 18, 12)
    source_mass = sum(float(link.find('inertial/mass').get('value')) for link in source.findall('link')
                      if link.find('inertial/mass') is not None and not placeholder(link))
    np.testing.assert_allclose(model.body_mass.sum(), source_mass, atol=1e-10)
    for i, name in enumerate(JOINTS):
        assert model.actuator_trnid[i, 0] == model.joint(name).id
        assert model.actuator_gear[i, 0] == 1
    qids = [model.jnt_qposadr[model.joint(name).id] for name in JOINTS]
    rng = np.random.default_rng(42)
    errors = []
    for _ in range(32):
        angles = dict(zip(JOINTS, rng.uniform(-1.3, 1.3, 12)))
        root = np.eye(4)
        rotation = Rotation.random(random_state=rng)
        root[:3, :3] = rotation.as_matrix()
        root[:3, 3] = rng.uniform(-1, 1, 3)
        xyzw = rotation.as_quat()
        data.qpos[:3] = root[:3, 3]
        data.qpos[3:7] = xyzw[[3, 0, 1, 2]]
        data.qpos[qids] = [angles[n] for n in JOINTS]
        mujoco.mj_forward(model, data)
        poses = {"BODY": root}
        pending = list(source.findall("joint"))
        while pending:
            progressed = False
            for joint in pending[:]:
                parent = joint.find("parent").get("link")
                if parent not in poses:
                    continue
                transform = np.eye(4)
                origin=joint.find('origin')
                transform[:3, 3] = np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                transform[:3, :3] = Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
                if joint.get("type") == "revolute":
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * angles[joint.get("name")]).as_matrix()
                child = joint.find("child").get("link")
                poses[child] = poses[parent] @ transform
                pending.remove(joint)
                progressed = True
            assert progressed, "URDF 连接树无法解析"
        for name, expected in poses.items():
            if "virtual" in name:
                continue
            if "foot" in name:
                actual = data.site_xpos[model.site(name).id]
                errors.append(np.max(np.abs(actual - expected[:3, 3])))
            else:
                bid = model.body(name).id
                errors.append(np.max(np.abs(data.xpos[bid] - expected[:3, 3])))
                errors.append(np.max(np.abs(data.xmat[bid].reshape(3, 3) - expected[:3, :3])))
    assert max(errors) < 1e-9, max(errors)
    print(f"PASS：32 组随机关节/根位姿，URDF 与 MJCF 位姿最大元素误差 {max(errors):.3g}；"
          "12 转动自由度、6 基座自由度、12 驱动器；有效实体（含电机）质量保持不变。")


if __name__ == "__main__":
    main()
