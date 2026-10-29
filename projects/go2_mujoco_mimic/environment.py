"""Go2 MuJoCo 场景与明确关节顺序；不依赖训练检查点。"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'assets/unitree_model/Go2/mjcf'
JOINT_NAMES = tuple(f'{leg}_{part}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
                    for part in ('hip', 'thigh', 'calf'))


class Go2Environment:
    def __init__(self, physics_hz=600, control_hz=30):
        if physics_hz <= 0 or control_hz <= 0 or physics_hz % control_hz:
            raise ValueError('physics_hz 必须是 control_hz 的正整数倍')
        root = ET.parse(MODEL_DIR / 'go2.xml').getroot()
        root.find('compiler').set('meshdir', str(MODEL_DIR / 'assets'))
        opt = root.find('option')
        opt.set('timestep', str(1 / physics_hz))
        opt.set('integrator', 'implicitfast')
        world = root.find('worldbody')
        ET.SubElement(world, 'geom', name='ground', type='plane', size='0 0 .1',
                      rgba='.38 .42 .47 1', friction='.5 .005 .0001')
        ET.SubElement(world, 'light', pos='0 0 3', dir='0 0 -1', directional='true')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                                   for n in JOINT_NAMES])
        if np.any(self.joint_ids < 0):
            raise ValueError('Go2 模型缺少所需关节')
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = []
        for joint_id in self.joint_ids:
            ids = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if len(ids) != 1:
                raise ValueError('每个关节必须对应一个电机')
            self.actuator_ids.append(int(ids[0]))
        self.actuator_ids = np.asarray(self.actuator_ids)
        if not np.allclose(self.model.actuator_gear[self.actuator_ids, 0], 1):
            raise ValueError('PD 控制要求电机 gear=1')
        self.default_q = np.tile((0., .8, -1.6), 4)
        self.substeps = physics_hz // control_hz
        self.control_dt = 1 / control_hz
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = [0., 0., .4, 1., 0., 0., 0.]
        self.data.qpos[self.qpos_ids] = self.default_q
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        return {'root_position': self.data.qpos[:3].copy(),
                'root_quaternion_wxyz': self.data.qpos[3:7].copy(),
                'joint_position': self.data.qpos[self.qpos_ids].copy(),
                'joint_velocity': self.data.qvel[self.dof_ids].copy()}

    def step(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (12,) or not np.isfinite(target).all():
            raise ValueError('关节目标必须是 12 维有限值')
        for _ in range(self.substeps):
            torque = 25 * (target - self.data.qpos[self.qpos_ids]) - .5 * self.data.qvel[self.dof_ids]
            self.data.ctrl[self.actuator_ids] = np.clip(torque, -23.7, 23.7)
            mujoco.mj_step(self.model, self.data)
        return self.state()
