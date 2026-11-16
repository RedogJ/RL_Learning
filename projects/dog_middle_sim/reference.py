"""Middle 原生关节顺序参考数据；SE(2) 循环接续根部，不回到起点。"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from .build_model import OUTPUT,JOINTS
from .retarget import quaternion,rotation,DEFAULT_OUTPUT


class Motion:
    def __init__(self,path=DEFAULT_OUTPUT/'motion.npz'):
        self.path=Path(path)
        metadata_path=self.path.with_name('clip.json')
        self.metadata=json.loads(metadata_path.read_text())
        if self.metadata.get('format')!='middle_motion_v1':raise ValueError('不是 middle_motion_v1 数据')
        if self.metadata['model_sha256']!=hashlib.sha256(OUTPUT.read_bytes()).hexdigest():
            raise ValueError('参考对应的 MJCF 与当前模型不同，请重新映射')
        with np.load(self.path,allow_pickle=False) as data:
            self.qpos=data['qpos'].copy();self.qvel=data['qvel'].copy()
            self.times=data['times'].copy();self.fps=float(data['fps'])
            if data['joint_names'].tolist()!=list(JOINTS):raise ValueError('关节顺序不匹配')
            self.targets=data['target_feet'].copy()
        if (self.qpos.shape!=(len(self.times),19) or self.qvel.shape!=(len(self.times),18)
                or len(self.times)<2 or not np.all(np.diff(self.times)>0)
                or not np.isfinite(self.qpos).all() or not np.isfinite(self.qvel).all()):
            raise ValueError('参考形状/时间/有限性检查失败')
        np.testing.assert_allclose(np.linalg.norm(self.qpos[:,3:7],axis=-1),1,atol=1e-5)
        self.duration=float(self.times[-1])
        self.slerp=Slerp(self.times,rotation(self.qpos[:,3:7]))
        self.delta_yaw=float(self.metadata['cycle_delta_yaw_rad'])
        self.delta_R=Rotation.from_euler('z',self.delta_yaw)
        self.delta_translation=self.qpos[-1,:3]-self.delta_R.apply(self.qpos[0,:3])
        self.translations=[np.zeros(3)]

    def cycle_transform(self,cycle):
        while len(self.translations)<=cycle:
            self.translations.append(self.delta_R.apply(self.translations[-1])+self.delta_translation)
        return Rotation.from_euler('z',cycle*self.delta_yaw),self.translations[cycle]

    def sample(self,time,loop=True):
        if time<0:raise ValueError('播放时间不能为负')
        cycle=int(time/self.duration) if loop else 0
        phase=time-cycle*self.duration if loop else min(time,self.duration)
        phase=float(np.clip(phase,0,self.duration))
        index=min(int(np.searchsorted(self.times,phase,side='right')-1),len(self.times)-2)
        alpha=(phase-self.times[index])/(self.times[index+1]-self.times[index])
        qp=(1-alpha)*self.qpos[index]+alpha*self.qpos[index+1]
        qp[3:7]=quaternion(self.slerp([phase])[0])
        qv=(1-alpha)*self.qvel[index]+alpha*self.qvel[index+1]
        target=(1-alpha)*self.targets[index]+alpha*self.targets[index+1]
        turn,translation=self.cycle_transform(cycle)
        qp[:3]=turn.apply(qp[:3])+translation
        qp[3:7]=quaternion(turn*rotation(qp[3:7]))
        qv[:3]=turn.apply(qv[:3])
        # MuJoCo自由关节角速度是机身坐标分量，累计世界yaw不改变该分量。
        return qp,qv,turn.apply(target)+translation
