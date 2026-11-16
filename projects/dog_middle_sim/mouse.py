"""原生 MuJoCo 鼠标扰动：预选机身、开启提示、报告实际输入物理的外力。"""
import mujoco
import numpy as np
import time


class MouseInteraction:
    def __init__(self, viewer, model, data):
        self.viewer,self.model,self.data=viewer,model,data
        self.last_status=None
        self.last_force_log=0.0
        with viewer.lock():
            viewer.perturb.select=model.body('BODY').id
            viewer.perturb.localpos[:]=0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]=1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ]=1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT]=1
        print('鼠标施力：已选中 BODY。先按住 Ctrl，再按右键拖动；'
              '双击其他部位可更换施力对象。拖动时会报告实际外力数值。',flush=True)

    def sync(self):
        # 原生 sync 内部会清除旧扰动并调用 mjv_applyPerturbForce。
        self.viewer.sync()
        with self.viewer.lock():
            selected=int(self.viewer.perturb.select)
            active=int(self.viewer.perturb.active)
        status=(selected,active)
        if status!=self.last_status:
            name=self.model.body(selected).name if selected>0 else '未选中动态刚体'
            force=float(np.linalg.norm(self.data.xfrc_applied[selected,:3])) if selected>0 else 0.0
            print(f'[鼠标] 对象={name} 拖动={bool(active)} 当前力={force:.2f} N',flush=True)
            self.last_status=status
        if selected>0 and active and time.monotonic()-self.last_force_log>0.5:
            vector=self.data.xfrc_applied[selected,:3]
            if np.linalg.norm(vector)>1e-4:
                print(f'[施力] {self.model.body(selected).name} '
                      f'F=({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}) N',flush=True)
                self.last_force_log=time.monotonic()

    def force_norm(self):
        return float(np.linalg.norm(self.data.xfrc_applied[:,:3],axis=-1).max())
