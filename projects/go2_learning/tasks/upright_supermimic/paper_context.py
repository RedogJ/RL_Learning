"""论文附录 A.3 的逐回合物理随机化，隔离在 SuperMimic 实验中。"""
import torch
import warp as wp

CONTEXT_DIM = 16  # 一个摩擦系数、12 个电机关节标定偏差、机身质心偏移 XYZ。


def configure_paper_physics(task):
    """论文每次控制 1/50 秒，执行四个 1/200 秒的物理子步。"""
    task.cfg.environment.physics_dt=1.0/200.0
    task.cfg.environment.decimation=4


def configure_ground(task):
    """地面摩擦 1，与每个机器人的材料摩擦相乘，得到采样系数。"""
    original = task.configure_scene

    def configure(scene_cfg):
        original(scene_cfg)
        material = scene_cfg.ground.spawn.physics_material
        material.static_friction = 1.0
        material.dynamic_friction = 1.0
        material.friction_combine_mode = 'multiply'

    task.configure_scene = configure


class PaperContext:
    """真实写入仿真，保存采样值；关节标定模拟执行器位置零点偏移。"""
    def __init__(self, env, seed):
        self.env = env
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.value = torch.zeros((env.num_envs, CONTEXT_DIM), device=env.device)
        self.base_id = env.robot.body_names.index('base')
        # 保存未加偏移的机身 COM；重复 reset 不能累计偏移。
        self.default_com = env.robot.data.body_com_pose_b.torch[:, self.base_id:self.base_id+1].clone()
        self._original_reset = env.reset
        self._original_set_action = env._set_action
        env.reset = self.reset
        env._set_action = self.set_action
        # 保留初始关节扰动，但不调度随机外力。
        if env.randomization is not None:
            env.randomization.apply = lambda active=None: None
        self.reset(torch.arange(env.num_envs, device=env.device))

    def _uniform(self, n, low, high):
        return low + (high-low)*torch.rand((n,), generator=self.generator)

    def reset(self, env_ids=None):
        env = self.env
        ids = torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
        if not ids.numel():
            return self._original_reset(ids)
        n=ids.numel(); cpu_ids=ids.to('cpu',dtype=torch.int32).contiguous()
        mu=self._uniform(n,.3,1.2)
        calibration=torch.stack([self._uniform(n,-.01,.01) for _ in range(12)],-1)
        com=torch.stack((self._uniform(n,-.025,.025), self._uniform(n,-.05,.05), self._uniform(n,-.05,.05)),-1)
        self.value[ids]=torch.cat((mu[:,None],calibration,com),-1).to(env.device)
        materials=wp.to_torch(env.robot.root_view.get_material_properties()).clone()
        materials[cpu_ids.long(),:,0]=mu[:,None]
        materials[cpu_ids.long(),:,1]=mu[:,None]
        env.robot.root_view.set_material_properties(wp.from_torch(materials.contiguous(),dtype=wp.float32),
                                                    wp.from_torch(cpu_ids,dtype=wp.int32))
        target=self.default_com[ids].clone();target[:,0,:3]+=com.to(env.device)
        env.robot.set_coms_index(coms=wp.from_torch(target.contiguous(),dtype=wp.transformf),
                                 body_ids=[self.base_id],env_ids=ids)
        return self._original_reset(ids)

    def set_action(self, actions):
        self._original_set_action(actions)
        target=self.env.action_center+self.env.action_half_range*self.env.actions+self.value[:,1:13]
        self.env.robot.actuators.target_command.set_position_index(value=target)

    def verify(self):
        """从 PhysX 回读实际摩擦及 COM，不以记录值代替仿真属性。"""
        env=self.env
        props=wp.to_torch(env.robot.root_view.get_material_properties())
        mu=self.value[:,0].cpu()
        friction_error=max(float((props[:,:,0]-mu[:,None]).abs().max()),
                           float((props[:,:,1]-mu[:,None]).abs().max()))
        com=env.robot.data.body_com_pose_b.torch[:,self.base_id,:3]
        expected=self.default_com[:,0,:3]+self.value[:,13:16]
        com_error=float((com-expected).abs().max())
        return {'friction_max_abs_error':friction_error,'com_max_abs_error_m':com_error,
                'friction_range': [float(mu.min()),float(mu.max())]}
