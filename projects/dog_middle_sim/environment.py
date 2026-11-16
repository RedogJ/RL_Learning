"""MuJoCo/Warp 真实环境；不包含网络与优化器。"""
from contextlib import contextmanager
import numpy as np
import mujoco
import torch

from .build_model import OUTPUT, JOINTS
from .math_utils import proprio, exp_quat, multiply


class Environment:
    def __init__(self, cfg, backend='warp', device='cuda:0'):
        self.cfg, self.backend, self.device = cfg, backend, torch.device(device)
        if cfg.physics_hz % cfg.control_hz or min(cfg.num_envs,cfg.history,cfg.future) < 1:
            raise ValueError('要求 physics_hz 整除 control_hz，环境数/历史/目标帧数为正')
        self.model = mujoco.MjModel.from_xml_path(str(OUTPUT))
        self.model.opt.timestep = 1/cfg.physics_hz
        self.substeps, self.dt = cfg.physics_hz//cfg.control_hz, 1/cfg.control_hz
        self.qids_np = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINTS],dtype=np.int64)
        self.vids_np = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in JOINTS],dtype=np.int64)
        self.qids = torch.tensor(self.qids_np,device=self.device)
        self.vids = torch.tensor(self.vids_np,device=self.device)
        key = self.model.key('stand').id
        nominal_qpos = self.model.key_qpos[key].copy()
        self.initial = torch.tensor(nominal_qpos,device=self.device,dtype=torch.float32)
        self.reference = torch.cat((self.initial[:7],self.initial[self.qids],self.initial.new_zeros(18)))
        self.default = self.reference[7:19]
        self.limits = torch.tensor(self.model.jnt_range[[self.model.joint(n).id for n in JOINTS]],
                                   device=self.device,dtype=torch.float32)
        self.ctrl_limits = torch.tensor(self.model.actuator_ctrlrange,device=self.device,dtype=torch.float32)
        # 固定仿真上下文；后续增加随机化时必须同步写入逐环境实际参数。
        self.context = torch.tensor([0.8,cfg.kp/40,cfg.kd,20/20],device=self.device)
        self.history = torch.zeros(cfg.num_envs,cfg.history,42,device=self.device)
        self.last_action = torch.zeros(cfg.num_envs,12,device=self.device)
        self.steps = torch.zeros(cfg.num_envs,device=self.device,dtype=torch.long)
        self.graph = None
        if backend == 'warp':
            if self.device.type != 'cuda':
                raise ValueError('Warp 后端要求 CUDA；CPU 调试请用 --backend cpu --device cpu')
            import warp as wp
            import mujoco_warp as mjw
            from .physics import pd_control
            self.wp,self.mjw,self.pd_kernel = wp,mjw,pd_control
            wp.config.quiet=True
            wp.init(); wp.set_device(str(self.device))
            # Graph 不能录在 Torch 的默认 CUDA stream 0 上；持有专用 Warp stream。
            self.warp_stream=wp.get_stream(str(self.device))
            self.torch_stream=torch.cuda.ExternalStream(self.warp_stream.cuda_stream,device=self.device)
            self.consumer_stream=torch.cuda.current_stream(self.device)
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD) | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            cpu = mujoco.MjData(self.model)
            cpu.qpos[:] = nominal_qpos
            self.wmodel = mjw.put_model(self.model)
            self.data = mjw.put_data(self.model,cpu,nworld=cfg.num_envs,njmax=512,nconmax=128)
            self.qpos,self.qvel,self.ctrl = (wp.to_torch(getattr(self.data,k)) for k in ('qpos','qvel','ctrl'))
            self.target = torch.empty(cfg.num_envs,12,device=self.device)
            self.target_wp = wp.from_torch(self.target)
            self.qids_wp,self.vids_wp = wp.from_torch(self.qids),wp.from_torch(self.vids)
            wp.synchronize_device()
            with self.stream():
                self.target[:] = self.default
                self.physics()
                if cfg.cuda_graph:
                    wp.synchronize_device()
                    wp.capture_begin(device=str(self.device))
                    self.physics()
                    self.graph = wp.capture_end(device=str(self.device))
            wp.synchronize_device()
        elif backend == 'cpu':
            self.cpu_data = [mujoco.MjData(self.model) for _ in range(cfg.num_envs)]
        else:
            raise ValueError(backend)
        self.reset(torch.ones(cfg.num_envs,device=self.device,dtype=torch.bool))

    @contextmanager
    def stream(self):
        # 默认 Torch 流与持有的 Warp 流建立双向依赖；数据不回 CPU。
        if self.backend != 'warp':
            yield
            return
        caller=torch.cuda.current_stream(self.device)
        self.consumer_stream=caller
        self.torch_stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self.torch_stream),self.wp.ScopedStream(self.warp_stream):
                yield
        finally:
            caller.wait_stream(self.torch_stream)

    def physics(self):
        for _ in range(self.substeps):
            self.wp.launch(self.pd_kernel,dim=(self.cfg.num_envs,12),
                           inputs=[self.data.qpos,self.data.qvel,self.data.ctrl,self.target_wp,
                                   self.qids_wp,self.vids_wp,self.cfg.kp,self.cfg.kd,20.0])
            self.mjw.step(self.wmodel,self.data)
        self.mjw.forward(self.wmodel,self.data)

    def state(self):
        if self.backend == 'warp':
            state=torch.cat((self.qpos[:,:7],self.qpos[:,self.qids],
                             self.qvel[:,:6],self.qvel[:,self.vids]),-1)
            state.record_stream(self.consumer_stream)
            return state
        qp = np.stack([d.qpos for d in self.cpu_data]); qv = np.stack([d.qvel for d in self.cpu_data])
        return torch.as_tensor(np.concatenate((qp[:,:7],qp[:,self.qids_np],qv[:,:6],qv[:,self.vids_np]),-1),
                               device=self.device,dtype=torch.float32)

    @torch.no_grad()
    def reset(self, mask, randomize=True):
        with self.stream():
            if self.backend=='warp':mask.record_stream(self.torch_stream)
            n = self.cfg.num_envs
            qpos = self.initial.expand(n,-1).clone()
            if randomize:
                qpos[:,self.qids] += (torch.rand(n,12,device=self.device)*2-1)*self.cfg.reset_joint_noise
                rv = (torch.rand(n,3,device=self.device)*2-1)*self.cfg.reset_tilt_noise
                rv[:,2] = 0
                qpos[:,3:7] = multiply(qpos[:,3:7],exp_quat(rv))
                qpos[:,2] += 0.004
            if self.backend == 'warp':
                self.mjw.reset_data(self.wmodel,self.data,self.wp.from_torch(mask.contiguous(),dtype=self.wp.bool))
                self.qpos.copy_(torch.where(mask[:,None],qpos,self.qpos))
                self.qvel.masked_fill_(mask[:,None],0)
                self.mjw.forward(self.wmodel,self.data)
            else:
                for i in mask.nonzero().flatten().cpu().tolist():
                    d = self.cpu_data[i]
                    mujoco.mj_resetData(self.model,d)
                    d.qpos[:] = qpos[i].cpu().numpy()
                    mujoco.mj_forward(self.model,d)
            self.steps.masked_fill_(mask,0)
            self.last_action.masked_fill_(mask[:,None],0)
            obs = proprio(self.state(),self.last_action,self.default)
            self.history.copy_(torch.where(mask[:,None,None],obs[:,None,:],self.history))

    @torch.no_grad()
    def step(self, action):
        with self.stream():
            if self.backend=='warp':action.record_stream(self.torch_stream)
            action = action.detach().clamp(-1,1)
            targets = (self.default+self.cfg.action_scale*action).clamp(self.limits[:,0],self.limits[:,1])
            if self.backend == 'warp':
                self.target.copy_(targets)
                if self.graph is not None:
                    self.wp.capture_launch(self.graph)
                else:
                    self.physics()
            else:
                targets_cpu = targets.cpu().numpy()
                for i,d in enumerate(self.cpu_data):
                    for _ in range(self.substeps):
                        torque=self.cfg.kp*(targets_cpu[i]-d.qpos[self.qids_np])-self.cfg.kd*d.qvel[self.vids_np]
                        d.ctrl[:]=np.clip(torque,-20,20)
                        mujoco.mj_step(self.model,d)
                    mujoco.mj_forward(self.model,d)
            self.steps += 1
            self.last_action.copy_(action)
            state = self.state()
            obs = proprio(state,action,self.default)
            self.history.copy_(torch.cat((self.history[:,1:],obs[:,None,:]),1))
            return state
