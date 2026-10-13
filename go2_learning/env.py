"""RSL-RL VecEnv：45 维观测、12 维关节位置动作，简单站立任务。"""

import math

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


def make_env(sim, num_envs=1):
    from isaaclab.scene import InteractiveScene

    from .scene import Go2SceneCfg, USD_PATH, GROUND_PATH

    if not USD_PATH.is_file():
        raise FileNotFoundError(f"找不到 Go2 模型：{USD_PATH}")
    if not GROUND_PATH.is_file():
        raise FileNotFoundError(f"找不到地面模型：{GROUND_PATH}")
    scene = InteractiveScene(Go2SceneCfg(num_envs=num_envs, env_spacing=3.0, replicate_physics=False))
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.3))
    sim.reset()
    return Go2Env(sim, scene)


class Go2Env(VecEnv):
    def __init__(self, sim, scene):
        self.sim, self.scene = sim, scene
        self.robot = scene["robot"]
        self.device = sim.device
        self.num_envs = scene.num_envs
        self.num_actions = self.robot.num_joints
        if self.num_actions != 12:
            raise ValueError(f"预期 Go2 有 12 个关节，实际为 {self.robot.joint_names}")
        self.cfg = {}
        self.physics_dt = sim.get_physics_dt()
        self.decimation = 4  # 物理 120 Hz，策略/环境 30 Hz。
        self.dt = self.physics_dt * self.decimation
        self.action_scale = 0.25  # [-1,1] 对应站姿附近 ±0.25 rad。
        self.max_episode_length = math.ceil(10.0 / self.dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        self.mouse_push = None
        self.reset()

    def get_observations(self):
        data = self.robot.data
        obs = torch.cat((
            data.root_lin_vel_b.torch,                       # 3：机身线速度
            data.root_ang_vel_b.torch,                       # 3：机身角速度
            data.projected_gravity_b.torch,                  # 3：机身坐标下的重力方向
            data.joint_pos.torch - self.default_joint_pos,   # 12：关节偏移
            0.1 * data.joint_vel.torch,                      # 12：缩放后的关节速度
            self.actions,                                   # 12：上一动作
        ), dim=-1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def step(self, actions, auto_reset=True):
        self.actions.copy_(actions.detach().clamp(-1.0, 1.0))
        target = self.default_joint_pos + self.action_scale * self.actions
        self.robot.actuators.target_command.set_position_index(value=target)
        for _ in range(self.decimation):
            if self.mouse_push is not None:
                self.mouse_push.apply()
            self.scene.write_data_to_sim()
            self.sim.step(render=True)
            self.scene.update(self.physics_dt)
        self.episode_length_buf += 1

        data = self.robot.data
        height = data.root_link_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        gravity = data.projected_gravity_b.torch
        tilt = gravity[:, :2].square().sum(dim=-1)
        speed = data.root_lin_vel_b.torch.square().sum(dim=-1)
        posture = (data.joint_pos.torch - self.default_joint_pos).square().sum(dim=-1)
        # 只练习站稳：保持高度和直立，抑制移动、偏离站姿和过大的动作。
        reward = (1.0 - 4.0 * tilt - 10.0 * (height - 0.32).square()
                  - speed - 0.1 * posture - 0.01 * self.actions.square().sum(dim=-1)) * self.dt
        failed = (height < 0.18) | (gravity[:, 2] > -0.5)
        timed_out = (self.episode_length_buf >= self.max_episode_length) & ~failed
        dones = failed | timed_out
        reward = reward - failed.float()
        extras = {
            "time_outs": timed_out,
            "log": {"Task/failures_per_step": failed.float().sum(), "Task/height": height.mean()},
        }
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if auto_reset and done_ids.numel():
            self.reset(done_ids)
        return self.get_observations(), reward, dones.long(), extras

    def reset(self, env_ids=None):
        if self.mouse_push is not None:
            self.mouse_push.clear()
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids,
        )
        self.robot.write_joint_position_to_sim_index(position=self.default_joint_pos[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.robot.data.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids,
        )
        self.scene.reset(env_ids)
        self.robot.actuators.target_command.set_position_index(value=self.default_joint_pos[env_ids], env_ids=env_ids)
        self.actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
