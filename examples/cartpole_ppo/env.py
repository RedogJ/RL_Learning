import math

import torch

from scene import read_cartpole_state


class CartpoleEnv:
    def __init__(self, sim, scene, scene_cfg):
        self.sim = sim
        self.scene = scene
        self.scene_cfg = scene_cfg

        # 当前计划：每个控制步推进一个物理步
        self.step_dt = sim.get_physics_dt()

        # 在小车撞到 ±2 米的行程限制之前结束
        self.max_cart_position = 1.8

        # 我们先训练竖直附近的平衡
        self.max_pole_angle = math.radians(30.0)


        self.cart = scene["cart"]
        self.pole = scene["ball"]
        self.pole2 = scene["ball2"]
        self.num_envs = scene_cfg.num_envs
        self.observation_dim = 6

        self.force_limit = 20.0  # N
        self.max_episode_steps = math.ceil(5.0 / self.step_dt)

        self.episode_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=sim.device,
        )

        self.forces = torch.zeros(
            (self.num_envs, 1, 3),
            device=sim.device,
        )

    def get_observation(self):
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = read_cartpole_state(
            self.scene,
            self.scene_cfg,
        )

        return torch.stack(
            (x, x_dot, theta1, theta1_dot, theta2, theta2_dot),
            dim=-1,
        )
    
    def step(self, actions):
        # 输入形状：[环境数量, 1]
        # 网络动作先限制到 [-1, 1]
        applied_actions = actions.detach().clamp(-1.0, 1.0)

        # 归一化动作转换为实际推力
        self.forces[:, 0, 0] = (
            applied_actions[:, 0] * self.force_limit
        )

        self.cart.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=self.forces,
            is_global=True,
        )

        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(self.step_dt)

        self.episode_steps += 1

        # 读取动作执行后的状态
        observation = self.get_observation()
        rewards = self.compute_reward(observation, applied_actions)
        terminated = self.compute_terminated(observation)

        # 没有失败，但达到时间上限
        truncated = (
            self.episode_steps >= self.max_episode_steps
        ) & ~terminated

        return observation, rewards, terminated, truncated, {}

    def compute_reward(self, observation, applied_actions):
        x = observation[:, 0]
        theta1 = observation[:, 2]
        theta2 = observation[:, 4]

        # applied_actions 是限幅后实际使用的归一化动作，
        # 形状为 [环境数量, 1]，不是以 N 为单位的推力。
        action = applied_actions.squeeze(-1)

        reward_rate = (
            1.0
            - theta1.square()
            - theta2.square()
            - 0.1 * x.square()
            - 0.001 * action.square()
        )

        return reward_rate * self.step_dt

    def compute_terminated(self, observation):
        x = observation[:, 0]
        theta1 = observation[:, 2]
        theta2 = observation[:, 4]

        cart_out_of_bounds = x.abs() > self.max_cart_position
        pole_fallen = (theta1.abs() > self.max_pole_angle) | (theta2.abs() > self.max_pole_angle)

        return cart_out_of_bounds | pole_fallen

    def reset(self, env_ids=None):
        # 不指定编号，就重置全部环境
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                device=self.sim.device,
            )

        # 同时恢复小车和两根杆的位置、姿态、速度。
        for asset in (self.cart, self.pole, self.pole2):
            root_pose = asset.data.default_root_pose.torch[env_ids].clone()
            root_pose[:, :3] += self.scene.env_origins[env_ids]

            root_velocity = (
                asset.data.default_root_vel.torch[env_ids].clone()
            )

            asset.write_root_pose_to_sim_index(
                root_pose=root_pose,
                env_ids=env_ids,
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=root_velocity,
                env_ids=env_ids,
            )

        self.scene.reset(env_ids)
        self.forces[env_ids] = 0.0
        self.episode_steps[env_ids] = 0

        return self.get_observation()
