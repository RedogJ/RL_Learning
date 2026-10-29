"""旧模型奖励配置的读取兼容；日常调参只修改 reward_cfg.py。"""

from dataclasses import dataclass

from .reward_cfg import RewardCfg

CURRENT_REWARD_SCHEMA = 16


@dataclass
class GaussianFootSlidePlaybackCfg(RewardCfg):
    """schema 15 及更早模型保留当时的触地脚掌滑动指数收益。"""

    reward_schema_version: int = 15
    foot_slide_sigma: float = 1.0


@dataclass
class GaussianAngularPlaybackCfg(GaussianFootSlidePlaybackCfg):
    """schema 14 及更早模型保留当时的机身三轴角速度指数收益。"""

    reward_schema_version: int = 14
    angular_velocity_x_sigma: float = 1.0
    angular_velocity_y_sigma: float = 1.0
    angular_velocity_z_sigma: float = 1.0


@dataclass
class GaussianLinearPlaybackCfg(GaussianAngularPlaybackCfg):
    """schema 13 及更早模型保留当时的世界三轴线速度指数收益。"""

    reward_schema_version: int = 13
    linear_velocity_x_sigma: float = 0.3
    linear_velocity_y_sigma: float = 0.3
    linear_velocity_z_sigma: float = 0.3


@dataclass
class GaussianFrontPlaybackCfg(GaussianLinearPlaybackCfg):
    """schema 12 及更早模型保留当时的前腿目标指数收益。"""

    reward_schema_version: int = 12
    front_target_sigma: float = 0.5


@dataclass
class GaussianHipPlaybackCfg(GaussianFrontPlaybackCfg):
    """schema 11 及更早模型使用当时的后髋指数收益。"""

    reward_schema_version: int = 11
    hip_rear_sigma: float = 0.5


@dataclass
class SquaredMotionPlaybackCfg(GaussianHipPlaybackCfg):
    """保留回滚前 schema 10 模型的六轴速度平方惩罚口径。"""

    reward_schema_version: int = 10


@dataclass
class GaussianMotionPlaybackCfg(GaussianHipPlaybackCfg):
    """schema 9 回放保留六轴速度指数收益及关节速度平方惩罚。"""

    reward_schema_version: int = 9
    linear_velocity_x_sigma: float = 0.3
    linear_velocity_y_sigma: float = 0.3
    linear_velocity_z_sigma: float = 0.3
    angular_velocity_x_sigma: float = 1.0
    angular_velocity_y_sigma: float = 1.0
    angular_velocity_z_sigma: float = 1.0


@dataclass
class GaussianJointVelocityPlaybackCfg(GaussianMotionPlaybackCfg):
    """schema 8 回放保留当时的关节速度指数收益。"""

    reward_schema_version: int = 8
    joint_velocity_sigma: float = 5.0


@dataclass
class PlaybackRewardCfg(GaussianMotionPlaybackCfg):
    balance_weight: float = 60.0
    angular_velocity_weight: float = 40.0
    pitch_velocity_weight: float = 30.0
    reward_schema_version: int = 5
    horizontal_velocity_sigma: float = 0.4
    hip_rear_sigma: float = 0.5
    rear_symmetry_sigma: float = 0.3

    def __post_init__(self):
        super().__post_init__()
        if type(self.reward_schema_version) is not int or self.reward_schema_version not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError("不支持此历史奖励版本")
        if self.horizontal_velocity_sigma <= 0:
            raise ValueError("horizontal_velocity_sigma 必须为正数")
        if self.hip_rear_sigma <= 0 or self.rear_symmetry_sigma <= 0:
            raise ValueError("历史后髋与对称奖励 sigma 必须为正数")


def load_reward_config(saved):
    reward_cfg = dict(saved)
    # 历史记录可读取，但移除的低通/吸收代价不再参与环境计算。
    reward_cfg.pop("failure_absorbing_weight", None)
    reward_cfg.pop("failure_absorbing_alpha", None)
    # 此前保存的模型没有原始输出变化项；回放不默认启用新奖励。
    reward_cfg.setdefault("action_rate_weight", 0.0)
    reward_cfg.setdefault("action_rate_derivative_weight", 0.0)
    if reward_cfg.get("reward_schema_version") == 15:
        return GaussianFootSlidePlaybackCfg(**{"rear_posture_weight": 0.0, **reward_cfg})
    if reward_cfg.get("reward_schema_version") == 14:
        return GaussianAngularPlaybackCfg(**{"rear_posture_weight": 0.0, **reward_cfg})
    if reward_cfg.get("reward_schema_version") == 13:
        return GaussianLinearPlaybackCfg(**{"rear_posture_weight": 0.0, **reward_cfg})
    if reward_cfg.get("reward_schema_version") == 12:
        return GaussianFrontPlaybackCfg(**{"rear_posture_weight": 0.0, **reward_cfg})
    if reward_cfg.get("reward_schema_version") == 11:
        return GaussianHipPlaybackCfg(**{"rear_posture_weight": 0.0, **reward_cfg})
    if reward_cfg.get("reward_schema_version") == 10:
        return SquaredMotionPlaybackCfg(**{"rear_posture_weight": 0.0, **reward_cfg})
    if reward_cfg.get("reward_schema_version") == 8:
        return GaussianJointVelocityPlaybackCfg(**{"rear_posture_weight": 0.0, **reward_cfg})
    if reward_cfg.get("reward_schema_version") == 9:
        return GaussianMotionPlaybackCfg(**{"rear_posture_weight": 0.0, **reward_cfg})
    if reward_cfg.get("reward_schema_version") == CURRENT_REWARD_SCHEMA:
        reward_cfg.pop("reward_schema_version")
        # 历史模型没有该辅助项时保持关闭；新训练和续训使用当前默认配置。
        return RewardCfg(**{
            "rear_posture_weight": 0.0, **reward_cfg,
        })
    # 已移除的奖励项；旧配置仍可加载，原始历史文件不重写。
    for name in ("upright_progress_weight", "hip_weight", "hip_sigma",
                 "thigh_rear_weight", "thigh_rear_target", "thigh_rear_sigma",
                 "hip_front_weight", "hip_front_sigma",
                 "hip_knee_lift_weight", "hip_knee_lift_sigma",
                 "rear_airborne_weight", "rear_both_airborne_weight",
                 "rear_knee_and_foot_support_weight",
                 "front_pose_weight", "front_pose_target_x", "front_pose_target_y",
                 "front_pose_target_z", "front_pose_sigma",
                 "front_foot_position_weight", "front_foot_target_left", "front_foot_target_right",
                 "vertical_velocity_weight", "action_rate_weight"):
        reward_cfg.pop(name, None)
    config = PlaybackRewardCfg(**{
        # 无版本字段的旧记录沿用原来的角速度/接触叠加公式。
        "reward_schema_version": 1,
        "hip_contact_weight": {1: 60.0, 2: 120.0}.get(
            reward_cfg.get("reward_schema_version", 1), RewardCfg.hip_contact_weight),
        # 更早的模型没有这些奖励时按 0 加载；续训使用当前配置。
        "pitch_velocity_weight": 0.0,
        "rear_knee_contact_weight": 0.0,
        "rear_knee_clearance_weight": 0.0,
        # 旧动作变化权重不能按原数值解释为关节速度权重；旧模型缺少新项时关闭。
        "joint_velocity_weight": 0.0,
        "front_target_weight": 0.0,
        "rear_feet_airborne_weight": 0.0,
        "rear_posture_weight": 0.0,
        **reward_cfg,
    })
    return config
