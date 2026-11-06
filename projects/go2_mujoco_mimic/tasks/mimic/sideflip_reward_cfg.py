"""左侧空翻独立奖励配置：全身空间跟踪为主，关节姿态跟踪为辅。"""
from .reward_cfg import RewardCfg


def make_sideflip_reward_cfg() -> RewardCfg:
    return RewardCfg(
        # 空间跟踪尺度对齐 SuperMimic 表 A.2。
        body_orientation_sigma=0.4,
        root_orientation_sigma=0.4,
        root_position_sigma=0.3,
        # 每个关节上限 0.1，合计 1.2；全身位置、朝向与速度跟踪合计上限为 5。
        joint_position_weights=(0.1,) * 12,
        joint_sigmas=(0.45, 0.65, 0.65, 0.45, 0.65, 0.65,
                      0.45, 0.65, 0.50, 0.45, 0.65, 0.50),
        torque_l2_weight=1e-4,
        joint_acc_l2_weight=2.5e-7,
        failure_penalty=1.0,
    )
