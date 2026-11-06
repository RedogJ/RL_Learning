"""后空翻奖励参数；其他模仿任务继续使用各自默认配置。"""
from .reward_cfg import RewardCfg


def make_backflip_reward_cfg() -> RewardCfg:
    return RewardCfg(
        # 落地朝向误差约 40 度；先扩大有效奖励范围。
        body_orientation_sigma=0.7,
        root_orientation_sigma=0.7,
        root_position_sigma=0.2,  # m；收紧机身位置跟踪，4750 落地平均偏差约 0.115 m
        # 每个关节单独给分：sum_i weight_i * exp(-(角误差_i / sigma_i)^2)。
        # 每行 hip/thigh/calf；四行 FL/FR/RL/RR。权重可单独调，当前总上限 12。
        joint_position_weights=(
            1.0, 1.0, 1.0,  # FL 左前
            1.0, 1.0, 1.0,  # FR 右前
            1.0, 1.0, 1.0,  # RL 左后
            1.0, 1.0, 1.0,  # RR 右后
        ),
        joint_sigmas=(
            0.45, 0.65, 0.65,  # FL；rad
            0.45, 0.65, 0.65,  # FR；左右一致
            0.45, 0.65, 0.50,  # RL；4750 后大腿落地 RMS 约 22.4 度，继续收紧
            0.45, 0.65, 0.50,  # RR；与左后相同
        ),  # 固定尺度，不随训练自动改变
        torque_l2_weight=1e-4,  # 力矩平方和惩罚；v15 的 5e-4 疑似让落地大腿不愿出力，降为 1e-4
        joint_acc_l2_weight=2.5e-7,  # 关节加速度平方和惩罚；沿用 go2-supermimic 的权重，腾空段约占满分 3%
        failure_penalty=1.0,  # 触发失败时一次性扣 1
    )
