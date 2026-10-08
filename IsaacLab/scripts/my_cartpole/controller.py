def compute_pd_force(theta, theta_dot, kp, kd):
    """根据摆角 [rad] 和角速度 [rad/s]，计算小车推力 [N]。"""
    return kp * theta + kd * theta_dot