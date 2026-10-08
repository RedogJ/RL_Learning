# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

def compute_pd_force(theta, theta_dot, kp, kd):
    """根据摆角 [rad] 和角速度 [rad/s]，计算小车推力 [N]。"""
    return kp * theta + kd * theta_dot