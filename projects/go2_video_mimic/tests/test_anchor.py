import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from go2_mujoco_mimic.tasks.mimic.kinematics import rotate
from go2_mujoco_mimic.tasks.mimic.rewards import quat_mul
from go2_video_mimic.anchor import features, canonical_goal, yaw_quat
from go2_video_mimic.config import reward_config
from go2_video_mimic.task import AnchorTask
from go2_video_mimic.prepare import convert
from go2_mujoco_mimic.tasks.mimic.config import MimicTaskCfg
from go2_video_mimic.config import SLOW_MOTION


def pose():
    torch.manual_seed(42)
    p = torch.randn(2, 17, 3)*.1
    p[..., 2] += .3
    p[:, 0, :2] = torch.tensor([[2., -3.], [-1., 4.]])
    r = Rotation.from_euler('xyz', [[.1, -.2, .4], [-.1, .1, -.6]]).as_quat()[:, [3, 0, 1, 2]]
    q = torch.tensor(r, dtype=torch.float32)[:, None].expand(-1, 17, -1)
    return dict(pos=p, quat=q, linear_vel=torch.randn_like(p), angular_vel=torch.randn_like(p))


def gauge(x):
    angle = torch.tensor([.7, -1.2])/2
    z = torch.zeros_like(angle)
    q = torch.stack((angle.cos(), z, z, angle.sin()), -1)[:, None].expand_as(x['quat'])
    rp = rotate(q, x['pos'])
    shift = torch.tensor([[13., -7., 0.], [-9., 5., 0.]])[:, None]
    shift_v = torch.tensor([[3., -2., 0.], [-4., 1., 0.]])[:, None]
    w = torch.tensor([[0., 0., 1.1], [0., 0., -.8]])[:, None].expand_as(rp)
    return dict(pos=rp+shift, quat=quat_mul(q, x['quat']),
                linear_vel=rotate(q, x['linear_vel'])+torch.cross(w, rp, dim=-1)+shift_v,
                angular_vel=rotate(q, x['angular_vel'])+w)


class AnchorTests(unittest.TestCase):
    def test_time_varying_translation_and_yaw_cancel_including_velocity(self):
        a, b = features(pose()), features(gauge(pose()))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=5e-5, rtol=5e-5)

    def test_height_and_tilt_remain_observable(self):
        a = pose(); b = {k: v.clone() for k, v in a.items()}
        b['pos'][..., 2] += .15
        torch.testing.assert_close(features(b)['pos'][..., 2]-features(a)['pos'][..., 2],
                                   torch.full((2, 17), .15))
        self.assertGreater(float(features(a)['quat'][..., 1:3].abs().max()), .01)

    def test_reward_ignores_reference_camera_motion_but_has_pose_gradient(self):
        task = AnchorTask(MimicTaskCfg(reward=reward_config()))
        actual, ref = pose(), pose()
        actual['pos'] = (actual['pos']+.01).requires_grad_()
        ref['joint_pos'] = torch.zeros(2, 12)
        moved = dict(gauge(ref), joint_pos=ref['joint_pos'])
        zeros = torch.zeros(2, 12)
        limits = torch.tensor([[-3., 3.]]).expand(12, -1)
        kw = dict(joint_vel=zeros, prev_joint_vel=zeros, target=zeros,
                  torque_squared_mean=zeros)
        a, _ = task.reward(actual, ref, zeros, zeros, zeros, limits, .02, **kw)
        b, _ = task.reward(actual, moved, zeros, zeros, zeros, limits, .02, **kw)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        a.sum().backward()
        self.assertTrue(torch.isfinite(actual['pos'].grad).all())
        self.assertGreater(float(actual['pos'].grad.abs().sum()), 0.)

    def test_policy_goal_excludes_xy_and_absolute_yaw(self):
        a = dict(pos=torch.tensor([[2., -3., .3]]), quat=torch.tensor([[1., 0., 0., 0.]]),
                 joints=torch.zeros(1, 12))
        b = dict(a, pos=torch.tensor([[200., 90., .3]]),
                 quat=torch.tensor([[.70710678, 0., 0., .70710678]]))
        torch.testing.assert_close(canonical_goal(a), canonical_goal(b), atol=1e-6, rtol=1e-6)

    def test_prepare_joint_order_height_and_noncyclic_endpoints(self):
        q = np.arange(48).reshape(4, 12)/50
        source = dict(robot_type='unitree_go2', dof_pos=q.copy(), fps=50.,
                      root_pos=np.array([[1, 2, .3], [2, 4, .32], [3, 5, .31], [4, 6, .33]]),
                      root_rot=Rotation.from_euler('z', np.array([0., .2, .4, .6])[:, None]).as_quat())
        out = convert(source)
        decoded = out[:, 6:18].reshape(-1, 3, 4).transpose(0, 2, 1)
        np.testing.assert_allclose(decoded, q.reshape(4, 4, 3)[:, [1, 0, 3, 2]], atol=1e-7)
        np.testing.assert_allclose(out[:, 48:50], 0.)
        np.testing.assert_allclose(out[:, 50], source['root_pos'][:, 2])
        np.testing.assert_allclose(out[:, 51:55], np.tile([1, 0, 0, 0], (4, 1)), atol=1e-7)
        self.assertFalse(np.allclose(out[0, 6:18], out[-1, 6:18]))
        np.testing.assert_array_equal(source['dof_pos'], q)

    def test_prepared_reference_matches_physical_go2_and_action_coverage(self):
        import mujoco
        from go2_mujoco_mimic.environment import Go2Environment
        from go2_mujoco_mimic.tasks.mimic.kinematics import BodyKinematics
        cpu = Go2Environment(600, 50)
        frames = np.load(SLOW_MOTION)
        joints = frames[:, 6:18].reshape(-1, 3, 4).transpose(0, 2, 1).reshape(-1, 12)
        limits = cpu.model.jnt_range[cpu.joint_ids]
        self.assertTrue((joints >= limits[:, 0]-1e-5).all())
        self.assertTrue((joints <= limits[:, 1]+1e-5).all())
        scale = np.maximum(limits[:, 1]-cpu.default_q, cpu.default_q-limits[:, 0])
        self.assertLessEqual(float(np.abs((joints-cpu.default_q)/scale).max()), 1.00001)
        names = ['base_link']+[f'{leg}_{part}' for leg in ('FL', 'FR', 'RL', 'RR')
                               for part in ('hip', 'thigh', 'calf', 'foot')]
        ids = [cpu.model.body(n).id for n in names]
        fk = BodyKinematics(cpu.model, ids, cpu.joint_ids)
        for i in (0, len(frames)//2, len(frames)-1):
            cpu.data.qpos[:3] = frames[i, 48:51]; cpu.data.qpos[3:7] = frames[i, 51:55]
            cpu.data.qpos[cpu.qpos_ids] = joints[i]
            mujoco.mj_forward(cpu.model, cpu.data)
            state = torch.tensor(np.r_[frames[i, 48:55], joints[i], np.zeros(18)], dtype=torch.float32)[None]
            positions, _ = fk(state)
            np.testing.assert_allclose(positions.detach().numpy()[0], cpu.data.xpos[ids], atol=1e-6)

    def test_retiming_scales_velocity_and_acceleration_on_same_physical_clock(self):
        t = np.arange(21)/50.
        # Polynomial motion has exact known derivatives on a centered stencil.
        q = np.tile((t*t)[:, None], (1, 12))
        src = dict(robot_type='unitree_go2', fps=50., dof_pos=q,
                   root_pos=np.column_stack((t*0, t*0, .3+t*t)),
                   root_rot=np.tile([0., 0., 0., 1.], (len(t), 1)))
        a, b = convert(src), convert(src, speed=2.)
        self.assertEqual(len(b), 11)
        va = np.gradient(a[:, 6], 1/50.)
        vb = np.gradient(b[:, 6], 1/50.)
        aa = np.gradient(va, 1/50.)
        ab = np.gradient(vb, 1/50.)
        np.testing.assert_allclose(vb[2:-2], 2*va[4:-4:2], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ab[2:-2], 4*aa[4:-4:2], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(b[:, 50], .3+4*t[:11]**2, atol=1e-7)
        for speed in (0., -1., float('nan')):
            with self.assertRaises(ValueError): convert(src, speed=speed)


if __name__ == '__main__':
    unittest.main()
