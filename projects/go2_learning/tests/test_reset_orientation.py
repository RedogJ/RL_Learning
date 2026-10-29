"""竖直圆锥分布、支撑点几何和每回合策略接管时序。"""
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
import torch
from go2_learning.control.reset_orientation import ResetOrientationSampler,multiply,rotate
from go2_learning.tasks import get_task
from go2_learning.tasks.upright.reset_cfg import ResetCfg
from go2_learning.tests.test_reset_settling import load_env_class,Scene,Sim


class OrientationResetTests(unittest.TestCase):
    def sampler(self,n=4):
        q=torch.tensor([0.,-math.sqrt(.5),0.,math.sqrt(.5)]).repeat(n,1)
        pose=torch.cat((torch.tensor([0.,0.,.45]).repeat(n,1),q),-1)
        feet=torch.tensor([[[0.,-.12,.0223],[0.,.12,.0223]]]).repeat(n,1,1)
        data=NS(root_link_pos_w=NS(torch=pose[:,:3]),root_link_quat_w=NS(torch=pose[:,3:]),body_link_pos_w=NS(torch=feet))
        robot=NS(data=data,body_names=['RL_foot','RR_foot'])
        robot.write_root_pose_to_sim_index=lambda root_pose,env_ids:pose.__setitem__(env_ids,root_pose)
        robot.write_root_velocity_to_sim_index=Mock();robot.write_joint_velocity_to_sim_index=Mock()
        env=NS(device='cpu',robot=robot,num_envs=n,num_actions=12,reset_root_pose=pose.clone())
        return ResetOrientationSampler(env,10.,43),env,pose,feet

    def test_distribution_is_bounded_reproducible_and_global_rng_independent(self):
        a,_,_,_=self.sampler();b,_,_,_=self.sampler()
        q=torch.tensor([0.,-math.sqrt(.5),0.,math.sqrt(.5)]).repeat(8192,1)
        rng=torch.get_rng_state().clone();qa,tilt=a.sample(q);qb,_=b.sample(q)
        torch.testing.assert_close(qa,qb,atol=0,rtol=0)
        torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
        self.assertTrue(((tilt>=0)&(tilt<=10.001)).all())
        torch.testing.assert_close(qa.norm(dim=-1),torch.ones(8192),atol=2e-6,rtol=0)
        forward=rotate(qa,torch.tensor([1.,0.,0.]).expand(8192,-1))
        self.assertAlmostEqual(float(forward[:,2].mean()),(1+math.cos(math.radians(10)))/2,delta=.0003)
        self.assertLess(float(forward[:,:2].mean(0).abs().max()),.005)
        self.assertFalse(torch.equal(a.sample(q)[0],qa))

    def test_partial_rotation_keeps_lowest_rear_foot_height(self):
        sampler,env,pose,feet=self.sampler();old=pose.clone();ids=torch.tensor([1,3])
        sampler.apply(ids)
        torch.testing.assert_close(pose[[0,2]],old[[0,2]],atol=0,rtol=0)
        conjugate=old[ids,3:].clone();conjugate[:,:3].neg_()
        delta=multiply(pose[ids,3:],conjugate)
        new_feet=pose[ids,None,:3]+rotate(delta[:,None].expand(-1,2,-1),feet[ids]-old[ids,None,:3])
        torch.testing.assert_close(new_feet[:,:,2].min(-1).values,feet[ids,:,2].min(-1).values,atol=1e-6,rtol=0)
        torch.testing.assert_close(new_feet[:,:,:2].mean(1),feet[ids,:,:2].mean(1),atol=1e-6,rtol=0)
        self.assertTrue((env.robot.write_root_velocity_to_sim_index.call_args.kwargs['root_velocity']==0).all())
        self.assertTrue((env.robot.write_joint_velocity_to_sim_index.call_args.kwargs['velocity']==0).all())

    def test_apply_once_at_handoff_per_episode(self):
        task=get_task('upright');task.cfg.reset.orientation_cone_degrees=0.
        scene=Scene();env=load_env_class()(Sim(scene),scene,task=task)
        sampler=Mock();env.reset_orientation_sampler=sampler
        env.finish_initial_settling()
        self.assertEqual(sampler.apply.call_count,1)
        torch.testing.assert_close(sampler.apply.call_args.args[0],torch.arange(3))
        env.step(env.standing_actions,auto_reset=False)
        self.assertEqual(sampler.apply.call_count,1)
        env.reset(torch.tensor([1]));env.finish_initial_settling()
        self.assertEqual(sampler.apply.call_count,2)
        torch.testing.assert_close(sampler.apply.call_args.args[0],torch.tensor([1]))

    def test_old_playback_keeps_fixed_initial_pose_and_new_config_persists(self):
        task=get_task('upright');saved=task.run_config(False)
        self.assertEqual(saved['reset_cfg']['orientation_cone_degrees'],10.)
        del saved['reset_cfg']['orientation_cone_degrees']
        restored=get_task('upright');restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.reset.orientation_cone_degrees,0.)
        for bad in (-1,90,float('nan')):
            with self.assertRaises(ValueError):ResetCfg(orientation_cone_degrees=bad)
