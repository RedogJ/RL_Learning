"""窗口在环形缓存覆盖后仍按同一环境的真实时间排列。"""
import unittest
import torch
from go2_mujoco_mimic.tasks.mimic.replay import Replay


class ReplayTests(unittest.TestCase):
    def test_wrap_keeps_consecutive_time_and_environment(self):
        replay=Replay(5,3,'cpu')
        for t in range(12):
            values=torch.tensor([[t,env] for env in range(3)],dtype=torch.float32,requires_grad=True)
            replay.add((values,torch.tensor([False,t%2==0,True])))
        torch.manual_seed(12)
        values,done=replay.sample(100,4)
        self.assertTrue((values[...,0]>=7).all())
        torch.testing.assert_close(values[:,1:,0]-values[:,:-1,0],torch.ones(100,3))
        torch.testing.assert_close(values[...,1],values[:,0:1,1].expand(-1,4))
        self.assertFalse(values.requires_grad)
        envs=values[...,1].long();times=values[...,0].long()
        self.assertTrue(torch.equal(done,(envs==2)|((envs==1)&(times%2==0))))

    def test_not_enough_contiguous_steps_is_rejected(self):
        replay=Replay(5,1,'cpu')
        replay.add((torch.zeros(1,2),))
        with self.assertRaises(ValueError):replay.sample(2,3)


if __name__=='__main__':unittest.main()
