import unittest
import torch
from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel, enforce_executed_action
from go2_learning.tasks.upright_supermimic.state_bridge import next_policy_action


class PaperBufferTests(unittest.TestCase):
    def test_terminal_may_end_window_but_context_never_crosses_reset(self):
        b=PaperBuffer(12,1,2,1,'cpu')
        for i in range(8):
            x=torch.tensor([[float(i),0.]])
            c=torch.full((1,16),0. if i<4 else 1.)
            done=torch.tensor([i==3]); active=torch.tensor([True]);remain=torch.ones(1)
            b.add(x,torch.zeros(1,1),c,x+1,done,done,active,remain)
        state,_,ctx,_,fail,done,_=b.windows(1024,3,torch.tensor([0]))
        starts=state[:,0,0]
        self.assertTrue(torch.all((starts<=1) | (starts>=4)))
        self.assertTrue(torch.equal(ctx[:,0],ctx[:,1]));self.assertTrue(torch.equal(ctx[:,1],ctx[:,2]))
        self.assertTrue(bool((starts==1).any()))  # 允许真实终止位于第 3 步。
        self.assertTrue(bool(fail[starts==1,-1].all()))

    def test_context_changes_prediction_and_backpropagates_action(self):
        torch.manual_seed(0);m=PaperWorldModel(90,12)
        x=torch.zeros(2,90);a=torch.zeros(2,12,requires_grad=True);c=torch.zeros(2,16);c[1,0]=1.
        y,_=m(x,a,c);self.assertFalse(torch.equal(y[0],y[1]))
        y.sum().backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertGreater(float(a.grad.abs().sum()),0.)
        self.assertTrue(torch.equal(y[:,80:88],torch.zeros(2,8)))

    def test_executed_action_not_filtered_twice(self):
        old=torch.zeros(2,90);raw=torch.full((2,12),2.);alpha=.3
        executed=next_policy_action(old[:,33:45],raw,alpha)
        predicted=enforce_executed_action(torch.randn_like(old),executed)
        torch.testing.assert_close(predicted[:,33:45],torch.full((2,12),.7))


if __name__=='__main__':unittest.main()
