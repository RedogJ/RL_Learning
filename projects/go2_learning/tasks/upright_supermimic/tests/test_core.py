import unittest
import torch
from go2_learning.tasks.upright_supermimic.sequence_buffer import SequenceBuffer
from go2_learning.tasks.upright_supermimic.state_bridge import enforce_action_state, enforce_context

class CoreTests(unittest.TestCase):
    def test_window_does_not_cross_done_or_wait(self):
        b=SequenceBuffer(8,2,1,1,'cpu')
        for t in range(8):
            obs=torch.tensor([[float(t)],[100.+t]]);v=torch.zeros(2);done=torch.tensor([t==3,False]);active=torch.tensor([True,t!=5])
            b.add(obs,obs,obs+1,v,done,done,active,v,v)
        o,_,_=b.windows(512,3,torch.tensor([0,1]))
        starts=o[:,0,0]
        # 环境 0 禁止跨越 step 3；环境 1 禁止跨越等待 step 5。
        self.assertTrue(torch.all((starts[starts < 100] < 1) | (starts[starts < 100] > 3)))
        other=starts[starts >= 100]-100
        self.assertTrue(torch.all((other < 3) | (other > 5)))
        self.assertTrue(torch.all(o[:,1,0] == starts+1))

    def test_action_filter_state(self):
        old=torch.zeros(1,90);new=torch.randn(1,90);action=torch.full((1,12),2.)
        x=enforce_action_state(new,old,action,.3)
        torch.testing.assert_close(x[:,33:45],torch.full((1,12),.7))
        self.assertTrue(torch.equal(x[:,88:],new[:,88:]))

    def test_observed_context_and_unknown_future(self):
        old=torch.zeros(1,90);new=torch.randn(1,90);old[:,80]=1.;old[:,81]=.1;old[:,82:85]=10.
        measured=torch.ones(1,90)
        self.assertTrue(torch.equal(enforce_context(new,old,.05,measured)[:,80:88],measured[:,80:88]))
        x=enforce_context(new,old,.05)
        self.assertAlmostEqual(x[0,81].item(),.05,places=6)
        self.assertAlmostEqual(x[0,82].item(),10.,places=6)

if __name__=='__main__':unittest.main()
