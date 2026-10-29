"""当前指令条件：禁止未来命令泄漏，并验证采样确实读取指令。"""
import unittest
import torch
from go2_learning.tasks.mimic.config import ModelCfg
from go2_learning.tasks.mimic.models import TrajectoryDiffusion
from go2_learning.tasks.mimic.data import Windows


class CommandTests(unittest.TestCase):
    def cfg(self):
        return ModelCfg(command_dim=6, history=2, horizon=3, embedding=16, heads=2, layers=1, noise_steps=3)

    def test_window_alignment_and_command_change(self):
        cfg=self.cfg()
        intent=torch.tensor([[1.,0.,1.5,0.,0.,.3]]).repeat(15,1)
        intent[6:,2]=0.
        row=dict(state=torch.randn(15,34),latent=torch.randn(15,16),intent=intent,expert='running')
        windows=Windows([row],cfg)
        # current=6 刚收到停车指令，过去仍在前进：必须保留。
        self.assertIn((0,4),windows.indices)
        # current=5，未来将收到未知停车指令：必须排除。
        self.assertNotIn((0,3),windows.indices)
        for j,(i,start) in enumerate(windows.indices):
            torch.testing.assert_close(windows.commands(torch.tensor([j]))[0],intent[start+cfg.history])
        self.assertGreater(windows.excluded_command_changes,0)

    def test_command_mapping_sampling_and_gradient(self):
        torch.manual_seed(7)
        model=TrajectoryDiffusion(34,self.cfg()).eval()
        h=torch.randn(2,2,50);state=torch.randn(2,34)
        target=torch.tensor([[1.5,0.,0.,0.,0.,-1.,.3,0.],[0.,0.,0.,-1.,0.,0.,.4,0.]])
        expected=torch.tensor([[1.,0.,1.5,0.,0.,.3],[0.,1.,0.,0.,0.,.4]])
        torch.testing.assert_close(model.command_from_target(target),expected)
        noise=torch.randn(2,model.length,model.dim)
        a=model.sample(h,state,target,0.,noise=noise)
        target[:,0]=-.8
        b=model.sample(h,state,target,0.,noise=noise)
        self.assertGreater(float((a[:,2,34:]-b[:,2,34:]).abs().max()),1e-5)
        torch.testing.assert_close(a[:,:2],h,atol=0,rtol=0)
        torch.testing.assert_close(a[:,2,:34],state,atol=0,rtol=0)
        loss=model.loss(torch.randn(2,model.length,50),command=expected)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.command_encoder.parameters()),0.)
        with self.assertRaises(ValueError):model.loss(torch.randn(2,model.length,50))


class HorizontalGuidanceTests(unittest.TestCase):
    def test_horizontal_projection_through_rise(self):
        from go2_learning.tasks.mimic.config import GuidanceCfg
        import math
        model=TrajectoryDiffusion(34,ModelCfg(history=1,horizon=2,embedding=16,heads=2,layers=1))
        cfg=GuidanceCfg(velocity=1.,orientation=0.,height=0.,yaw=0.)
        target=torch.tensor([[0.,0.,0.,-1.,0.,0.,.4]])
        for angle in (0.,math.pi/4,math.pi/2):
            gravity=torch.tensor([-math.sin(angle),0.,-math.cos(angle)])
            horizontal=torch.tensor([math.cos(angle),0.,-math.sin(angle)])
            for velocity,expected in ((gravity*3,0.),(horizontal*2,2.*3)):
                raw=torch.zeros(1,model.length,model.raw_dim)
                raw[:,:,:3]=velocity;raw[:,:,6:9]=gravity*1.7
                loss=model.guidance_terms(model.encode(raw),target,cfg)['velocity']
                self.assertAlmostEqual(float(loss),expected,places=4)
        # 四足切upright时，后退必须有代价，上升不应被当作水平漂移。
        raw=torch.zeros(1,model.length,model.raw_dim);raw[:,:,8]=-1.;raw[:,:,0]=-1.
        self.assertAlmostEqual(float(model.guidance_terms(model.encode(raw),target,cfg)['velocity']),1.5)
        raw.requires_grad_(True)
        cost=model.guidance_terms(model.encode(raw),target,cfg)['velocity']
        self.assertTrue(torch.isfinite(torch.autograd.grad(cost,raw)[0]).all())
