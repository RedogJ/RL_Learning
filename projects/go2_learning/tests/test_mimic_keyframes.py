"""未来关键帧：掩码、去噪条件、真实物理时刻与CUDA Graph。"""
from pathlib import Path
import tempfile
from unittest import TestCase,skipUnless
from unittest.mock import patch
import torch
from go2_learning.tasks.mimic.models import TrajectoryDiffusion
from go2_learning.tasks.mimic.config import ModelCfg,GuidanceCfg
from go2_learning.tasks.mimic.inference import GraphSampler
from go2_learning.tasks.mimic.keyframes import KeyframeTimeline


class KeyframeTests(TestCase):
    def cfg(self,**kw):
        return ModelCfg(history=2,horizon=8,embedding=16,heads=2,layers=1,noise_steps=3,
                        future_keyframe_prob=.2,**kw)

    def test_projected_known_future_state_exact_and_optional_path_unchanged(self):
        m=TrajectoryDiffusion(34,self.cfg(emphasis=6.)).eval()
        h=torch.randn(2,2,50);s=torch.randn(2,34);t=torch.tensor([[0.,0.,0.,0.,0.,-1.,.3]]).repeat(2,1)
        noise=torch.randn(2,m.length,m.dim);v=torch.randn(2,8,34);mask=torch.zeros(2,8,dtype=torch.bool)
        a=m.sample(h,s,t,0.,noise=noise);b=m.sample(h,s,t,0.,noise=noise,keyframes=(v,mask))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        mask[0,3]=True;mask[1,6]=True
        out=m.sample(h,s,t,.01,noise=noise,guidance_mode='input',guidance_cfg=GuidanceCfg(),keyframes=(v,mask))
        torch.testing.assert_close(out[:,3:,:34][mask],v[mask],atol=0,rtol=0)
        torch.testing.assert_close(out[:,:2],h,atol=0,rtol=0)
        torch.testing.assert_close(out[:,2,:34],s,atol=0,rtol=0)
        self.assertTrue(torch.isfinite(out).all())
        with self.assertRaises(ValueError):m.sample(h,s,t,keyframes=(v,mask.float()))

    def test_training_has_known_future_states_but_unknown_latents(self):
        cfg=self.cfg();cfg.future_keyframe_prob=1.
        m=TrajectoryDiffusion(34,cfg)
        captured=[]
        hook=m.register_forward_pre_hook(lambda module,inputs:captured.append(tuple(x.detach().clone() if torch.is_tensor(x) else x for x in inputs)))
        loss=m.loss(torch.randn(2,m.length,50));loss.backward();hook.remove()
        levels=captured[0][1]
        self.assertTrue((levels[:,3:,0]==0).all())
        self.assertTrue((levels[:,3:,1]>0).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(m.output.weight.grad.abs().sum()),0.)

    def test_keyframe_deadline_moves_toward_present_and_reset_is_per_env(self):
        m=TrajectoryDiffusion(34,self.cfg())
        reference=torch.arange(20*34,dtype=torch.float32).reshape(20,34)
        with tempfile.TemporaryDirectory() as folder,patch('go2_learning.tasks.mimic.keyframes.load_experts',return_value={}):
            p=Path(folder)/'frames.pt'
            torch.save(dict(format='go2_mimic_keyframes_v1',experts={},control_dt=.1,vae_sha256='test',upright_states=reference),p)
            timeline=KeyframeTimeline(p,m,2,'cpu',.1,entry=.4,interval=.2)
            self.assertFalse(timeline.future()[1].any())
            timeline.select('upright');v,mask=timeline.future()
            self.assertEqual(mask[0].nonzero().flatten().tolist(),[3,5,7])
            torch.testing.assert_close(v[0,3],reference[0])
            torch.testing.assert_close(v[0,5],reference[2])
            timeline.advance(torch.tensor([True,False]));v,mask=timeline.future()
            self.assertEqual(mask[0].nonzero().flatten().tolist(),[2,4,6])
            self.assertEqual(mask[1].nonzero().flatten().tolist(),[3,5,7])
            timeline.reset(torch.tensor([0]));self.assertEqual(timeline.elapsed.tolist(),[0,0])
            timeline.elapsed.fill_(30);self.assertFalse(timeline.future()[1].any())
            timeline.select('running');self.assertFalse(timeline.future()[1].any())

    @skipUnless(torch.cuda.is_available(),'需要CUDA验证动态关键帧图捕获')
    def test_graph_refreshes_mask_and_values(self):
        m=TrajectoryDiffusion(34,self.cfg(command_dim=6,emphasis=6.)).cuda().eval()
        for p in m.parameters():p.requires_grad_(False)
        h=torch.randn(2,2,50,device='cuda');s=torch.randn(2,34,device='cuda')
        t=torch.tensor([[0.,0.,0.,-1.,0.,0.,.4]],device='cuda').repeat(2,1)
        v=torch.randn(2,8,34,device='cuda');mask=torch.zeros(2,8,dtype=torch.bool,device='cuda')
        sampler=GraphSampler(m,.01,'input',GuidanceCfg())
        for i in range(2):
            mask.zero_();mask[:,i+2]=True;v.add_(.1)
            actual=sampler(h,s,t,(v,mask))
            expected=m.sample(h,s,t,.01,noise=sampler.noise,guidance_mode='input',guidance_cfg=GuidanceCfg(),keyframes=(v,mask))
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(actual[:,3:,:34][mask],v[mask],atol=0,rtol=0)
