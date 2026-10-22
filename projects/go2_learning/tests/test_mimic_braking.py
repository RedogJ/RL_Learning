"""指令切换、制动验收和训练窗口的边界检查。"""
from types import SimpleNamespace as NS
import unittest
import torch
from go2_learning.tasks.mimic.commands import TransitionCommands
from go2_learning.tasks.mimic.response import braking_metrics
from go2_learning.tasks.mimic.data import Windows
from go2_learning.tasks.mimic.config import ModelCfg,GuidanceCfg
from go2_learning.tasks.mimic.models import TrajectoryDiffusion


class BrakingTests(unittest.TestCase):
    def test_profile_is_seeded_and_restarts_without_touching_physics(self):
        a=TransitionCommands(2,'cpu',.1,42);b=TransitionCommands(2,'cpu',.1,42)
        torch.testing.assert_close(a.ends,b.ends)
        env=NS(episode_length_buf=torch.tensor([0,30]),commands=NS(values=torch.zeros(2,3)))
        a.apply(env)
        self.assertEqual(float(env.commands.values[0,0]),0.)
        self.assertEqual(abs(float(env.commands.values[1,0])),1.5)
        self.assertEqual(env.episode_length_buf.tolist(),[0,30])
        env.episode_length_buf.zero_();a.reset(torch.tensor([1]));a.apply(env)
        self.assertTrue((env.commands.values==0).all())

    def traces(self):
        state=torch.zeros(20,1,34);state[:6,0,0]=1.5
        commands=torch.zeros(20,1,3);commands[:5,0,0]=1.5
        active=torch.ones(20,1,dtype=torch.bool)
        done=torch.zeros_like(active);failed=torch.zeros_like(active)
        return state,commands,active,done,failed

    def test_brake_requires_prior_motion_and_sustained_stop(self):
        trace=self.traces()
        result=braking_metrics(*trace,dt=.1)
        self.assertEqual(result['successful_stops'],1)
        self.assertAlmostEqual(result['mean_settle_time'],.1)
        trace[0].zero_()
        result=braking_metrics(*trace,dt=.1)
        self.assertEqual(result['eligible_stops'],0)
        self.assertEqual(result['not_moving'],1)

    def test_reset_is_not_a_successful_brake(self):
        state,commands,active,done,failed=self.traces()
        state[6:,0,0]=1.
        done[7]=True;failed[7]=True
        state[8:]=0.;active[8:10]=False
        result=braking_metrics(state,commands,active,done,failed,dt=.1)
        self.assertEqual(result['successful_stops'],0)
        self.assertEqual(result['events'][0]['status'],'failed')

    def test_windows_keep_command_changes_but_never_cross_episodes(self):
        cfg=ModelCfg(history=2,horizon=3)
        rows=[]
        for expert in ('running','upright'):
            intent=torch.zeros(24,6)
            if expert=='running':intent[12:,2]=1.5
            rows.append({'expert':expert,'state':torch.zeros(24,34),'latent':torch.zeros(24,16),'intent':intent})
        windows=Windows(rows,cfg)
        ids=windows.sample_ids(40,.5)
        running=ids[windows.labels[ids]==0]
        self.assertEqual(len(running),20)
        self.assertEqual(int(windows.transitions[running].sum()),10)
        self.assertEqual(int(windows.transitions.sum()),cfg.history+cfg.horizon)

    def test_velocity_cost_sums_time_without_decay_or_average(self):
        model=TrajectoryDiffusion(34,ModelCfg(history=2,horizon=4,embedding=32,layers=1))
        raw=torch.zeros(1,model.length,model.raw_dim)
        target=torch.tensor([[1.,0.,0.,0.,0.,-1.,.3]])
        cfg=GuidanceCfg()
        raw[:,model.cfg.history,:2]=target[:,:2]
        x=model.encode(raw)
        self.assertAlmostEqual(float(model.guidance_terms(x,target,cfg)['velocity']),.5*model.cfg.horizon)
        early=raw.clone();early[:,model.cfg.history+1:,0]=1.;early[:,model.cfg.history+1,0]=0.
        late=raw.clone();late[:,model.cfg.history+1:,0]=1.;late[:,-1,0]=0.
        self.assertAlmostEqual(float(model.guidance_terms(model.encode(early),target,cfg)['velocity']),
                               float(model.guidance_terms(model.encode(late),target,cfg)['velocity']))


if __name__=='__main__':unittest.main()
