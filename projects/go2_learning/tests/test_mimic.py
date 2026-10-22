"""蒸馏数据和采样条件的边界检查，不依赖 Isaac Sim。"""
import tempfile
from pathlib import Path
import unittest
import torch
from go2_learning.tasks.mimic.config import ModelCfg
from go2_learning.tasks.mimic.models import ConditionalVAE, TrajectoryDiffusion
from go2_learning.tasks.mimic.features import inverse_rotate
from go2_learning.tasks.mimic.data import Windows, balanced_indices
from go2_learning.tasks.mimic.experts import load_experts


class MimicTests(unittest.TestCase):
    def test_emphasis_projection_roundtrip_and_root_weighting(self):
        from go2_learning.tasks.mimic.projection import EmphasisProjection
        projection=EmphasisProjection(34,6.,42)
        state=torch.randn(2,7,34)
        torch.testing.assert_close(projection.decode(projection(state)),state,atol=2e-5,rtol=2e-5)
        root_gain=projection.matrix[:,[0,1,2,3,4,5,6,7,8,33]].square().sum(0).mean()
        other_gain=projection.matrix[:,9:33].square().sum(0).mean()
        self.assertGreater(float(root_gain/other_gain),20.)
        clone=EmphasisProjection(34,6.,123)
        clone.load_state_dict(projection.state_dict())
        torch.testing.assert_close(clone(state),projection(state))

    def test_separate_velocity_cost_is_planar_temporal_sum_with_independent_weights(self):
        from go2_learning.tasks.mimic.config import GuidanceCfg
        model=TrajectoryDiffusion(34,self.cfg())
        trajectory=torch.zeros(1,model.length,model.raw_dim)
        trajectory[:,:,2]=20.  # vz 不混入平面速度项。
        target=torch.tensor([[1.,0.,0.,0.,0.,0.,10.,.6]])
        cfg=GuidanceCfg(velocity=2.,orientation=0.,height=0.,yaw=0.)
        terms=model.guidance_terms(model.encode(trajectory),target,cfg)
        self.assertAlmostEqual(float(terms['velocity']),model.cfg.horizon+1)
        self.assertEqual(float(terms['height']+terms['orientation']+terms['yaw']),0.)
        upright=target.clone();upright[:,0]=0.;upright[:,3]=-1.
        upright_terms=model.guidance_terms(model.encode(trajectory),upright,cfg)
        self.assertAlmostEqual(float(upright_terms['velocity']),400*(model.cfg.horizon+1))
        doubled=torch.cat((trajectory,trajectory),0)
        self.assertAlmostEqual(float(model.guidance_cost(model.encode(doubled),target.repeat(2,1),cfg)),
                               2*(model.cfg.horizon+1))

    def test_projected_training_and_sampling_keep_raw_observations(self):
        from dataclasses import replace
        from go2_learning.tasks.mimic.config import GuidanceCfg
        cfg=replace(self.cfg(),emphasis=6.)
        model=TrajectoryDiffusion(34,cfg)
        raw=torch.randn(2,model.length,50)
        loss=model.loss(raw);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.dim,84)
        self.assertEqual(model.raw_dim,50)
        clone=TrajectoryDiffusion(34,cfg);clone.load_state_dict(model.state_dict());clone.eval()
        history=raw[:,:cfg.history];state=raw[:,cfg.history,:34]
        target=torch.tensor([[0.,0.,0.,0.,0.,-1.,.3]]).repeat(2,1)
        for mode in ('input','output'):
            sample=clone.sample(history,state,target,.05,guidance_mode=mode,guidance_cfg=GuidanceCfg())
            self.assertEqual(sample.shape,raw.shape)
            torch.testing.assert_close(sample[:,:cfg.history],history,atol=0,rtol=0)
            torch.testing.assert_close(sample[:,cfg.history,:34],state,atol=0,rtol=0)
            self.assertTrue(torch.isfinite(sample).all())

    def test_input_guidance_reaches_current_latent_without_updating_weights(self):
        model=TrajectoryDiffusion(34,self.cfg()).eval()
        for p in model.parameters():p.requires_grad_(False)
        x=torch.randn(1,model.length,model.dim,requires_grad=True)
        levels=torch.full((1,model.length,2),2,dtype=torch.long)
        target=torch.tensor([[1.5,0.,0.,0.,0.,-1.,.3]])
        cost=model.guidance_cost(model(x,levels),target)
        grad=torch.autograd.grad(cost,x)[0]
        index=(0,model.cfg.history,model.state_dim)
        self.assertGreater(float(grad[:,model.cfg.history,model.state_dim:].abs().sum()),1e-8)
        # 有限差分核对：未来代价确实对当前 latent 所在的输入位置敏感。
        delta=torch.zeros_like(x);delta[index]=.001
        with torch.no_grad():
            difference=(model.guidance_cost(model(x+delta,levels),target)-
                        model.guidance_cost(model(x-delta,levels),target))/.002
        torch.testing.assert_close(grad[index],difference,atol=1e-3,rtol=.03)
        history=torch.randn(1,model.cfg.history,model.dim)
        state=torch.randn(1,34);noise=torch.randn_like(x)
        sample=model.sample(history,state,target,.05,noise=noise,guidance_mode='input')
        torch.testing.assert_close(sample[:,:model.cfg.history],history)
        torch.testing.assert_close(sample[:,model.cfg.history,:34],state)
        self.assertTrue(torch.isfinite(sample).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_skill_switch_keeps_physics_and_accepts_both_valid_postures(self):
        from types import SimpleNamespace as NS
        from go2_learning.tasks import get_task
        from go2_learning.tasks.mimic.switching import SwitchingTask, apply_keyboard_commands
        tasks={name:get_task(name) for name in ('running','upright')}
        task=SwitchingTask(tasks,'running')
        env=NS(num_envs=1,device='cpu',dt=1/30,actions=torch.randn(1,12),
               commands=NS(values=torch.zeros(1,3)),episode_length_buf=torch.tensor([22]))
        actions=env.actions.clone()
        self.assertTrue(task.select('upright',env))
        self.assertEqual(env.episode_length_buf.item(),22)
        torch.testing.assert_close(env.actions,actions)
        self.assertFalse(task.select('upright',env))
        apply_keyboard_commands(env,(1,1),tasks['running'].cfg.commands,'upright')
        self.assertTrue((env.commands.values==0).all())
        task.select('running',env)
        apply_keyboard_commands(env,(1,-1),tasks['running'].cfg.commands,'running')
        torch.testing.assert_close(env.commands.values,torch.tensor([[1.5,0.,-.6]]))
        apply_keyboard_commands(env,(0,0),tasks['running'].cfg.commands,'running')
        self.assertTrue((env.commands.values==0).all())
        # 四足水平、后腿直立均有效，翻到机背朝下则失败。
        state={'height':torch.full((3,),.4),
               'gravity':torch.tensor([[0.,0.,-1.],[-1.,0.,0.],[0.,0.,1.]])}
        torch.testing.assert_close(task.failures(state),torch.tensor([False,False,True]))

    def test_skill_keys_debounce_and_release(self):
        from go2_learning.tasks.mimic.switching import SkillKeyboard
        keyboard=SkillKeyboard.__new__(SkillKeyboard)
        keyboard.keys={1:'running',2:'upright',3:'toggle'}
        keyboard.pressed=set();keyboard.requested_skill='running';keyboard.handled=True
        keyboard.on_key_press(3,0);keyboard.on_key_press(3,0)
        self.assertEqual(keyboard.requested_skill,'upright')
        keyboard.on_key_release(3,0);keyboard.on_key_press(3,0)
        self.assertEqual(keyboard.requested_skill,'running')
        keyboard.on_deactivate();keyboard.on_key_press(2,0)
        self.assertEqual(keyboard.requested_skill,'upright')
        self.assertIsNone(keyboard.on_key_press(999,0))

    @unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA 验证图捕获与流顺序')
    def test_projected_cuda_graph_uses_physical_cost_and_preserves_conditioning(self):
        from dataclasses import replace
        from go2_learning.tasks.mimic.inference import GraphSampler
        from go2_learning.tasks.mimic.config import GuidanceCfg
        cfg=replace(self.cfg(),emphasis=6.)
        model=TrajectoryDiffusion(34,cfg).cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        history=torch.randn(1,cfg.history,model.raw_dim,device='cuda')
        state=torch.randn(1,34,device='cuda')
        target=torch.tensor([[1.,0.,0.,0.,0.,-1.,.3,.6]],device='cuda')
        costs=GuidanceCfg(velocity=2.,orientation=.5,height=1.,yaw=.3)
        for mode in ('input','output'):
            sampler=GraphSampler(model,.05,mode,costs)
            actual=sampler(history,state,target)
            with torch.no_grad():
                expected=model.sample(history,state,target,.05,noise=sampler.noise,
                                      guidance_mode=mode,guidance_cfg=costs)
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(actual[:,:cfg.history],history,atol=0,rtol=0)
            torch.testing.assert_close(actual[:,cfg.history,:34],state,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA 验证图捕获与流顺序')
    def test_cuda_graph_matches_eager_and_refreshes_inputs_and_noise(self):
        from go2_learning.tasks.mimic.inference import GraphSampler
        model=TrajectoryDiffusion(34,self.cfg()).cuda().eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        history=torch.randn(1,model.cfg.history,model.dim,device='cuda')
        state=torch.randn(1,34,device='cuda')
        target=torch.tensor([[1.5,0.,0.,0.,0.,-1.,.3,.6]],device='cuda')
        for strength,mode in ((0.,'output'),(.05,'output'),(.05,'input')):
            sampler=GraphSampler(model,strength,mode)
            old_noise=None
            for index in range(2):
                state=state+.01
                target[:,0]=-1.5 if index else 1.5
                graph=sampler(history,state,target)
                with torch.no_grad():
                    eager=model.sample(history,state,target,strength,noise=sampler.noise,guidance_mode=mode)
                torch.testing.assert_close(graph,eager,atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(graph[:,:model.cfg.history],history)
                torch.testing.assert_close(graph[:,model.cfg.history,:34],state)
                if old_noise is not None:self.assertFalse(torch.equal(old_noise,sampler.noise))
                old_noise=sampler.noise.clone()

    def cfg(self):
        return ModelCfg(hidden=32,embedding=32,heads=4,layers=1,noise_steps=4,history=2,horizon=3)

    def test_rotation_uses_xyzw_and_inverts_yaw(self):
        q=torch.tensor([[0.,0.,2**-.5,2**-.5]])
        v=torch.tensor([[[0.,1.,0.]]])
        torch.testing.assert_close(inverse_rotate(q,v),torch.tensor([[[1.,0.,0.]]]),atol=1e-6,rtol=1e-6)

    def test_windows_do_not_cross_trajectory_boundaries_and_balance_skills(self):
        cfg=self.cfg()
        trajectories=[]
        for expert,value in [('running',1.),('upright',9.)]:
            trajectories.append({'expert':expert,'state':torch.full((8,34),value),'latent':torch.zeros(8,16)})
        windows=Windows(trajectories,cfg)
        ids=balanced_indices(windows.labels,10)
        self.assertEqual(int((windows.labels[ids]==0).sum()),5)
        batch=windows.batch(ids)
        self.assertTrue(all(len(torch.unique(row[:,0]))==1 for row in batch))

    def test_vae_loss_reaches_encoder_and_decoder_and_round_trip(self):
        cfg=self.cfg();model=ConditionalVAE(cfg)
        prediction,mean,logvar=model(torch.randn(8,6),torch.randn(8,49))
        loss=prediction.square().mean()+.01*(mean.square()+logvar.exp()-logvar-1).mean()
        loss.backward()
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.decoder[0].weight.grad.abs().sum()),0)
        clone=ConditionalVAE(cfg);clone.load_state_dict(model.state_dict())
        goal=torch.randn(2,6);obs=torch.randn(2,49)
        torch.testing.assert_close(model(goal,obs,False)[0],clone(goal,obs,False)[0])

    def test_diffusion_training_and_sampling_preserve_observed_history(self):
        cfg=self.cfg();model=TrajectoryDiffusion(34,cfg)
        clean=torch.randn(4,6,50)
        loss=model.loss(clean);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        history=torch.randn(2,2,50);state=torch.randn(2,34)
        target=torch.tensor([[0.,0.,0.,0.,0.,-1.,.4]]).repeat(2,1)
        model.eval()
        sample=model.sample(history,state,target,strength=.01)
        self.assertTrue(torch.isfinite(sample).all())
        torch.testing.assert_close(sample[:,:2],history)
        torch.testing.assert_close(sample[:,2,:34],state)

    def test_fixed_experts_are_resolvable_and_integrity_checked(self):
        entries=load_experts()
        self.assertEqual(set(entries),{'running','upright'})
        self.assertIn('v25',entries['upright']['source'])

    def test_offline_trainers_save_reloadable_checkpoints(self):
        # 合成数据只测试两个训练入口的保存/加载，不代表真实技能性能。
        import contextlib
        import io
        from types import SimpleNamespace
        from go2_learning.tasks.mimic.train import train_vae, train_diffusion, load_vae
        from go2_learning.tasks.mimic.experts import sha256
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';data.mkdir()
            experts=load_experts()
            for expert in ('running','upright'):
                trajectories=[]
                for i in range(3):
                    trajectories.append({'proprio':torch.randn(24,49),'intent':torch.randn(24,6),
                        'teacher_action':torch.randn(24,12),'state':torch.randn(24,34),'failed':False})
                torch.save({'format':'go2_mimic_v1','expert':expert,'experts':experts,
                            'joint_names':['test'],'body_names':['test'],'control_dt':1/30,
                            'trajectories':trajectories},data/f'{expert}.pt')
            args=SimpleNamespace(data=data,device='cpu',seed=42,updates=2,batch_size=4,
                                 resume=None,output=root/'vae.pt')
            with contextlib.redirect_stdout(io.StringIO()):train_vae(args)
            vae,_=load_vae(args.output,'cpu')
            for path in data.glob('*.pt'):
                content=torch.load(path,weights_only=False)
                content['vae_sha256']=sha256(args.output)
                for trajectory in content['trajectories']:
                    trajectory['latent']=torch.randn(24,16)
                torch.save(content,path)
            args.vae=args.output;args.output=root/'diffusion.pt'
            with contextlib.redirect_stdout(io.StringIO()):train_diffusion(args)
            saved=torch.load(args.output,weights_only=False)
            self.assertEqual(saved['stage'],'state_latent_diffusion')
            self.assertEqual(saved['vae_sha256'],sha256(args.vae))

    def test_running_expert_readiness_rejects_stationary_teacher(self):
        from go2_learning.tasks.mimic.data import validate_running_response
        trajectory={'state':torch.zeros(200,34),'intent':torch.zeros(200,6)}
        trajectory['intent'][:,2]=1.5
        with self.assertRaisesRegex(ValueError,'速度响应未通过'):
            validate_running_response([trajectory],1/30)
        trajectory['state'][:,0]=1.4
        report=validate_running_response([trajectory],1/30)
        self.assertAlmostEqual(report['forward']['actual_vx'],1.4,places=5)
        trajectory['state'][:,0]=-1.4
        with self.assertRaises(ValueError):validate_running_response([trajectory],1/30)

    def test_unrelated_teacher_change_keeps_upright_supervision_usable(self):
        from copy import deepcopy
        from go2_learning.tasks.mimic.data import load_trajectories
        with tempfile.TemporaryDirectory() as folder:
            manifest=load_experts();old=deepcopy(manifest)
            old['running']['sha256']='old-running'
            payload={'format':'go2_mimic_v1','expert':'upright','experts':old,
                     'joint_names':['test'],'body_names':['test'],'control_dt':1/30,
                     'trajectories':[{'state':torch.zeros(2,34),'proprio':torch.zeros(2,49),'failed':False}]}
            path=Path(folder)/'upright.pt';torch.save(payload,path)
            rows,_=load_trajectories(folder)
            self.assertEqual(len(rows),1)
            payload['expert']='running';torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'不同专家'):load_trajectories(folder)

    def test_runtime_rejects_changed_action_limits(self):
        from types import SimpleNamespace
        from go2_learning.tasks.mimic.experts import validate_runtime
        env=SimpleNamespace(robot=SimpleNamespace(joint_names=['a']),
                            default_joint_pos=torch.zeros(1,1),
                            action_joint_limits=torch.tensor([[[-1.,1.]]]),dt=1/30)
        saved={'joint_names':['a'],'default_joint_pos':[0.],'action_joint_limits':[[-1.,1.]],
               'env_cfg':{'physics_dt':1/120,'decimation':4}}
        validate_runtime(env,saved)
        env.action_joint_limits[0,0,1]=.5
        with self.assertRaisesRegex(ValueError,'action_joint_limits'):validate_runtime(env,saved)

    def test_legacy_integral_reconstruction_preserves_expert_inputs(self):
        from go2_learning.tasks.mimic.supervision import complete_proprio, teacher_observation
        state=torch.zeros(4,34);state[:,6:9]=torch.tensor([.1,.2,-.97])
        proprio=torch.zeros(4,46);proprio[:,45]=.25
        intent=torch.tensor([[1.,0.,1.5,0.,0.,.3]]).repeat(4,1)
        trajectory=complete_proprio({'proprio':proprio,'intent':intent,'state':state},'running',1/30)
        full=trajectory['proprio']
        self.assertEqual(full.shape[-1],49)
        torch.testing.assert_close(full[0,46:],torch.zeros(3))
        self.assertTrue((full[1,46:].abs()>0).all())
        obs=teacher_observation(full,intent)
        self.assertEqual(obs.shape[-1],52)
        torch.testing.assert_close(obs[:,45:48],intent[:,2:5])
        torch.testing.assert_close(obs[:,49:],full[:,46:])

    def test_goal_relabels_share_validation_split_with_source_trajectory(self):
        from go2_learning.tasks.mimic.supervision import split_trajectories
        trajectories=[]
        for expert in ('running','upright'):
            for source in range(6):
                key=f'{expert}_{source}'
                trajectories.append({'expert':expert,'trajectory_id':key})
                trajectories.append({'expert':expert,'trajectory_id':key+'_relabel',
                                     'source_trajectory_id':key})
        train,valid=split_trajectories(trajectories,ModelCfg())
        def families(rows):return {t.get('source_trajectory_id',t['trajectory_id']) for t in rows}
        self.assertFalse(families(train)&families(valid))
        self.assertTrue(families(train) and families(valid))
