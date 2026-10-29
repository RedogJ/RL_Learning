"""专家/学生播放、带专家标签的 DAgger 采集、潜变量采集和扩散闭环。"""
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
import json
import time
import torch
from .experts import expert_path, load_experts, sha256
from .features import extract
from .train import load_vae, save_atomic
from .models import ConditionalVAE, TrajectoryDiffusion
from .config import ModelCfg, GuidanceCfg


def run(args):
    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_physx.physics import PhysxCfg
    from go2_learning.tasks import get_task
    from go2_learning.simulation.env import make_env
    from rsl_rl.runners import OnPolicyRunner
    torch.manual_seed(args.seed)
    guidance_cfg=GuidanceCfg(args.cost_mode,args.velocity_guidance_weight,args.orientation_guidance_weight,
                             args.height_guidance_weight,args.yaw_guidance_weight)
    checkpoint = expert_path(args.expert)
    run_cfg = json.loads(checkpoint.with_name('run.json').read_text())
    task = get_task(args.expert);task.load_playback_config(run_cfg)
    switch_task=None;keyboard=None;velocity_keyboard=None;mouse_push=None;keyframe_timeline=None;switches=0
    if args.skill_switch:
        from .switching import SwitchingTask
        tasks={args.expert:task}
        for name in ('running','upright'):
            if name not in tasks:
                other_cfg=json.loads(expert_path(name).with_name('run.json').read_text())
                tasks[name]=get_task(name);tasks[name].load_playback_config(other_cfg)
        task=switch_task=SwitchingTask(tasks,args.expert)
    cfg = SimulationCfg(dt=task.cfg.environment.physics_dt, physics=PhysxCfg(),
                        render_interval=task.cfg.environment.decimation)
    with launch_simulation(cfg,args):
        sim=SimulationContext(cfg)
        try:
            # running 采集时保留每回合 ±1.5/0 的专家命令分布，upright 命令固定 0。
            env=make_env(sim,args.num_envs,task=task,
                         random_commands=args.command=='collect' and args.command_profile=='episode')
            env.render_every_physics_steps=env.decimation
            from .experts import validate_runtime
            validate_runtime(env, run_cfg)
            if switch_task is not None:
                for name in switch_task.tasks:
                    validate_runtime(env,json.loads(expert_path(name).with_name('run.json').read_text()))
            print('固定专家：' + load_experts()[args.expert]['source'], flush=True)
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            runner.load(str(checkpoint),map_location=str(sim.device))
            teacher=runner.get_inference_policy(device=str(sim.device))
            vae=None;diffusion=None;latent_sha=None;sampler=None
            if args.controller=='vae':
                vae,_=load_vae(args.vae,env.device);vae.eval();latent_sha=sha256(args.vae)
            elif args.controller=='diffusion':
                saved=torch.load(args.model,map_location=env.device,weights_only=False)
                latent_sha=saved.get('vae_sha256')
                if saved['experts']!=load_experts():raise ValueError('模型专家来源与当前 manifest 不一致')
                model_cfg=ModelCfg(**saved['config'])
                diffusion=TrajectoryDiffusion(saved['state_dim'],model_cfg).to(env.device)
                diffusion.load_state_dict(saved['model']);diffusion.eval()
                diffusion.vae_sha256=saved.get('vae_sha256')
                if args.keyframes:
                    from .keyframes import KeyframeTimeline
                    keyframe_timeline=KeyframeTimeline(args.keyframes,diffusion,args.num_envs,env.device,
                        env.dt,args.expert,args.keyframe_entry,args.keyframe_interval)
                    print(f'[关键帧] 切upright后首帧={keyframe_timeline.entry*env.dt:.3f}s，'
                          f'间隔={keyframe_timeline.interval*env.dt:.3f}s；真实参考状态硬条件。',flush=True)
                vae=ConditionalVAE(model_cfg).to(env.device)
                vae.load_state_dict(saved['vae']['model']);vae.eval()
                # 引导对输入求导，无需计算网络参数梯度。
                for parameter in diffusion.parameters():parameter.requires_grad_(False)
                if not args.eager:
                    from .inference import GraphSampler
                    sampler=GraphSampler(diffusion,args.guidance,args.guidance_mode,guidance_cfg)
                    print('扩散推理：CUDA Graph（首次准备后复用）；去噪次数不变。',flush=True)
                print(f'状态强调={model_cfg.emphasis:g}；当前指令条件={bool(model_cfg.command_dim)}；'
                      f'引导代价={guidance_cfg}',flush=True)
            env.finish_initial_settling()
            if args.command=='play':
                from go2_learning.ui.mouse import attach_mouse_push
                mouse_push=attach_mouse_push(env)
                if mouse_push is not None:
                    env.before_physics_step=mouse_push.apply
                    env.on_reset=mouse_push.clear
            command_schedule=None
            if args.command_profile=='transitions':
                from .commands import TransitionCommands
                command_schedule=TransitionCommands(env.num_envs,env.device,env.dt,args.seed+100)
            if switch_task is not None:
                from .switching import attach_skill_keyboard
                keyboard=attach_skill_keyboard(sim,args.expert)
                print('[技能切换] 共用一个模型，过渡允许两个任务有效姿态区域的并集；'
                      '是否能完成过渡由现有策略决定。',flush=True)
            if args.keyboard:
                from go2_learning.ui.keyboard import attach_keyboard_input
                velocity_keyboard=attach_keyboard_input(sim,test_only=False)
                command_cfg=(switch_task.tasks['running'].cfg.commands if switch_task is not None
                             else get_task('running').cfg.commands)
            if args.command == 'play' and args.expert == 'running':
                env.commands.values[:, 0] = args.vx
                env.commands.values[:, 1] = args.vy
            rows=[];evaluation_rows=[];segments=[];starts=[0]*args.num_envs
            ou=torch.zeros_like(env.actions);history=None;previous_latent=None
            resets=0;failure_total=0;speeds=[]
            measured_start=None
            for step in range(args.steps):
                if command_schedule is not None:command_schedule.apply(env)
                if keyboard is not None:
                    if switch_task.select(keyboard.requested_skill,env):
                        switches+=1
                        print(f'[技能切换] step={step} → {switch_task.skill}；保留物理状态和历史，无 reset。',flush=True)
                    env.commands.values.zero_()
                    if switch_task.skill=='running':
                        env.commands.values[:,0]=args.vx
                        env.commands.values[:,1]=args.vy
                skill=switch_task.skill if switch_task is not None else args.expert
                if velocity_keyboard is not None:
                    from .switching import apply_keyboard_commands
                    apply_keyboard_commands(env,velocity_keyboard.axes,command_cfg,skill)
                active=env.policy_active.clone()
                with torch.no_grad():
                    proprio,intent,state=extract(env,skill)
                    label=(teacher(env.get_observations()).detach()
                           if args.command=='collect' or args.controller=='expert' else None)
                    latent=state.new_zeros((args.num_envs,16))
                    actions=label
                    if args.controller=='vae':
                        latent,_,_=vae.sample_latent(intent)
                        actions=vae.decode(latent,proprio)
                        if args.teacher_mix:
                            mask=torch.rand(args.num_envs,1,device=env.device)<args.teacher_mix
                            actions=torch.where(mask,label,actions)
                if args.controller=='diffusion':
                    c=diffusion.cfg
                    if history is None:
                        previous_latent=state.new_zeros((args.num_envs,c.latent_dim))
                        history=torch.cat((state,previous_latent),-1)[:,None].repeat(1,c.history,1)
                    target=state.new_zeros((args.num_envs,8 if args.keyboard else 7))
                    target[:,:2]=env.commands.values[:,:2]
                    target[:,3:6]=state.new_tensor((-1.,0.,0.) if skill=='upright' else (0.,0.,-1.))
                    target[:,6]=task.target_height
                    if args.keyboard:target[:,7]=env.commands.values[:,2]
                    keyframes=None
                    if keyframe_timeline is not None:
                        keyframe_timeline.select(skill)
                        keyframes=keyframe_timeline.future()
                    if sampler is not None:
                        prediction=(sampler(history,state,target,keyframes) if keyframes is not None
                                    else sampler(history,state,target))
                    else:
                        prediction=diffusion.sample(history,state,target,args.guidance,
                            guidance_mode=args.guidance_mode,guidance_cfg=guidance_cfg,keyframes=keyframes)
                    latent=prediction[:,c.history,diffusion.state_dim:]
                    with torch.no_grad():actions=vae.decode(latent,proprio)
                    history=torch.cat((history[:,1:],torch.cat((state,latent),-1)[:,None]),1).detach()
                with torch.no_grad():
                    # 专家标签不加噪声；仅执行动作加入 OU 扰动，构建可恢复偏差带。
                    if args.noise:
                        ou.add_(-.8*ou + args.noise*torch.randn_like(ou))
                        executed=actions+ou
                    else:executed=actions
                    _,reward,done,extras=env.step(executed,auto_reset=False,
                                                evaluate=args.command=='collect')
                    if args.command=='evaluate':
                        evaluation_rows.append({'state':state.detach().clone(),
                                                'commands':intent[:,2:5].detach().clone(),
                                                'active':active.clone(),'done':done.bool().clone(),
                                                'failed':extras['failed'].clone()})
                    if args.command=='collect':
                        rows.append({'proprio':proprio.detach().clone(),'intent':intent.detach().clone(),
                             'state':state.detach().clone(),'teacher_action':label.clone(),
                             'latent':latent.detach().clone(),'executed_action':env.actions.clone(),
                             'active':active.clone()})
                    if keyframe_timeline is not None:keyframe_timeline.advance(active)
                    speeds.append(proprio[:,:2].norm(dim=-1).detach())
                    if done.any():
                        ids=done.nonzero(as_tuple=False).flatten()
                        resets+=len(ids)
                        # failed 用任务正式失败条件；超时归成功片段。
                        failures=(extras['failed'] if args.command!='collect'
                                  else task.failures(env._read_state()))
                        for i in ids.cpu().tolist():
                            failed = bool(failures[i])
                            failure_total += int(failed)
                            if args.command=='collect':segments.append((i,starts[i],step+1,failed))
                            starts[i]=step+1
                        env.reset(ids);ou[ids]=0
                        if keyframe_timeline is not None:keyframe_timeline.reset(ids)
                        if command_schedule is not None:command_schedule.reset(ids)
                        if history is not None:
                            _,_,fresh=extract(env,skill)
                            reset_history=torch.cat((fresh,torch.zeros_like(latent)),-1)
                            history[ids]=reset_history[ids,None].expand(-1,diffusion.cfg.history,-1)
                if step%100==0:print(f'{args.controller}/{skill} {step}/{args.steps} resets={resets}',flush=True)
                if args.command in ('play','evaluate') and step==9:
                    if torch.device(env.device).type=='cuda':torch.cuda.synchronize(env.device)
                    measured_start=time.perf_counter()
            measured_elapsed=None
            if measured_start is not None and args.steps>10:
                if torch.device(env.device).type=='cuda':torch.cuda.synchronize(env.device)
                measured_elapsed=time.perf_counter()-measured_start
            if args.command=='collect':
                segments.extend((i,starts[i],args.steps,False) for i in range(args.num_envs))
                stacked={key:torch.stack([r[key] for r in rows]).cpu() for key in rows[0]}
                trajectories=[]
                for i,start,end,failed in segments:
                    mask=stacked['active'][start:end,i].bool()
                    if int(mask.sum())<2:continue
                    trajectories.append({**{key:value[start:end,i][mask].contiguous() for key,value in stacked.items() if key!='active'},
                                         'failed':failed,'truncated':end==args.steps})
                from .data import running_response, validate_running_response
                response = running_response(trajectories, env.dt) if args.expert == 'running' else {}
                if args.expert == 'running' and args.controller == 'expert':
                    validate_running_response(trajectories, env.dt)
                print('速度响应：' + json.dumps(response, ensure_ascii=False), flush=True)
                path=Path(args.output)/f'{args.expert}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.pt'
                save_atomic(path,{'format':'go2_mimic_v1','expert':args.expert,'controller':args.controller,
                    'experts':load_experts(),'vae_sha256':latent_sha,'control_dt':env.dt,
                    'latent_mode':'posterior_sample' if args.controller=='vae' else None,
                    'joint_names':list(env.robot.joint_names),'body_names':list(env.robot.body_names),
                    'noise':args.noise,'teacher_mix':args.teacher_mix,'command_profile':args.command_profile,
                    'running_response':response,'trajectories':trajectories})
                print(f'数据已保存 {path}; trajectories={len(trajectories)}',flush=True)
            report={'expert':args.expert,'controller':args.controller,'steps':args.steps,'resets':resets,'failures':failure_total,'timeouts':resets-failure_total,
                    'mean_horizontal_speed':float(torch.stack(speeds).mean()),
                    'note':'统计包含重置等待，详细性能评估应使用独立多种子长回合。'}
            if diffusion is not None:
                report.update(emphasis=diffusion.cfg.emphasis,guidance=args.guidance,
                              guidance_mode=args.guidance_mode,guidance_cost=asdict(guidance_cfg),
                              model=str(args.model),keyframes=args.keyframes,
                              keyframe_entry=args.keyframe_entry,keyframe_interval=args.keyframe_interval)
            report.update(command_profile=args.command_profile,seed=args.seed,num_envs=args.num_envs,
                          noise=args.noise,teacher_mix=args.teacher_mix,vae_sha256=latent_sha)
            if args.controller=='vae':report['latent_mode']='posterior_sample'
            if args.command=='evaluate':
                from .response import braking_metrics
                trace={key:torch.stack([row[key] for row in evaluation_rows]).cpu()
                       for key in evaluation_rows[0]}
                response=braking_metrics(**trace,dt=env.dt)
                report['braking']={k:v for k,v in response.items() if k!='events'}
                output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
                save_atomic(output/'trace.pt',{'format':'go2_mimic_evaluation_v1',**trace,'control_dt':env.dt})
                (output/'braking.json').write_text(json.dumps(response,indent=2,ensure_ascii=False))
            if switch_task is not None:report.update(skill_switches=switches,final_skill=switch_task.skill)
            if measured_elapsed is not None:
                report.update(control_hz=(args.steps-10)/measured_elapsed,
                              realtime_factor=(args.steps-10)*env.dt/measured_elapsed)
            print(json.dumps(report,ensure_ascii=False),flush=True)
            if args.command=='evaluate':(Path(args.output)/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        finally:
            if mouse_push is not None:mouse_push.close()
            if velocity_keyboard is not None:velocity_keyboard.close()
            if keyboard is not None:keyboard.close()
            SimulationContext.clear_instance()
