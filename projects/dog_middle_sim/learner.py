"""多步 WM 监督、闭环可微想象和 λ-return Critic 回归。"""
import torch
from .math_utils import rotvec
from .math_utils import proprio
from .models import state_loss,critic_observation


def optimize(optimizer,loss,parameters,element_clip=None):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters=list(parameters)
    if element_clip is not None:
        torch.nn.utils.clip_grad_value_(parameters,element_clip)
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
    optimizer.step()
    return norm.detach()


def fit_world(model,optimizer,batch,task,cfg):
    state=batch['state'][:,0]
    alive=torch.ones_like(batch['done'][:,0],dtype=torch.float32)
    total=state.new_zeros(());denominator=state.new_zeros(())
    for t in range(batch['action'].shape[1]):
        state=model(state,batch['action'][:,t],task.context.expand(state.shape[0],4))
        loss=state_loss(state,batch['next_state'][:,t],cfg.model)
        total=total+(loss*alive).sum();denominator=denominator+alive.sum()
        alive=alive*(~batch['done'][:,t]).float()
    loss=total/denominator.clamp_min(1)
    optimize(optimizer,loss,model.parameters())
    return loss.detach()


@torch.no_grad()
def validate_world(model,batch,task):
    """保留环境上的闭环状态滚动预测，报告物理单位误差，不使用归一化 loss 冒充精度。"""
    state=batch['state'][:,0]
    valid=torch.ones_like(batch['done'][:,0],dtype=torch.bool)
    logs={}
    horizon=batch['action'].shape[1]
    for t in range(horizon):
        state=model(state,batch['action'][:,t],task.context.expand(state.shape[0],4))
        target=batch['next_state'][:,t]
        if t==0 or t==horizon-1:
            count=valid.sum()
            values={'root_cm':(state[:,:3]-target[:,:3]).norm(dim=-1)*100,
                    'root_deg':rotvec(state[:,3:7],target[:,3:7]).norm(dim=-1)*180/torch.pi,
                    'joint_deg':(state[:,7:19]-target[:,7:19]).square().mean(-1).sqrt()*180/torch.pi}
            for name,error in values.items():
                logs[f'wm_eval_{t+1}step_{name}']=((error*valid).sum()/count.clamp_min(1)).item() if count else None
            logs[f'wm_eval_{t+1}step_samples']=int(count.item())
        valid=valid&~batch['done'][:,t]
    return logs


def imagine(actor,critic,model,task,start,cfg,actor_optimizer,critic_optimizer,update_actor=True):
    tc=cfg.train; ec=cfg.env
    episode_steps=round(ec.episode_seconds*ec.control_hz)
    state=start['state'][:,0];history=start['history'][:,0]
    step=start['step'][:,0].float();last_action=start['previous_action'][:,0]
    n=state.shape[0];survival=state.new_ones(n);objective=state.new_zeros(n)
    model.requires_grad_(False);critic.requires_grad_(False);actor.requires_grad_(update_actor)
    observations=[];rewards=[];dones=[]
    vmax=task.max_step_reward/(1-tc.gamma)
    for t in range(tc.horizon):
        observation=critic_observation(model,task,state,history,step,episode_steps)
        action=actor(history,task.goal(state))
        predicted=model(state,action,task.context.expand(n,4))
        reward=sum(task.reward(state,predicted,action,last_action).values())
        done=1-(1-task.termination(predicted,soft=True))*(step+1<episode_steps).float()
        objective=objective+tc.gamma**t*survival*reward
        observations.append(observation.detach());rewards.append(reward.detach());dones.append(done.detach())
        survival=survival*(1-done)
        state=predicted;step=step+1;last_action=action
        obs=proprio(state,action,task.default)
        history=torch.cat((history[:,1:],obs[:,None,:]),1)
    tail_obs=critic_observation(model,task,state,history,step,episode_steps)
    objective=objective+tc.gamma**tc.horizon*survival*critic(tail_obs).clamp(max=vmax)
    actor_grad=state.new_zeros(())
    if update_actor:
        actor_grad=optimize(actor_optimizer,-objective.mean(),actor.parameters(),element_clip=0.1)
    # 固定目标：使用 rollout 时的 detached 奖励和软终止，不回传进 Actor/WM。
    observations=torch.stack(observations)
    with torch.no_grad():
        values=critic(observations)
        next_value=critic(tail_obs.detach())
        advantage=torch.zeros_like(next_value);targets=[]
        for t in reversed(range(tc.horizon)):
            continuation=1-dones[t]
            delta=rewards[t]+tc.gamma*continuation*next_value-values[t]
            advantage=delta+tc.gamma*tc.lam*continuation*advantage
            targets.append((advantage+values[t]).clamp(max=vmax))
            next_value=values[t]
        targets=torch.stack(targets[::-1]).flatten()
    inputs=observations.flatten(0,1)
    critic.requires_grad_(True)
    losses=[]
    for _ in range(tc.critic_epochs):
        for ids in torch.randperm(len(targets),device=targets.device).tensor_split(min(tc.critic_minibatches,len(targets))):
            loss=(critic(inputs[ids])-targets[ids]).square().mean()
            optimize(critic_optimizer,loss,critic.parameters())
            losses.append(loss.detach())
    model.requires_grad_(True);actor.requires_grad_(True)
    return {'imagined_objective':objective.detach().mean(),'critic_loss':torch.stack(losses).mean(),
            'actor_grad':actor_grad,'imagined_survival':survival.detach().mean()}
