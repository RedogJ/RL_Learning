"""保存每回合真实 context 的 replay，允许终止作为窗口最后一帧。"""
import torch
from go2_learning.tasks.upright_supermimic.paper_context import CONTEXT_DIM


class PaperBuffer:
    def __init__(self,capacity,num_envs,state_dim,action_dim,device):
        self.capacity=capacity;self.num_envs=num_envs;self.device=device;self.count=0
        sh=(capacity,num_envs)
        self.state=torch.zeros(*sh,state_dim,device=device)
        self.next_state=torch.zeros_like(self.state)
        self.action=torch.zeros(*sh,action_dim,device=device)
        self.context=torch.zeros(*sh,CONTEXT_DIM,device=device)
        self.fail=torch.zeros(*sh,dtype=torch.bool,device=device)
        self.done=torch.zeros_like(self.fail);self.active=torch.zeros_like(self.fail)
        self.remaining=torch.zeros(*sh,device=device)
        self.step_id=torch.full((capacity,),-1,dtype=torch.long,device=device)

    def add(self,state,action,context,next_state,fail,done,active,remaining):
        i=self.count%self.capacity
        for key in ('state','action','context','next_state','fail','done','active','remaining'):
            getattr(self,key)[i].copy_(locals()[key])
        self.step_id[i]=self.count;self.count+=1

    def windows(self,n,horizon,env_ids):
        start=max(0,self.count-self.capacity);end=self.count-horizon
        if end<start:raise ValueError('缓冲区没有足够连续步骤')
        starts=torch.arange(start,end+1,device=self.device)
        slots=(starts[:,None]+torch.arange(horizon,device=self.device))%self.capacity
        sequential=(self.step_id[slots]==starts[:,None]+torch.arange(horizon,device=self.device)).all(1)
        flags=self.active[slots][:,:,env_ids].all(1)
        if horizon>1: flags &= ~self.done[slots[:,:-1]][:,:,env_ids].any(1)
        # 多步窗口不能混入新的回合参数。
        context=self.context[slots][:,:,env_ids]
        flags &= (context==context[:,:1]).all(-1).all(1)
        valid=(flags & sequential[:,None]).nonzero()
        if not len(valid):raise ValueError('没有有效连续世界模型窗口')
        chosen=valid[torch.randint(len(valid),(n,),device=self.device)]
        t=slots[chosen[:,0]];e=env_ids[chosen[:,1]][:,None].expand_as(t)
        return tuple(getattr(self,k)[t,e] for k in ('state','action','context','next_state','fail','done','remaining'))

    def transitions(self,n,env_ids):
        ids=torch.arange(min(self.count,self.capacity),device=self.device)
        valid=self.active[ids][:,env_ids].nonzero()
        if not len(valid):raise ValueError('缓冲区没有有效样本')
        chosen=valid[torch.randint(len(valid),(n,),device=self.device)]
        t=ids[chosen[:,0]];e=env_ids[chosen[:,1]]
        return tuple(getattr(self,k)[t,e] for k in ('state','action','context','next_state','fail','done','remaining'))
