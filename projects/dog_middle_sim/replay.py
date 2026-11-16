"""按时间×环境存放真实 transition；done 由学习器屏蔽跨 reset 的窗口尾部。"""
import torch


class Replay:
    def __init__(self, capacity, device):
        self.capacity,self.device,self.count=capacity,device,0
        self.storage=None

    def add(self, row):
        if self.storage is None:
            self.storage={k:torch.empty((self.capacity,*v.shape),device=self.device,dtype=v.dtype)
                          for k,v in row.items()}
        for k,v in row.items():
            self.storage[k][self.count%self.capacity].copy_(v.detach())
        self.count+=1

    def sample(self, batch, length=1, env_count=None, env_offset=0):
        size=min(self.count,self.capacity)
        if size<length:
            raise ValueError('真实轨迹窗口不足')
        n=next(iter(self.storage.values())).shape[1]
        n=n-env_offset if env_count is None else env_count
        if n<1 or env_offset<0 or env_offset+n>next(iter(self.storage.values())).shape[1]:
            raise ValueError('环境采样范围无效')
        start=torch.randint(self.count-size,self.count-length+1,(batch,),device=self.device)
        env=torch.randint(n,(batch,),device=self.device)+env_offset
        slots=(start[:,None]+torch.arange(length,device=self.device))%self.capacity
        return {k:v[slots,env[:,None]] for k,v in self.storage.items()}
