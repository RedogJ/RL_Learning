"""检查两个模型单步递推、动作槽和跨步梯度。"""
import torch
from go2_learning.tasks.upright_supermimic.paper_model import PaperWorldModel
from go2_learning.tasks.upright_supermimic.deeponet_study.model import TimeDeepONet
from go2_learning.tasks.upright_supermimic.recursive_study.train import next_step


def main():
    torch.manual_seed(1)
    state=torch.randn(2,90,requires_grad=True)
    a0=torch.randn(2,12,requires_grad=True)
    a1=torch.randn(2,12,requires_grad=True)
    context=torch.randn(2,16)
    for model,is_deep in ((PaperWorldModel(90,12),False),(TimeDeepONet(width=64,rank=4),True)):
        state.grad=None; a0.grad=None; a1.grad=None
        first,_=next_step(model,state,a0,context,is_deep)
        first.retain_grad()
        second,_=next_step(model,first,a1,context,is_deep)
        assert torch.equal(first[:,33:45],a0) and torch.equal(second[:,33:45],a1)
        assert torch.count_nonzero(second[:,80:88])==0
        second[:,:33].square().mean().backward()
        assert first.grad is not None and first.grad[:,:33].abs().sum()>0, '跨步梯度未传回'
        assert a0.grad is not None and a0.grad.abs().sum()>0, '前一步动作没有梯度'
        model.zero_grad(set_to_none=True)
    print('通过：单步输入、上一动作槽、推力槽与跨步梯度')


if __name__=='__main__':
    main()
