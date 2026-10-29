"""确认动作因果遮罩、动作状态和策略所需的可微性。"""
import torch

from go2_learning.tasks.upright_supermimic.deeponet_study.model import TimeDeepONet


def main():
    torch.manual_seed(5)
    model = TimeDeepONet(width=64, rank=4)
    state = torch.randn(3, 90)
    actions = torch.randn(3, 8, 12, requires_grad=True)
    context = torch.randn(3, 16)
    predicted, _ = model(state, actions, context, 3)
    changed = actions.detach().clone()
    changed[:, 3:] += 100.0
    same, _ = model(state, changed, context, 3)
    assert torch.equal(predicted, same), '未来动作泄漏进当前预测'
    assert torch.equal(predicted[:, 33:45], actions[:, 2]), '上一动作槽不正确'
    assert torch.count_nonzero(predicted[:, 80:88]) == 0, '无推力槽不为零'
    predicted[:, :33].sum().backward()
    assert actions.grad[:, :3].abs().sum() > 0, '可用动作没有梯度'
    assert actions.grad[:, 3:].abs().sum() == 0, '未来动作有梯度'
    assert model.branch[0].weight.grad is not None and model.trunk[0].weight.grad is not None
    print('DeepONet: 因果遮罩、动作槽、推力槽及梯度检查通过')


if __name__ == '__main__':
    main()
