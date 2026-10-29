"""验证历史窗口不跨回合、未来动作遮罩及历史梯度。"""
import torch

from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.history_study.sampler import WindowSampler
from go2_learning.tasks.upright_supermimic.history_study.model import HistoryDeepONet, HistoryMLP


def main():
    torch.manual_seed(7)
    buf = PaperBuffer(16, 2, 90, 12, 'cpu')
    for i in range(20):
        state = torch.full((2, 90), float(i))
        action = torch.zeros(2, 12)
        context = torch.full((2, 16), float(i >= 11))
        done = torch.tensor([i == 10, i == 13])
        active = torch.tensor([i != 6, i != 8])
        buf.add(state, action, context, state + 1, done, done, active, torch.zeros(2))
    for allow_last in (True, False):
        h = 4
        sampler = WindowSampler(buf, h, torch.arange(2), allow_last_done=allow_last)
        observed = {(int(sampler.starts[i]), int(sampler.env_ids[j])) for i, j in sampler.valid.tolist()}
        expected = set()
        for start in range(max(0, buf.count - buf.capacity), buf.count - h + 1):
            for env in range(2):
                slots = [(start + k) % buf.capacity for k in range(h)]
                if all(buf.step_id[s] == start + k and buf.active[s, env] and
                       torch.equal(buf.context[s, env], buf.context[slots[0], env]) and
                       (not buf.done[s, env] or (k == h - 1 and allow_last)) for k, s in enumerate(slots)):
                    expected.add((start, env))
        assert observed == expected, '有效历史窗口与逐项检查不一致'
    history = torch.randn(2, 10, 90, requires_grad=True)
    actions = torch.randn(2, 8, 12, requires_grad=True)
    context = torch.randn(2, 16)
    model = HistoryDeepONet(width=64, rank=4)
    pred, _ = model(history, actions, context, 3)
    changed = actions.detach().clone(); changed[:, 3:] += 100
    pred_changed, _ = model(history, changed, context, 3)
    assert torch.equal(pred, pred_changed), '未来动作泄漏'
    assert torch.equal(pred[:, 33:45], actions[:, 2]), '执行动作槽不正确'
    pred[:, :33].sum().backward()
    assert history.grad[:, 0].abs().sum() > 0, '最早历史帧没有梯度'
    assert actions.grad[:, :3].abs().sum() > 0 and actions.grad[:, 3:].abs().sum() == 0
    assert model.branch[0].weight.grad is not None and model.trunk[0].weight.grad is not None
    mlp = HistoryMLP()
    out, _ = mlp(history.detach(), actions.detach()[:, 0], context)
    assert out.shape == (2, 90)
    print('通过：环形 replay 窗口、回合/context 边界、历史输入、动作遮罩、梯度和输出')


if __name__ == '__main__':
    main()
