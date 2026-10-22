"""固定形状 CUDA Graph 推理，支持 output/input 两种引导的加速对照。"""
import torch


class GraphSampler:
    def __init__(self, model, strength, guidance_mode='output', guidance_cfg=None):
        self.model, self.strength = model, strength
        self.guidance_mode = guidance_mode
        self.guidance_cfg = guidance_cfg
        self.graph = None

    def __call__(self, history, state, target):
        if state.device.type != 'cuda':
            return self.model.sample(history, state, target, self.strength, guidance_mode=self.guidance_mode,
                                     guidance_cfg=self.guidance_cfg)
        if self.graph is None:
            self.inputs = [x.clone() for x in (history, state, target)]
            self.noise = state.new_empty((state.shape[0], self.model.length, self.model.dim))
            self.noise.normal_()
            stream = torch.cuda.Stream(device=state.device)
            stream.wait_stream(torch.cuda.current_stream(state.device))
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    self.model.sample(*self.inputs, self.strength, noise=self.noise, guidance_mode=self.guidance_mode,
                                      guidance_cfg=self.guidance_cfg)
            torch.cuda.current_stream(state.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream), torch.no_grad():
                self.output = self.model.sample(*self.inputs, self.strength, noise=self.noise, guidance_mode=self.guidance_mode,
                                               guidance_cfg=self.guidance_cfg)
        for dst, src in zip(self.inputs, (history, state, target)):
            if dst.shape != src.shape or dst.device != src.device or dst.dtype != src.dtype:
                raise ValueError('CUDA Graph 输入的形状/设备/类型不能在播放途中改变')
            dst.copy_(src)
        # 在 graph 外产生新噪声；既不会每步复用固定噪声，也可用固定输入验证数值等价。
        self.noise.normal_()
        self.graph.replay()
        return self.output.clone()
