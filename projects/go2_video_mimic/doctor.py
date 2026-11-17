"""Check the portable training environment without starting a training run."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch
import warp
import mujoco_warp
from go2_mujoco_mimic.environment import Go2Environment
from .config import MOTION


def main():
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'scipy', 'mujoco', 'mujoco-warp', 'warp-lang', 'pillow')}
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: check the NVIDIA driver and Docker GPU runtime')
    # Allocate on the GPU to check driver/runtime compatibility, beyond visibility.
    torch.ones(1, device='cuda').add_(1)
    cpu = Go2Environment(600, 50)
    frames = np.load(MOTION, allow_pickle=False)
    meta = json.loads(MOTION.with_name('clip.json').read_text())
    if frames.shape != (meta['frames'], 61) or not np.isfinite(frames).all():
        raise ValueError('Invalid default reference')
    encoder = os.environ.get('VIDEO_FFMPEG') or shutil.which('ffmpeg')
    video_ready = bool(encoder and Path(encoder).is_file())
    with mujoco.Renderer(cpu.model, height=64, width=64) as renderer:
        renderer.update_scene(cpu.data)
        renderer.render()
    print(json.dumps(dict(versions=versions, gpu=torch.cuda.get_device_name(),
                          reference=str(MOTION), frames=len(frames), fps=meta['fps'],
                          duration_s=meta['duration_s'], playback_speed=meta['playback_speed'],
                          encoder=encoder if video_ready else None, video_ready=video_ready,
                          egl_render='ok', status='training_ready'), indent=2))
    if not video_ready:
        print('Training is ready. For MP4 output, copy .cache/ffmpeg from the original machine or set VIDEO_FFMPEG.')


if __name__ == '__main__':
    main()
