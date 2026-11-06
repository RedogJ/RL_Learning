# 从仓库根目录构建：docker build -f projects/go2_mujoco_mimic/docker/Dockerfile -t go2-supermimic:local .
# 独立 MuJoCo 环境；已有 isaaclab-base:local 用户可直接复用原镜像。
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libgles2 libglfw3 libx11-6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY projects/go2_mujoco_mimic/requirements.txt /tmp/supermimic-requirements.txt
RUN python -m pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130 \
    && python -m pip install --no-cache-dir -r /tmp/supermimic-requirements.txt
ENV PYTHONUNBUFFERED=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
WORKDIR /workspace
