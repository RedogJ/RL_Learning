sudo docker build -f docker/Dockerfile.base -t isaaclab-base:local .
这个命令是 构建isaaclab的base镜像

然后遇到了一个问题 需要NVDIA Container Toolkit 才能用 --gpus all 让容器看到gpu