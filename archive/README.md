# 早期学习与维护文件归档

2026-10-13 按用途归档，文件内容保持原样。

| 现在的位置 | 原来的位置 | 用途 |
|---|---|---|
| `early_gym_demo/src/` | 根目录 `src/` | 早期 Gymnasium CartPole 随机动作示例 |
| `early_gym_demo/docker/` | 根目录 `docker/` | 早期普通 Python Docker 配置 |
| `maintenance/move-docker-to-home.sh` | 根目录同名脚本 | 曾用于本机 Docker 存储迁移的一次性脚本 |

早期 Dockerfile 引用了 `requirements.txt`，本次检查时该示例未附带这个文件，
`compose.yaml` 也是空文件。因此这里保留为学习记录，不作为当前训练入口。

存储迁移脚本会修改宿主机服务和系统配置，仅针对当时机器状态设计；
当前训练、复制工程到服务器都不需要执行它。

当前可运行的小工程见 [../examples/rsl_rl_version/README.md](../examples/rsl_rl_version/README.md)。
