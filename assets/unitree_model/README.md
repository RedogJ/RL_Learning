# 宇树 Go2 USD 模型

入口文件：[Go2/usd/go2.usd](Go2/usd/go2.usd)。加载入口时保留整个 `Go2/` 目录结构。

来源：[宇树官方 Hugging Face 模型库](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main/Go2)。

- 下载日期：2026-10-13。
- 固定版本：`323e350252b9c3aee9c40acbfdad84f6ce46a5ac`。
- 范围：该版本下完整的 `Go2/`，共 4 个 USD 文件、18,815,207 字节（约 18.8 MB）。
- 下载的是实际二进制 USD 文件，不是 Git LFS 指针。
- 每个文件已核对远端大小，以及 Git blob SHA-1 或 LFS SHA-256。
- 来源 URL、文件大小和本地 SHA-256 见 [download_manifest.json](Go2/download_manifest.json)。

```text
Go2/usd/
├── go2.usd
└── configuration/
    ├── go2_description_base.usd
    ├── go2_description_physics.usd
    └── go2_description_sensor.usd
```

[LICENSE](LICENSE) 原文取自同一官方项目的
[GitHub 仓库](https://github.com/unitreerobotics/unitree_model/blob/main/LICENSE)。
该仓库已注明模型更新转移至 Hugging Face。

本次完成下载及字节完整性检查，尚未在 Isaac Sim 中加载模型或核验其物理配置。
新工程 [go2_learning](../../projects/go2_learning/README.md) 的启动脚本已将本资产目录挂载到容器，
可在工程根目录执行 `bash run.sh go2` 加载。宿主机路径不会自动出现在容器中。
