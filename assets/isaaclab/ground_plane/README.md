# Isaac Lab 官方棋盘地面

入口：`default_ground_plane.usda`。来源为 NVIDIA Isaac 6.1 资产库的
`Isaac/IsaacLab/Environments/Grid/default_ground_plane_checker_v1/`。
原始文件未经修改，完整下载地址、字节数、SHA-256 和 ETag 见 `download_manifest.json`。
下载日期：2026-10-13；三个文件合计 6,040 字节。

- `default_ground_plane.usda`：场景、碰撞平面、网格和材质定义。
- `Materials/Textures/default_ground_plane_albedo.png`：512 × 512 棋盘颜色贴图。
- `Materials/Textures/default_ground_plane_roughness.png`：512 × 512 粗糙度贴图。
- `OmniPBR.mdl`：由现有 Isaac Sim 运行环境提供的内置材质模块，不是缺失的下载文件。

USD 文件头保留了 Isaac Lab 的版权和 BSD-3-Clause 标识。
迁移时保留整个目录。`go2_learning/run.sh` 已通过 assets 挂载提供离线访问。

地面的物理形状是静态平面，可视网格默认 100 m × 100 m、棋盘格每格 1 m。
`go2_learning/scene.py` 使用 `GroundPlaneCfg` 加载，设置 z=0、静摩擦 0.5、
动摩擦 0.5、恢复系数 0。这些是学习场景参数，不代表实测地面参数。

已检查 USD 的相对资产引用和 PNG 文件完整性。实际渲染与接触效果需在 Docker 中运行验证。
