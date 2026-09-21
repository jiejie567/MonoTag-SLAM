<div align="center">

# MonoTag SLAM

**结合方形标记的单目米制重建**

<a href="https://anyverse.com/"><img src="docs/images/anyverse-dynamics-logo.png" width="280" alt="无界动力"></a>

[完整采集方案](https://github.com/jiejie567/MonoEgo) · [安装说明](docs/INSTALL.md) · [复现说明](REPRODUCIBILITY.md)

</div>

[English](README.md) | 中文

MonoTag 在 ORB-SLAM3 的自然特征地图中引入已知尺寸的 marker 角点约束，从单目视频重建米制相机轨迹。它面向离线处理，支持 marker 优先初始化、视觉地图米制化、区间尺度再锚定、共同 marker 合图，以及有几何证据支持的历史帧回溯定位。

仓库沿用 **EgoMono** 名称以保持链接兼容；算法名称是 **MonoTag SLAM**，完整采集系统叫 **MonoEgo**。目前是 private 的待发布整理版。

## 核心功能

- 固定 marker 可在背景三角化条件尚未满足时提供米制相机初值，不为背景伪造深度。
- 新的 marker 观测约束关联路径的尺度；更新通过几何检查后才提交。
- 通过视觉及共同锚点验证回环、重定位和地图合并，不因 ID 相同就强行合图。
- 利用后续地图匹配恢复初始化前或短暂失跟的帧，不把插值当作有效测量。
- 固定工位码与动态腕带码分开处理，腕带不会成为静态 SLAM 锚点。

![离线回溯](docs/images/retrospective-recovery.png)

## 安装与运行

重建在 Ubuntu 24.04 上验证。先按[安装说明](docs/INSTALL.md)准备原生依赖和 ORB 词袋：

```bash
git clone https://github.com/jiejie567/EgoMono.git
cd EgoMono
make setup
make native DEPS_PREFIX=/path/to/dependencies
.venv/bin/python scripts/init_runtime.py \
  --deps-prefix /path/to/dependencies --vocabulary /path/to/ORBvoc.txt
make smoke
make native-smoke DEPS_PREFIX=/path/to/dependencies
```

```bash
.venv/bin/python process_monotag.py input.mp4 \
  --calib camera.json --head-slam --slam-init auto --auto-marker-map \
  --static-marker-ids 20-49 --static-marker-size-mm 48 \
  --no-hand-joints --slam-replay --save-atlas runs/demo/atlas.osa \
  --output runs/demo/actions.jsonl
```

尺寸、ID 和内参必须与实际录制一致。48 mm 只是示例。每次运行使用新的输出目录。纯 SLAM 不需要 GPU 或学习模型；手部推理为可选模块，权重需按各自许可另行获取。

## 输出与边界

输出包含相机/腕带轨迹、地图身份和修订号、米制状态及有效性。不能可靠定位的区间保留无效；回放中的显示平滑不改变测量标签。最终离线地图与当时建图过程分别呈现。

Odin 里程计仅作为相机参考，不输入算法。静止腕带散布是精密度，不等于动态或解剖学手腕绝对精度。跨平台重编译不承诺逐位一致。

源码为 GPL-3.0，第三方声明保留。原创文档另按 CC BY 4.0 提供；公司 Logo 不授予商标使用权。详见 [LICENSE](LICENSE) 和[素材许可](docs/ASSET_LICENSE.md)。
