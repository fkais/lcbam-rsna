# LCBAM-RSNA

一个基于 YOLOv8n 和轻量注意力机制的 RSNA 肺炎病灶检测可复现实验项目。

> **项目状态：正式实验前验证**
> 当前仓库同时保留历史实验材料和已测试的 LCBAMv2 + YOLOv8n 实现。普通 smoke 训练与独立 Optuna/TPE 结构搜索入口均已具备；正式长训练和正式贝叶斯优化尚未开始。

---

## 项目概述

本项目研究注意力机制对轻量级胸部 X 光肺炎病灶检测模型的影响。

主要实验模型包括：

- **YOLOv8n** —— 基线检测模型
- **YOLOv8n + CBAM** —— 标准卷积块注意力模块
- **YOLOv8n + LCBAMv2** —— 重新设计的轻量注意力模块

项目基于 **RSNA Pneumonia Detection Challenge** 数据集。

当前目标并不是直接复现以前的实验流程，而是重新构建整个项目，使其具备：

- 统一的训练配置；
- 可复现的数据集划分；
- 独立的模型源码；
- 标准化的评估流程；
- 多随机种子实验；
- 模型效率测试。

---

## 仓库当前状态

本仓库目前仍处于整理和正式实验前验证阶段。

现阶段主要包含：

```text
configs/          # 数据与三种 LCBAMv2 插入位置 YAML
scripts/          # 数据验证和独立 smoke 训练入口
src/models/       # LCBAMv2、YOLOv8 注册及兼容权重迁移
src/optimize_lcbam.py
tests/
archive/legacy_experiments/
```

`archive/legacy_experiments/` 中保存的是早期实验产生的历史材料。

其中可能包括：

- 旧模型配置文件；
- 旧 CBAM / LCBAM 实现；
- 训练参数；
- 实验指标；
- 训练曲线和可视化结果；
- 其他历史实验文件。

这些文件主要用于保存实验记录和追踪项目演变过程。

> **注意：** Legacy 实验不应被视为本项目最终的可复现实现。

部分早期实验使用的训练配置并不完全一致，并且旧版 LCBAM 实现与早期报告中的结构描述并不完全对应。

因此，旧实验结果不会直接作为最终实验结论。

---

## 当前项目结构

核心可执行部分已经整理为：

```text
lcbam-rsna/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── rsna.yaml
│   └── models/
│
├── src/
│   ├── models/
│   └── optimize_lcbam.py
│
├── scripts/
│   ├── train_smoke.py
│   └── validate_rsna.py
│
├── tests/
├── results/
├── paper/
│
└── archive/
    └── legacy_experiments/
```

---

## 数据集

本项目使用 **RSNA Pneumonia Detection Challenge** 数据集。

原始数据集不会包含在本 GitHub 仓库中。

数据目前已经转换成 YOLO 目标检测格式，包括：

- 胸部 X 光图像；
- YOLO 格式边界框标签；
- 数据集配置文件。

本地数据完整性与固定 train / validation / test 路径已经通过项目验证脚本检查；正式搜索仍应保留同一份划分，不得重新随机划分。数据不会提交到 Git。

可复核命令：

```powershell
.venv\Scripts\python.exe scripts\validate_rsna.py --data configs\rsna.yaml
.venv\Scripts\python.exe scripts\audit_rsna_split.py --data configs\rsna.yaml --metadata-csv E:\paperdata\meddet\raw\stage_2_train_labels.csv --patient-id-column patientId
```

数据治理仍应持续记录：

- 图片和标签是否一一对应；
- 是否存在缺失或多余标签；
- 空标签是否确实代表正常病例；
- 固定 train / validation / test 数据划分；
- 阳性病例数量；
- bounding box 数量统计。

同时会生成固定的数据划分文件：

```text
split_manifest.csv
```

用于保证后续实验使用完全相同的数据划分。

---

## 模型设计

### YOLOv8n

YOLOv8n 作为本项目的轻量级基线模型。

它在以下方面具有较好的平衡：

- 检测精度；
- 模型大小；
- 计算复杂度；
- 部署效率。

### CBAM

CBAM 依次使用：

1. 通道注意力；
2. 空间注意力。

本项目将其作为标准注意力机制对比模型。

### LCBAMv2

原来的实验版本 LCBAM 将作为历史版本保存在 archive 中。

新的轻量模块暂时命名为：

**LCBAMv2**

目前的设计方向包括：

- 使用类似 ECA 的 1D 卷积实现轻量通道注意力；
- 使用较小的空洞卷积实现轻量空间注意力；
- 初始化时明确确定通道数；
- 不在第一次 `forward()` 时动态创建参数；
- 完整支持模型保存和重新加载；
- 以低于标准 CBAM 的参数开销为设计目标；统一插入条件下的最终参数量对比仍待正式 benchmark。

当前实现已经完成结构、注册、模型构建、forward、预训练权重迁移和 CUDA smoke 验证，但尚未形成正式精度结论。

真实构造参数为：

- `kernel_size`：ECA 风格通道注意力的一维卷积核；
- `dilation`：空间分支空洞率，当前搜索固定为 3；
- `gamma_init`：残差门控初始值，当前搜索固定为 0；
- `spatial_kernel`：空间分支二维卷积核，支持正奇数，结构搜索限定为 3、5、7。

本模块没有通道压缩瓶颈，因此不存在可忠实映射的 `reduction_ratio`。为避免制造无效维度，第一阶段不搜索该参数；`num_lcbam` 同样固定为 1。

## 运行入口

### 普通训练链路

现有训练链路保持独立，不经过 Optuna。下面的命令只抽取 2% train 数据并运行 1 epoch，用于 smoke 验证，不用于报告精度：

```powershell
.venv\Scripts\python.exe scripts\train_smoke.py --model lcbamv2 --data configs\rsna.yaml --epochs 1 --imgsz 512 --batch 4 --device 0 --workers 0 --seed 0 --fraction 0.02
```

### Optuna/TPE 结构搜索

搜索入口使用 seeded `TPESampler`、`direction="maximize"`，objective 为训练返回的 validation `metrics/mAP50-95(B)`。每个 trial 的训练条件保持一致，并显式使用 `val=True, split="val"`；test split 不参与搜索或模型选择。

```powershell
.venv\Scripts\python.exe -m src.optimize_lcbam --formal --trials 20 --epochs 20 --imgsz 512 --batch 4 --device 0 --workers 0 --seed 0 --sampler-seed 0 --study-name lcbam_structure_search_v1 --storage sqlite:///results/optuna/lcbam_structure_search_v1.db --data configs/rsna.yaml --split-audit-report results/data_audits/rsna_split_audit_v1.json --weights yolov8n.pt --fraction 1.0 --optimizer AdamW --lr0 0.002 --lrf 0.01 --momentum 0.9 --weight-decay 0.0005 --warmup-epochs 3.0 --warmup-momentum 0.8 --warmup-bias-lr 0.0 --patience 0 --deterministic --amp --no-cache --output-root runs/optuna --results-root results/optuna
```

以上是已冻结的 v1 正式入口，但本次只记录、没有执行。完整固定条件、split audit 结果、test policy 和 resume 限制见 `docs/experiments/lcbam_structure_search_v1.md`。

第一阶段搜索空间：

| 参数 | 候选 | 真实含义 |
|---|---|---|
| `channel_kernel` | 3、5、7 | LCBAMv2 通道分支 Conv1d kernel |
| `spatial_kernel` | 3、5、7 | LCBAMv2 空间分支 dilated Conv2d kernel |
| `insert_position` | `p3`、`p4`、`p5` | 分别位于 backbone 的 P3/C2f 后、P4/C2f 后、P5/SPPF 后 |

每个 trial 写入独立目录：

```text
runs/optuna/<study-name>/trial_NNN/
```

持久化摘要写入：

```text
results/optuna/<study-name>/trials.csv
results/optuna/<study-name>/best_params.yaml
```

SQLite 文件位置由 `--storage` 决定。中断后使用完全相同的 `--study-name`、`--storage` 和固定条件再次执行；正式 v1 的 `--trials 20` 表示整个 study 的总 attempted-trial 预算，resume 只运行剩余数量，不会追加或扩展预算。协议不一致会拒绝 resume，已有 trial 目录不会被覆盖。当前代码会绑定 data/model YAML、关键模型/搜索源码和预训练权重的 SHA-256；预训练权重必须在创建 study 前已存在于本地，避免首次自动下载导致身份漂移。该内容指纹修复已通过自动化测试，但现有真实 GPU smoke 早于最终指纹版本，尚未用最终版本重新 smoke。这里的 resume 是 study 历史续跑，不是恢复某个中断 trial 的 checkpoint，也不承诺跨进程重启后逐 trial TPE 随机序列完全相同。

---

## 实验计划

正式实验计划比较：

| 模型 | 正式 seed / epoch 状态 |
|---|---|
| YOLOv8n | 待协议冻结 |
| YOLOv8n + CBAM | 待协议冻结 |
| YOLOv8n + LCBAMv2 | 待协议冻结 |

所有模型会尽可能保持完全一致的训练配置。

计划评估指标包括：

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- 参数量
- GFLOPs
- 模型大小
- 单张推理延迟
- FPS
- 峰值 GPU 显存占用

对于多个随机种子的实验结果，将报告：

```text
平均值 ± 标准差
```

---

## 可复现性

重构后的项目将尽量提供：

- 固定的数据集划分；
- 一致的超参数；
- 明确的随机种子；
- 独立的模型源码；
- 训练脚本；
- 评估脚本；
- Benchmark 脚本；
- 环境依赖；
- 可复现的运行命令。

最终 Python、PyTorch、CUDA 和 Ultralytics 环境将在本地及云 GPU 环境验证后确定。

---

## 历史实验

早期实验统一保存在：

```text
archive/legacy_experiments/
```

这些实验主要用于：

- 追踪项目发展过程；
- 对比历史模型实现；
- 保留早期实验记录；
- 分析旧实验与新实验之间的差异。

但这些实验可能存在：

- Batch size 不完全一致；
- 随机种子实验不完整；
- 依赖旧 Python 环境；
- 代码实现和报告描述不完全一致。

因此 archive 中的数据应被视为：

**历史参考结果，而非最终实验结果。**

---

## 当前进度

- [x] RSNA 数据集转换为 YOLO 格式
- [x] 完成旧版 YOLOv8n 实验
- [x] 完成旧版 YOLOv8n + CBAM 实验
- [x] 完成旧版 YOLOv8n + LCBAM 实验
- [x] 旧实验文件归档
- [x] 整理干净的项目结构
- [x] 设计 LCBAMv2
- [x] 实现 LCBAMv2
- [x] 完成基础单元测试
- [x] 检查本地数据完整性
- [x] 验证固定数据集划分路径
- [x] 完成 LCBAMv2 + YOLOv8 CUDA smoke
- [x] 实现 Optuna/TPE 结构搜索框架
- [x] 完成 P3/P4/P5 build 与 forward 测试
- [x] 冻结正式 Optuna 搜索协议
- [ ] 执行正式 Bayesian/TPE optimization
- [ ] 确定最佳结构参数
- [ ] 云端训练流程验证
- [ ] 多随机种子正式实验
- [ ] 模型效率测试
- [ ] 完成技术报告
- [ ] 发布最终可复现版本

---

## 声明

本项目仅用于医学影像目标检测相关的学习、研究和工程实验。

本项目 **不用于临床诊断或医疗决策**。

---

## License

项目源码许可证将在第一个稳定版本发布前确定。

RSNA 原始数据不会通过本仓库重新分发，数据使用仍需遵守其原始许可证和相关规定。
