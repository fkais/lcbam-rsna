# LCBAM-RSNA 项目长期上下文

本文件是供 AI / Codex 和项目维护者快速恢复事实状态的长期基线，不是性能宣传页。实现细节以当前源码为准，实验结果以可复核的 run 产物和日志为准，实验规则以 `EXPERIMENT_PROTOCOL.md` 为准。若三者与本文件冲突，应先核实，再更新本文件并在 `PROJECT_LOG.md` 记录纠正。

## Project Goal

- 数据集：RSNA Pneumonia Detection Challenge。
- 基础检测器：YOLOv8n。
- 当前研究对象：与 legacy `LCBAM` 分开维护的 `LCBAMv2`。
- 目标：建立可复现 baseline；集成 LCBAMv2；与 baseline / CBAM 公平比较；完成消融；用 Optuna/TPE 搜索 LCBAMv2 结构参数；做多 seed 稳定性验证；最后在独立 test set 上评估冻结方案。
- 项目用途：项目研究、开源展示和考研复试项目介绍。

除非后续正式实验产物支持，否则不得声称项目已发表论文、LCBAMv2 已优于 baseline / CBAM，或 Bayesian Optimization 已提升性能。

## Repository and Dataset

- 当前活跃仓库：`D:/project/lcbam-rsna`。
- 数据集根目录：`E:/paperdata/meddet/datasets/rsna`，由 `configs/rsna.yaml` 指向。
- 固定结构：`images/{train,val,test}` 与 `labels/{train,val,test}`。

2026-09-01 对本地目录的只读复核结果：

| split | images 目录文件 | 可训练 `.png` | 辅助 `.npy` | label `.txt` |
|---|---:|---:|---:|---:|
| train | 37,287 | 18,677 | 18,610 | 18,677 |
| val | 10,672 | 5,336 | 5,336 | 5,336 |
| test | 2,670 | 2,670 | 0 | 2,670 |

RSNA 合法包含无病灶的阴性样本，空标签文件可以正确表示无目标图像，不能仅因标签为空判定数据异常。当前转换中每张 `.png` 都有对应 `.txt`；目录级 `images` 文件数大于 label 数，主要因为 train / val 同时保存了 `.npy` 辅助文件，而不是因为阴性图像缺少标签。统计和审计必须区分文件扩展名，不能只比较目录总数。

当前 train / val / test 划分不得随意修改。2026-09-01 的 patient/content-level 最终审计已通过：26,683 张可训练 PNG 的跨 split basename、RSNA patientId 和精确内容 overlap 均为 0。train 内有一个由两张合法阴性图组成的精确重复组；原始 metadata 中另有一个已知阴性 patientId 未进入当前冻结转换数据。细节和限制见 `docs/experiments/lcbam_structure_search_v1.md`；正式入口会按全量 SHA-256 manifest 复核当前磁盘数据。

## LCBAMv2 Definition

真实定义见 `src/models/lcbam_v2.py`：

- Channel branch：对全局 average / max descriptor 堆叠后的 `[B,2,C]` 张量使用 ECA-style `Conv1d(2,1,...)`，沿通道维做局部交互。
- Spatial branch：对通道 average / max map 拼接后的 `[B,2,H,W]` 使用 dilated `Conv2d(2,1,...)`。
- Fusion：`identity + gamma * attended`，当前默认 `gamma_init=0.0`，从 identity mapping 开始。
- 当前构造参数名是 `kernel_size`、`dilation`、`gamma_init`、`spatial_kernel`。Optuna 的搜索字段 `channel_kernel` 映射到构造参数 `kernel_size`。

当前 LCBAMv2 不是传统 CBAM 的 reduction-MLP channel attention，不存在 `reduction_ratio`。未来不得因套用 CBAM 常识而自动添加该参数。

第一阶段真实结构搜索变量：

- `channel_kernel`: `[3, 5, 7]`
- `spatial_kernel`: `[3, 5, 7]`
- `insert_position`: `[p3, p4, p5]`

当前固定结构条件为 `dilation=3`、`gamma_init=0.0`、`num_lcbam=1`。其中 `num_lcbam=1` 由每个静态 YAML 恰好放置一个 LCBAMv2 层实现，不是模块构造参数。

## YOLOv8 Integration

三个支持位置由静态模型 YAML 定义，并已通过 build / full forward 测试：

| 位置 | YAML | 真实顶层层索引 | 语义 |
|---|---|---:|---|
| P3 | `configs/models/yolov8n-lcbamv2-p3.yaml` | 5 | P3 backbone C2f 后 |
| P4 | `configs/models/yolov8n-lcbamv2-p4.yaml` | 7 | P4 backbone C2f 后 |
| P5 | `configs/models/yolov8n-lcbamv2.yaml` | 10 | P5 backbone SPPF 后 |

不得把 `backbone`、`neck`、`SPPF` 等抽象字符串直接推断为合法插入位置。新增位置前必须检查真实 YOLO YAML、tensor shape、model build、完整 forward 和相关测试。预训练迁移代码会检测唯一 LCBAMv2 的真实顶层索引，并对后续层索引偏移。

## Optuna / Bayesian Optimization

- 依赖与 smoke 环境记录为 Optuna 4.9.0。
- `src/optimize_lcbam.py` 使用 seeded `TPESampler`、`direction="maximize"`。
- objective 是 training 返回的 validation `metrics/mAP50-95(B)`。
- trial 显式使用 `val=True, split="val"`；测试覆盖了 objective 不调用 test evaluation。
- 正式搜索尚未开始，当前没有“最佳结构”结论。

第一阶段只允许搜索三个结构变量；正式 v1 的 dataset、imgsz、batch、epochs、显式 AdamW、augmentation、seed policy、dilation、gamma_init 和 num_lcbam 已冻结在 `docs/experiments/lcbam_structure_search_v1.md`。CLI formal guard 会拒绝条件漂移；smoke 参数不属于正式协议。

Test set 不得用于 trial scoring、搜索空间调整、超参数选择、早停模型选择或 best-trial 选择。流程必须是 `train -> val selection -> freeze final model/protocol -> one final test evaluation`。

BO-LCBAMv2 表示对 LCBAMv2 的结构优化实验，不是一个独立提出的注意力模块。基础公平比较是 YOLOv8n、YOLOv8n + CBAM、YOLOv8n + LCBAMv2；BO-LCBAMv2 是后续优化组。

## Validation Levels

- **Code Validated**：import、build、forward 和相关 tests 正常，只证明代码级正确性。
- **Smoke Validated**：最小 CPU / GPU 训练链路成功，只证明工程链路可以运行。
- **Experiment Validated**：按冻结协议完成正式训练。
- **Result Validated**：完成合理正式训练、必要重复、多 seed 或稳定性分析，以及最终冻结模型的 test evaluation 后，才允许做性能结论。

不得把上述层级混用。

## Current Optuna Smoke Evidence

最新可用的真实 GPU 链路证据为 `lcbam_optuna_smoke_20260901_v2`：

- 1 trial，1 epoch，`fraction=0.02`，`batch=4`，`workers=0`，`imgsz=512`。
- 设备：NVIDIA GeForce RTX 3050 Laptop GPU；环境产物记录 PyTorch 2.12.1+cu132、CUDA 13.2、Ultralytics 8.4.135、Optuna 4.9.0。
- trial 0：`channel_kernel=5`、`spatial_kernel=7`、`insert_position=p5`，状态 `COMPLETE`。
- validation mAP50-95：`0.00011168922747715054`。

该数值没有性能比较意义，也不是最佳结构证据。它证明当时版本的 Optuna -> trial config -> model YAML -> YOLO build -> pretrained weight migration -> CUDA training -> validation -> metric return -> SQLite/CSV/YAML persistence 链路可运行。

独立审计确认：该 v2 SQLite study 的协议记录没有当前代码后来新增的 `definition_identities`。因此，源码/配置/权重内容指纹修复目前只通过自动化测试，尚未用修复后的最终代码重新做真实 CUDA smoke；该 v2 study 用当前代码 resume 会因协议不一致而拒绝。不得把 v2 smoke 表述为已验证最终指纹逻辑。

更早的 `lcbam_optuna_smoke_20260901` 使用错误的相对输出路径，已判为 `invalid`。其 CSV、SQLite 和 `best_params.yaml` 仍保留原始 `COMPLETE` / best 字段，只代表训练进程当时完成，不改变研究有效性判定；已在对应产物目录补充 `INVALID_RUN.md`。任何汇总必须以 invalid 标记和本文件为准，不能把该 run 当作有效 smoke 或最佳参数证据。

## Confirmed

- LCBAMv2 已实现，并与 legacy LCBAM 分离。
- YOLOv8 注册、静态 YAML 和兼容预训练权重迁移已实现。
- P3 / P4 / P5 均已完成 build / forward 验证，索引为 5 / 7 / 10。
- Optuna/TPE 框架、validation-only objective、独立 trial 目录和持久化已实现；协议内容指纹已通过自动化测试，但尚未完成修复后真实 CUDA smoke。
- 最近一次已有证据记录完整 pytest 为 63 passed；本文件变更后的最终复跑结果以 `PROJECT_LOG.md` 最新条目为准。
- Baseline、LCBAMv2 和 Optuna 的最小 CUDA smoke 链路已有成功记录。
- 当前数据格式/目录验证已通过；阴性空标签是允许状态。

## Not Yet Confirmed

- 正式 baseline、CBAM、LCBAMv2 结果及三者公平比较。
- 当前版 CBAM 的实现、插入位置和正式配置。
- 正式 LCBAMv2 消融结果。
- 正式 Optuna 搜索、最佳结构及其是否优于默认结构。
- BO-LCBAMv2、LCBAMv2、CBAM 或 baseline 之间的性能优劣。
- 多 seed 稳定性结论与最终独立 test set 结果。
- 最终参数量、FLOPs、显存和 latency 对比。
- Baseline/CBAM/LCBAMv2 primary comparison 的最终训练参数冻结与云 GPU / 租用 GPU 执行计划（LCBAMv2 structure-search v1 参数已单独冻结）。
- `configs/experiment_protocol.yaml` 中 `lcbam_structure_search_v1` 的 structure-search 条件已经冻结；它不等同于尚待冻结的 Baseline/CBAM/LCBAMv2 primary-comparison 协议。

未来 Agent 不得根据项目目标、legacy 结果、README 计划或 smoke 指标把这些项目写成已完成。不得把本 RSNA / YOLOv8n 项目与 YOLO11、食堂或其他仓库内容混合。
