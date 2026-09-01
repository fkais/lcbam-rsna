# LCBAM-RSNA 项目持续记忆

本文件只记录项目实际上发生的事情：当前状态、重要历史、已确认结论、待验证假设、已否定方案、实验记录和下一步。`AGENTS.md` 负责规定“应该怎么开发”，本文件负责提供跨任务持续记忆，避免重复工作或把旧状态误当成当前状态。

## 当前状态（2026-09-01 19:26 +08:00）

- 当前阶段：独立 Optuna/TPE 结构搜索框架、测试和最小 CUDA smoke 已完成；正式贝叶斯优化尚未开始。
- LCBAMv2 模块已实现；YOLOv8 注册与集成、自定义模型 YAML 和兼容预训练权重迁移均已完成。结构搜索支持三个经 build/forward 验证的插入位置：P3/C2f 后、P4/C2f 后、P5/SPPF 后。
- LCBAMv2 的真实可搜索参数为 ECA 风格通道卷积核 `channel_kernel=[3,5,7]`、空间卷积核 `spatial_kernel=[3,5,7]` 和 `insert_position=[p3,p4,p5]`。模块没有 reduction bottleneck，因此未虚构 `reduction_ratio`；`dilation=3`、`gamma_init=0`、`num_lcbam=1` 固定。
- RSNA 固定数据路径已确认为 `E:/paperdata/meddet/datasets/rsna`，不重新随机划分；patient/content-level split audit 已完成并为 PASS，跨 split basename/patient/exact-content overlap 均为 0。
- RSNA 数据验证已重新执行并返回 `PASS`：train/val/test 均无孤立标签、无未标注的受支持图片、无无效标签行。
- Windows smoke training 入口默认使用 `workers=0`，并默认以 `fraction=0.02` 抽取 2% 训练数据；二者仅用于轻量链路验证，不是正式实验超参数。
- CUDA 环境已配置成功：PyTorch 2.12.1+cu132 可见 NVIDIA GeForce RTX 3050 Laptop GPU。
- Baseline CUDA smoke 已完成：`runs/detect/train-3`，1 epoch，`workers=0`、`fraction=0.02`，产出完整结果表和 checkpoint。
- LCBAMv2 CUDA smoke 已完成：`runs/detect/train-4`，1 epoch，`workers=0`、`fraction=0.02`，产出完整结果表和 checkpoint。
- 已建立 `PROJECT_CONTEXT.md` 和 `EXPERIMENT_PROTOCOL.md`，并在 `AGENTS.md` 固定新会话的上下文读取顺序；实验协议当前优先于 README 和未冻结机器配置。
- 完整 pytest 已重新执行：`63 passed in 7.39s`；上一轮源码/脚本覆盖率记录为 89%，本次未重跑 coverage。
- Optuna CUDA smoke 已真实完成：1 trial、1 epoch、`fraction=0.02`、`workers=0`，trial 0 状态 COMPLETE，使用 validation 指标；输出目录为 `runs/optuna/lcbam_optuna_smoke_20260901_v2/trial_000`。该 run 早于最终 `definition_identities` 指纹修复，只证明当时端到端搜索链路；最终指纹逻辑目前仅有自动化测试证据。
- `lcbam_structure_search_v1` 的 20-trial structure-only 协议已经冻结；formal preflight 对真实全量 SHA-256 manifest 返回 PASS，正式搜索仍为 **NOT STARTED**。
- 正式 Bayesian/TPE optimization、最佳参数确定、最优参数完整训练、多 seed 验证、test set 最终评估和效率 benchmark 均为 **NOT STARTED**。
- 下一步：在单独授权的任务中按冻结命令启动 20-trial 正式搜索；不得在 v1 中改训练参数/空间，也不得接触 test。CBAM / legacy LCBAM 核验与 primary comparison 冻结仍是后续独立工作。

## 重要历史

### 模型的长期角色

- **Baseline**：YOLOv8n，是所有注意力模块实验的正式对照组。
- **CBAM**：标准注意力对照组；实现、插入位置和公平对比配置仍需恢复并核验。
- **原始 LCBAM**：`legacy` 历史复现对象，保存在 `archive/legacy_experiments/`；必须保留且不得静默修改，历史结果不能未经复核直接作为当前正式结论。
- **LCBAMv2**：与原始 LCBAM 独立维护的当前主要候选模块；代码已进入仓库并完成 YOLOv8 代码级集成，但尚无正式训练结果。

### 持续有效的重要决策

- 新模块正式名称为 `LCBAMv2`，原始历史模块继续称为 `LCBAM`，避免新设计与 legacy 实现混淆。
- 自定义模块保存在仓库内，不手工修改 Python `site-packages`。
- 正式对比需覆盖 Baseline、CBAM、原始 LCBAM 和 LCBAMv2；配置变化必须记录时间和原因。
- 历史记录中的 epochs、batch size、image size 和 seed 仅作背景，不自动沿用为正式超参数。
- 实验状态使用 `planned`、`running`、`completed`、`failed` 或 `invalid`；配置错误或对比不公平的已完成实验应标为 `invalid`，不得作为正式结论。

## 已确认结论

- LCBAMv2 原默认配置位于 YOLOv8n backbone 的 P5/SPPF 后；结构搜索另提供 P3/C2f 后和 P4/C2f 后 YAML，三个位置均已完成注册、模型构建和完整 forward 验证。
- RSNA 真实本地路径为 `E:/paperdata/meddet/datasets/rsna`；当前 train/val/test 划分固定，不重新随机划分。
- Baseline 使用 Ultralytics 官方 YOLOv8n 预训练权重；LCBAMv2 会检测 P3/P4/P5 的真实注意力层索引，再对后续层偏移映射并迁移形状兼容权重。
- 正式训练、多随机种子实验、性能评估和效率 benchmark 均尚未开始，因此当前没有模型优劣或性能提升结论。

## 待验证假设

- CBAM 和原始 LCBAM 的历史实现、插入位置及复现实验配置仍需核验。
- Baseline、CBAM、原始 LCBAM 与 LCBAMv2 的正式公平对比配置仍需固定。
- LCBAMv2 的正式精度收益和效率开销仍待实测；CUDA smoke 指标不能回答这些问题。

## 已否定方案

- 不把 `configs/rsna.yaml` 当作模型结构 YAML 传给 `YOLO(...)`。
- 不按未偏移的同名 state-dict key 直接加载 LCBAMv2 权重；SPPF 后插层会使后续层索引整体偏移一位。
- 不重新随机划分当前 RSNA 数据，也不因一次验证自动删除孤立标签或清理 `.npy` 文件。
- 不把空标签默认判为错误；它可以表示无目标图像。

## 实验记录

### 2026-08-30 — LCBAMv2 的 YOLOv8 集成验证

#### 前置检查

- 本次开始时仓库根目录没有 `AGENTS.md`，Git `HEAD` 中也未跟踪该文件，因此没有可读取的仓库级代理约定。
- 本次开始时仓库根目录没有 `PROJECT_LOG.md`，本文件为首次创建。
- `configs/rsna.yaml` 是 RSNA 数据集配置，不是模型结构配置，不能且不应直接传给 `YOLO(...)` 构建模型。

#### 发现并修复的集成问题

- `LCBAMv2` 原先只存在于 `src/models/lcbam_v2.py` 和独立单元测试中。
- Ultralytics 的 YAML 解析命名空间中没有 `LCBAMv2`，仓库也没有当前版 LCBAMv2 的模型 YAML 或项目构建入口。
- 新增 `register_lcbamv2()`，在模型构建前将项目模块显式注册到 Ultralytics 解析命名空间。
- 新增 `build_lcbamv2_yolov8()`，统一执行注册并通过 Ultralytics `YOLO(...)` 构建模型。
- 新增 `configs/models/yolov8n-lcbamv2.yaml`，在 YOLOv8n backbone 的 SPPF 后接入一个 `LCBAMv2`；未修改 LCBAMv2 设计或正式实验超参数。

#### 代码已验证

- TDD RED：新增集成测试首次运行时，因 `src.models` 缺少 `build_lcbamv2_yolov8` 而按预期失败。
- TDD GREEN：目标集成测试 `2 passed`。
- 模型解析：构建后的网络中恰好包含一个 `LCBAMv2` 实例。
- 完整 forward：CPU 输入形状为 `(1, 3, 64, 64)`，主预测输出形状为 `(1, 5, 84)`，并返回三个检测尺度特征图。
- 完整测试：`python -m pytest -v` 最终复跑为 `6 passed in 2.06s`。
- 覆盖率：`python -m pytest --cov=src --cov-report=term-missing -q` 最终复跑为 `94%`、`6 passed in 2.98s`。
- 验证环境：Python 3.11.9、PyTorch 2.13.0+cpu、Ultralytics 8.4.135。

#### Smoke training 未执行

- `torch.cuda.is_available()` 为 `False`，CUDA 设备数为 `0`。
- `configs/rsna.yaml` 指向的 `E:/meddet/datasets/rsna` 不存在；预期的 images/labels train/val/test 子目录均不存在。
- 因同时缺少 GPU 与数据集，本次没有执行 smoke training，也没有生成或声称任何训练结果。

#### 正式实验尚未验证

- 尚未完成数据完整性检查与固定划分验证。
- 尚未验证预训练权重加载策略以及 Baseline / CBAM / LCBAMv2 三组配置的严格公平性。
- 尚未在 CUDA 环境执行 smoke training。
- 尚未执行任何正式长训练、多随机种子实验、性能评估或效率 benchmark。

结论：LCBAMv2 的代码级 YOLOv8 注册、YAML 解析、模型构建和 forward 链路已经验证；当前仍不具备开始正式实验的完整条件。

### 2026-08-30 22:30 +08:00 — RSNA 本地验证与 CUDA smoke training 准备

#### 任务与修改前状态

- `configs/rsna.yaml` 已由当前 `HEAD` 修正到真实路径，但尚无仓库级数据验证脚本和统一 smoke training 入口。
- 原有 LCBAMv2 构建链路可运行，但直接按同名 state-dict key 加载 YOLOv8n 权重只匹配 162/358 项；SPPF 后插入新层使后续层索引整体偏移一位。
- 本地环境为 PyTorch 2.13.0+cpu，`torch.cuda.is_available()` 为 `False`。

#### 修改内容和文件

- 核对并保持 `configs/rsna.yaml` 的 `train`、`val`、`test`、`nc`、`names` 语义不变；未重新划分数据。
- 新增 `scripts/validate_rsna.py`：检查目录、文件数量、Ultralytics 支持的图像、孤立标签、空标签、YOLO 行格式、类别范围、有限数、坐标范围和类别分布。没有 txt 的可训练图片只计为背景候选，不作为错误。
- 新增 `scripts/train_smoke.py`：统一支持 `baseline`、`LCBAMv2`、data YAML、device、epochs、batch、imgsz、seed 和预训练权重参数。
- 新增 LCBAMv2 兼容权重迁移：SPPF 后的源层索引加一后按名称和形状迁移；不修改 LCBAMv2 结构或插入位置。
- 新增临时小数据测试和 TDD 证据；补充 PyYAML 直接依赖及 coverage 忽略项。

#### 本地数据实际结果

固定划分目录文件数（与维护者提供的数量一致）：

| split | images 目录文件 | labels txt |
|---|---:|---:|
| train | 37287 | 18678 |
| val | 10672 | 5336 |
| test | 2670 | 2670 |

- Ultralytics 8.4.135 支持的图片为 train 18677、val 5336、test 2670；其余 train 18610、val 5336 个文件均为 `.npy`，不会被当前 Ultralytics 当作训练图片。
- 可训练图片中缺少 txt 的数量为 train/val/test 均 0。原则保持不变：未来发现的无标签图片目前只视为背景候选，仍需验证，不默认判错。
- 空标签：train 14480、val 4129、test 2063；空标签本身不判错。
- 标签格式无效行：0；有效框类别分布：class 0 共 9555 个框。
- 唯一孤立标签：`labels/train/6227d915-6ca0-430d-8905-34a61e29e074.txt`；文件为 0 字节且没有对应的受支持图片。未擅自修改数据，验证命令因此返回 1。
- 数据划分固定，不重新随机划分。

#### 预训练权重机制

- Baseline：直接用 `YOLO("yolov8n.pt")`，由 Ultralytics 加载官方 YOLOv8n 预训练权重并在训练数据 `nc=1` 时适配任务头。
- LCBAMv2：先从自定义 YAML 构建 `nc=1` 模型，再读取 `yolov8n.pt`，将源模型 index 0～9 原样迁移、index 10 及之后映射到目标 index+1，并仅加载形状兼容参数。
- 结构兼容检查迁移 319/358 个 state-dict 项。未匹配项为 3 个新增 LCBAMv2 参数和 36 个 COCO 80 类与 RSNA 1 类形状不兼容的分类头参数。
- LCBAMv2 新参数保持模块自身初始化：`channel_conv.weight`、`spatial_conv.weight` 使用 PyTorch Conv 默认 Kaiming-uniform 初始化，`gamma` 明确初始化为 0；没有伪造或复制不兼容权重。

#### 测试和检查

- TDD RED：目标测试首次收集因缺少 `scripts.validate_rsna`、`scripts.train_smoke` 和迁移函数而失败；孤立标签路径测试也先按预期失败。
- TDD GREEN：目标测试最终 `4 passed`。
- 覆盖率：首次多模块精确 source 参数组合在 Windows/Torch 加载时发生一次 access violation；改用稳定命令 `python -m pytest --cov=scripts --cov=src --cov-report=term-missing -q` 后为 `12 passed`、总覆盖率 90%。
- 用户要求的完整命令 `python -m pytest -v` 在项目虚拟环境中最终为 `12 passed in 2.31s`。
- 未启动 smoke training 或正式长训练，也未生成性能结论。

#### 决定与风险

- 不删除孤立标签、不清理 `.npy`、不修改固定划分；由维护者确认数据来源后再处理。
- 当前仓库代码已具备 smoke 命令，但这台环境缺少 CUDA 版 PyTorch/GPU 可见性；本地还需可用 CUDA 环境以及 `yolov8n.pt` 本地文件或首次下载网络。

### 2026-08-30 — 合并旧持续记忆

#### 任务与目标

- 以 D 盘当前日志为唯一正式主版本，提取 E 盘旧日志中仍有效且主版本缺失的长期项目记忆。

#### 修改前状态与修改内容

- D 盘已记录最新的 LCBAMv2 集成、本地数据路径、固定划分和 smoke training 准备状态，但缺少持续记忆定位、四类模型长期角色及部分治理决策。
- 补充上述长期内容并按本文件限定的七类信息重组；保留 D 盘全部当前状态和验证结果。
- E 盘关于“等待源码推送”“路径待定”“插入位置待定”和“远程仓库没有 LCBAMv2”的旧状态未合入。
- 本次仅修改文档，没有修改模型代码、测试代码、配置或实验超参数。

#### 检查与结果

- 比较了 D/E 两份 `PROJECT_LOG.md` 和 `AGENTS.md`；两份 `AGENTS.md` 内容一致，因此未修改。
- 使用 `git diff -- PROJECT_LOG.md AGENTS.md` 和 `git status` 检查最终工作区；合并结果以对应命令输出为准。

### 2026-08-30 23:48 +08:00 — Windows smoke training worker 默认值

#### 任务与目标

- Windows 下 smoke training 曾因 multiprocessing 子进程与 pagefile 资源压力触发 `WinError 1455`。
- 目标是在不修改训练逻辑、模型结构或其他实验超参数的前提下，让 smoke training 默认禁用多进程数据加载，同时保留命令行覆盖能力。

#### 修改前状态与修改内容

- 修改前 `scripts/train_smoke.py` 未暴露 Ultralytics 的 `workers` 参数，训练调用使用其隐式默认值。
- 在 `scripts/train_smoke.py` 增加 `--workers`，类型为 `int`、默认值为 `0`，并通过 `workers=args.workers` 传给 `model.train(...)`。
- 在 `tests/test_train_smoke.py` 增加默认值测试，并扩展参数转发测试以验证显式 `--workers 2` 原样传递。
- 未修改其他训练逻辑、模型结构或实验超参数；未执行 smoke training 或正式训练。

#### 测试和检查

- TDD RED：`.venv\\Scripts\\python.exe -m pytest tests/test_train_smoke.py -v` 首次为 `2 failed, 2 passed`；失败分别证明默认调用未传 `workers`，且 argparse 尚不识别 `--workers`。
- TDD GREEN：同一目标命令实现后为 `4 passed in 2.34s`。
- 完整测试：在项目虚拟环境执行 `python -m pytest -v`，结果为 `13 passed in 4.26s`。
- 系统 Python 不含 pytest；直接调用系统解释器返回 `No module named pytest`，因此所有有效测试结果均来自项目 `.venv`。

#### 决定、风险与下一步

- Windows smoke training 默认固定为 `workers=0`，优先降低 multiprocessing 与 pagefile 压力；需要并行加载时由操作者显式传入更高值。
- `workers=0` 可能降低数据加载吞吐，但仅改变 smoke training 入口的数据加载进程数，不改变模型或实验超参数。
- 下一步在可用 CUDA 环境以默认 `workers=0` 重试 1～3 epoch smoke training；通过前不得开始正式长训练。

### 2026-08-30 23:55 +08:00 — Smoke training 默认抽取 2% 训练数据

#### 任务与目标

- Smoke test 只需在真实数据上验证 CUDA forward/backward、optimizer 和 checkpoint 链路，不应默认遍历完整 RSNA train split。
- 目标是在保留 `workers=0` 默认值的同时，为 smoke 入口增加可覆盖的数据抽样比例，缩短轻量验证时长。

#### 修改前状态与修改内容

- 修改前 `scripts/train_smoke.py` 会把整个训练集交给 Ultralytics，smoke 运行时长与其验证目标不相称。
- 在 `scripts/train_smoke.py` 增加 `--fraction`，类型为 `float`、默认值为 `0.02`，并通过 `fraction=args.fraction` 传给 `model.train(...)`。
- 在 `tests/test_train_smoke.py` 验证默认 `fraction=0.02`、默认 `workers=0` 保持不变，以及显式 `--fraction 0.15` 原样传递。
- 本次只修改 smoke training 入口；未修改正式训练入口、正式实验超参数、模型结构或其他训练逻辑，未启动任何训练。

#### 测试和检查

- TDD RED：`.venv\\Scripts\\python.exe -m pytest tests/test_train_smoke.py -v` 首次为 `2 failed, 2 passed`；失败分别证明默认调用未传 `fraction`，且 argparse 尚不识别 `--fraction`。
- TDD GREEN：同一目标命令实现后为 `4 passed in 2.44s`。
- 完整测试：在项目虚拟环境执行 `python -m pytest -v`，结果为 `13 passed in 4.45s`。
- 覆盖率：`python -m pytest --cov=scripts --cov=src --cov-report=term-missing -q` 结果为 `13 passed`、总覆盖率 91%。

#### 决定、风险与下一步

- Smoke training 默认只抽取 RSNA train split 的 2%，用途限定为快速验证端到端训练链路；可通过 `--fraction` 显式覆盖。
- 正式训练仍使用完整训练数据，不继承 smoke 入口的 `fraction=0.02`，正式实验超参数保持不变。
- 2% 子集不用于报告精度、比较模型或形成实验结论；下一步在可用 CUDA 环境执行 smoke test，确认链路后再单独准备正式训练。

### 2026-08-31 00:35 +08:00 — 当前阶段收尾与 Git 提交准备

#### 任务与修改前状态

- 目标仅为复核当前阶段、清理中断 smoke 产物、更新持续记忆并准备阶段性提交；不修改 LCBAMv2 结构，不调整正式实验超参数，不启动正式训练。
- `main` 相对 `origin/main` 已有 4 个本地 commit，工作区另有数据验证、smoke 入口、测试和文档改动尚未提交。
- `runs/detect/train`、`train-3`、`train-4` 同时存在，需要依据实际产物区分中断与成功运行。

#### 训练产物清理

- `runs/detect/train` 只有 `args.yaml` 和空 `weights/`，没有 `results.csv`、训练图表或 checkpoint；其参数仍为 `workers=8`、`fraction=1.0`，确认是此前中断产物后删除。
- `runs/detect/train-3` 确认为 Baseline CUDA smoke：1 epoch、`workers=0`、`fraction=0.02`，包含单行 `results.csv`、完整验证图表以及 `best.pt`、`last.pt`，予以保留。
- `runs/detect/train-4` 确认为 LCBAMv2 CUDA smoke：1 epoch、`workers=0`、`fraction=0.02`，包含单行 `results.csv`、完整验证图表以及 `best.pt`、`last.pt`，予以保留。
- 所有 `runs/` 和 `*.pt` 均由 `.gitignore` 排除，不进入提交。

#### 分层验证结果

**代码级验证**

- LCBAMv2 模块、YOLOv8 集成、自定义模型 YAML 和预训练权重迁移代码均已纳入本阶段提交范围；本次未修改注意力模块设计。
- 项目虚拟环境执行 `.venv\\Scripts\\python.exe -m pytest -v`，结果为 `13 passed in 5.00s`。

**数据级验证**

- 执行 `.venv\\Scripts\\python.exe scripts\\validate_rsna.py --data configs\\rsna.yaml`，结果为 `PASS`。
- 固定路径为 `E:/paperdata/meddet/datasets/rsna`；train/val/test 的受支持图片分别为 18677/5336/2670，孤立标签均为 0，无效标签行均为 0。
- 此前唯一孤立的 0 字节空标签已处理；train 空标签数由 14480 变为 14479。空标签可表示无目标样本，本身不视为错误。

**CUDA smoke 验证**

- PyTorch 2.12.1+cu132 检测到 1 张 NVIDIA GeForce RTX 3050 Laptop GPU。
- Baseline 和 LCBAMv2 均已完成 CUDA smoke，并证明真实数据上的 forward/backward、optimizer、验证和 checkpoint 链路可运行。
- smoke 使用 2% 数据和 1 epoch，只是链路检查；不得根据其 precision、recall 或 mAP 判断模型优劣。

**正式实验状态**

- 正式长训练、多随机种子实验、四模型公平比较、正式性能评估和效率 benchmark 均未开始（**NOT STARTED**）。
- 当前已具备进入正式实验配置冻结阶段的工程条件；仍需先核验 CBAM/原始 LCBAM 并固定四组公平对比配置，之后才能启动正式训练。

#### Git 与忽略规则

- `.gitignore` 已覆盖 `.venv/`、`runs/`、`data/`、`datasets/`、`*.pt`、`*.cache`、`__pycache__/`、`.pytest_cache/`、IDE 文件、coverage 产物和常见临时文件。
- 阶段性提交计划使用消息 `feat: integrate LCBAMv2 and validate RSNA CUDA smoke pipeline`；提交后只执行普通 `git push`，不强推、不重写历史。

#### 决定与风险

- 保留成功 smoke 的本地产物作为链路证据，但不提交生成文件或权重。
- 本次清理不会改变源码、数据集、配置、测试或成功训练结果。
- 当前 smoke 成功不等同于正式实验完成，也不构成 LCBAMv2 优于 Baseline 的证据。

### 2026-09-01 14:08 +08:00 — Optuna/TPE LCBAMv2 结构搜索框架

#### 任务、范围与改造前状态

- 目标是在不破坏普通 LCBAMv2 + YOLOv8n 训练链路的前提下，新增独立 Optuna/TPE 结构搜索、SQLite 持久化、resume、trial 隔离、CSV/YAML 摘要、测试和最小 smoke。
- 本次读取并遵守根目录 `AGENTS.md` 和本日志；主 Agent 检查真实仓库后，使用 Architecture、Implementation、Audit 三个子 Agent 分阶段分析、实现和独立审计，最终代码、测试和 smoke 均由主 Agent 复核。
- 改造前 LCBAMv2 构造参数为 `kernel_size`、`dilation`、`gamma_init`；模块是 ECA 风格通道 Conv1d，没有 reduction bottleneck。已有模型只在 P5/SPPF 后插入一个模块，Optuna 未安装，完整基线测试为 13 passed。
- 本次未修改数据划分，未开始正式 20～30 trial 搜索或长训练。

#### 设计和实现

- 在 `LCBAMv2` 构造函数末尾新增 `spatial_kernel`，默认值 3 保持原行为；只接受正奇数，并以 `dilation * (kernel // 2)` 保持空间尺寸。核心通道/空间/残差门控设计未改变。
- 新增 P3 与 P4 静态 YAML；P3/P4/P5 的注意力层实际索引分别为 5/7/10。Optuna trial 只复制对应静态 YAML 到自身目录并替换真实 kernel 参数，没有引入动态插件系统或大规模 Ultralytics 重构。
- 新增 `src/optimize_lcbam.py` 独立入口，使用 seeded `TPESampler` 和 `direction=maximize`。搜索空间为 `channel_kernel=[3,5,7]`、`spatial_kernel=[3,5,7]`、`insert_position=[p3,p4,p5]`；`dilation=3`、`gamma_init=0`、`num_lcbam=1` 固定。
- objective 显式调用 `model.train(..., val=True, split="val")`，只从返回的 validation `results_dict` 读取 precision、recall、mAP50、mAP50-95，并以 mAP50-95 作为返回值。没有调用 test split，也没有读取 test 指标。
- 数据守卫要求 train/val/test 三个非空且规范化后的路径互异；该守卫用于阻止明显 split alias，但不替代正式实验前的 patient/content-level manifest 审计。
- 每个 trial 原子占用 `runs/optuna/<study>/trial_NNN`，目录已存在时直接失败，Ultralytics `project` 使用绝对路径。SQLite study 使用 `load_if_exists=True`；恢复时核对搜索空间、训练条件、依赖版本，以及 data/model YAML、关键源码和权重的 SHA-256，设置或实现内容不一致则拒绝续跑。预训练权重必须在 study 创建前已存在于本地。
- 每个 trial 记录参数、四项 validation 指标、实际 output_dir、state 和 error；每次 trial 后刷新 `trials.csv`，存在 COMPLETE trial 时刷新 `best_params.yaml`。大型 checkpoint 仍由 `.gitignore` 排除。
- Python、NumPy、PyTorch CPU/CUDA、cuDNN deterministic/benchmark、Ultralytics training seed 和 Optuna sampler seed 均显式处理。不同 trial 固定相同训练 seed；`n_jobs=1` 避免单 GPU 并发造成不可比较。
- pruning 未接入：当前 Ultralytics 链路低成本只能稳定取得训练完成后的 validation metrics，为避免扩大改造范围，本阶段不实现逐 epoch pruning。

#### 审计发现和修正

- 独立审计发现：只在 `model.train(pretrained=False)` 前手工迁移权重会使 Ultralytics Trainer 从 YAML 重建模型，从而丢弃已经初始化和迁移的自定义模型。这是既有 smoke 入口也受影响的真实 trainer-boundary 问题。
- 新增 `preserve_lcbamv2_model_for_training()`，让当前 Ultralytics 8.4.135 Trainer 使用已经构建/迁移的自定义模型；普通 `scripts/train_smoke.py` 和 Optuna 共用该边界修正，并增加回归测试。
- 首次探索性 Optuna smoke 还暴露相对 `project` 会被 Ultralytics 嵌套写入 `runs/detect/runs/optuna/...`。该次运行按 trial 隔离要求标为 **invalid**，不作为最终 smoke；代码改为绝对 project，并添加先 RED 后 GREEN 的路径测试。
- 最终审计指出仅绑定路径会允许同路径 YAML/源码被修改后继续旧 study；新增内容指纹测试先复现该问题，再将 data/model YAML、关键源码和预训练权重 SHA-256 纳入协议。缺失本地权重会在 study 创建前 fail-fast，避免自动下载前后的 identity 漂移。
- 异常策略只把 CUDA OOM 和明确的不可比较训练条件包装为单 trial 失败；模型构建、YAML、数据路径和缺失指标错误不被宽泛 try/except 吞掉。自动 batch 变更会使 trial 失败。

#### 主 Agent 最终验证

- `.venv\Scripts\python.exe -m pytest -v`：`63 passed in 7.71s`。
- `.venv\Scripts\python.exe -m pytest --cov=scripts --cov=src --cov-report=term-missing -q`：`63 passed in 9.61s`，总覆盖率 89%，`src/optimize_lcbam.py` 覆盖率 88%。
- `.venv\Scripts\python.exe -m src.optimize_lcbam --help`：退出码 0，所需 CLI 参数和 `--n-trials` 别名可见。
- P3/P4/P5 独立 build/forward：注意力层索引分别为 5/7/10，输入 `(1,3,64,64)` 的 eval prediction 均为 `(1,5,84)`。
- 最终真实 CUDA smoke：study `lcbam_optuna_smoke_20260901_v2`，1 trial、1 epoch、imgsz 512、batch 4、workers 0、seed/sampler_seed 0、fraction 0.02，RTX 3050 Laptop GPU，退出码 0。
- 该 smoke 的 trial 0 采样 `channel_kernel=5`、`spatial_kernel=7`、`insert_position=p5`，validation mAP50-95 为 `0.00011168922747715054`，状态 COMPLETE；真实输出、CSV 中 output_dir 和 `args.yaml` project/name 一致。这个数值仅为链路证据，不构成最佳参数或性能结论。
- smoke 只使用 validation 作为 objective；没有运行 test set 最终评估。

#### 尚未完成

- 正式 Bayesian/TPE optimization：**NOT STARTED**。
- 最佳参数确定与最优结构完整训练：**NOT STARTED**。
- 多 seed 验证：**NOT STARTED**。
- test set 最终统一评估：**NOT STARTED**。
- 正式搜索前仍需冻结 epochs、batch、optimizer/augmentation 等全部非搜索条件，并记录 patient/content-level split manifest 审计证据。

### 2026-09-01 18:49 +08:00 — 长期上下文与实验协议基线

#### 做了什么与为什么

- 为降低新会话、Agent 切换和租用 GPU 后的事实漂移，新增 `PROJECT_CONTEXT.md` 和 `EXPERIMENT_PROTOCOL.md`，分别承载已核验事实与正式实验规则。
- 在 `AGENTS.md` 加入 `AGENTS.md -> PROJECT_CONTEXT.md -> EXPERIMENT_PROTOCOL.md -> PROJECT_LOG.md` 的固定读取顺序；未把 AI 上下文规则大规模塞入 README。
- 最小更新 README、`configs/experiment_protocol.yaml` 和历史 testing evidence：删除参数量优于 CBAM 的未验证断言，取消未冻结的 100 epoch / seed 表，明确机器配置仍为 draft，并标注旧 evidence 的时间快照属性。
- 实际调用一个独立只读 Document / Experiment Audit 子 Agent；主 Agent 复核其发现并完成修订，没有调用 Implementation / Architecture 子 Agent。

#### 核实与纠正

- 活跃仓库确认为 `D:/project/lcbam-rsna`；`E:/paperdata/lcbam-rsna` 是更早副本，数据集仍位于 `E:/paperdata/meddet/datasets/rsna`。
- 实际 `labels/train` 为 18,677，不是任务背景中的 18,678。images 目录总数较大主要来自 `.npy` 辅助文件；当前 `.png` 与 `.txt` 数量逐 split 相等，阴性样本由合法空标签表示。
- LCBAMv2 构造参数为 `kernel_size/dilation/gamma_init/spatial_kernel`；Optuna 字段 `channel_kernel` 映射到 `kernel_size`，没有 `reduction_ratio`。P3/P4/P5 索引 5/7/10 与源码、YAML 和测试一致。
- 审计发现 v2 CUDA smoke 的 SQLite 协议没有后来加入的 `definition_identities`。因此纠正此前可能造成的顺序暗示：v2 只验证较早链路，最终内容指纹修复仅经自动化测试，尚未重新做真实 GPU smoke。
- 更早的相对输出路径 smoke 虽然机器字段仍为 COMPLETE/best，但研究有效性为 invalid；未改写原始产物，只在 result/run 目录增加 `INVALID_RUN.md` 并在长期文档中明确禁用。
- primary comparison 定为 baseline / CBAM / LCBAMv2；legacy LCBAM 是必须核验的历史复现轨道，只有经协议修订后才可升级为同一主结果表的第四组。BO-LCBAMv2 是结构优化组，不是新模块。

#### 验证与尚未完成

- `.venv\Scripts\python.exe -m pytest -v`：最终复跑 `63 passed in 7.39s`。
- `git diff --check` 初稿检查无空白错误，仅有仓库正常的 LF-to-CRLF 提示；最终检查结果见本次交付汇报。
- 未启动训练、正式 Optuna、正式 baseline，未修改 LCBAMv2 核心设计、数据划分或正式超参数。
- 尚未完成 patient/content-level leakage 审计、正式训练参数冻结、CBAM/legacy 实现核验、最终指纹版本 GPU smoke、租用 GPU 计划、正式多 seed 与 test 评估。

## 下一步

1. 核验 CBAM 和原始 LCBAM 的实现、插入位置及复现配置。
2. 冻结 Optuna 结构搜索的全部非搜索条件，同时固定 Baseline、CBAM、LCBAMv2 三组 primary comparison；单独核验原始 LCBAM 历史复现轨道。
3. 配置冻结和 split manifest 复核通过后，再单独启动正式 20～30 trial 搜索；当前任务未启动。
4. 选择正式 study 的最佳 trial 后再做完整训练、多 seed 验证，并只在所有模型选择完成后统一评估 test set。

### 2026-09-01 19:26 +08:00 — LCBAMv2 structure search v1 最终冻结

#### 任务、修改前状态与多 Agent 审计

- 目标是在不启动正式搜索、不使用 test 选择模型的前提下，完成正式 Optuna structure-search 的训练条件冻结、RSNA patient/content manifest、运行保护、测试和协议文档。
- 用户要求的文件实际位于活跃仓库 `D:/project/lcbam-rsna`；`E:/paperdata/lcbam-rsna` 是缺少当前 `src/configs/tests` 的旧 checkout。三名只读子 Agent 分别完成 Experiment Design、Data Audit 和独立科研审计，主 Agent 统一实现与验证。
- 修改前 objective 仍使用 `optimizer="auto"` 和大量 Ultralytics 隐式默认；`--trials` resume 会每次追加预算；正式入口不要求 patient/content audit；output/results/storage 与完整训练条件未全部进入 study 指纹。

#### 冻结决定与实现

- v1 只搜索 `channel_kernel=[3,5,7]`、`spatial_kernel=[3,5,7]`、`insert_position=[p3,p4,p5]`；固定 `dilation=3`、`gamma_init=0`、`num_lcbam=1`，不引入不存在的 `reduction_ratio`。
- 冻结 20 attempted trials、20 epochs、imgsz 512、batch 4、device 0、workers 0、fraction 1.0、seed/sampler seed 0、deterministic true、AMP true、cache false、patience 0。
- 根据本机 Ultralytics 8.4.135 的真实 `optimizer=auto` 决策，改为显式 AdamW：lr0 0.002、lrf 0.01、beta1/momentum 0.9、weight decay 0.0005；显式冻结 warmup、loss、nbs 和全部 augmentation 参数，消除未来默认漂移。
- 本地 `yolov8n.pt` SHA-256 为 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`。P3/P4/P5 正式迁移必须精确为 319/358 项和 3 个新 LCBAMv2 参数项。
- 增加 formal precondition：要求 PASS audit、data YAML/manifest identity 与当前数据逐文件 path/size/SHA-256 一致；formal 与 smoke 名称分离；stale RUNNING trial fail closed；`--trials 20` 改为整个 study 的总预算。
- SQLite storage、绝对 output/results root、sampler 完整设置、formal 类型、trial budget、全部训练条件、源码/YAML/权重/audit identity 均进入 study protocol。trial build/train/metric 异常会持久化到 trial error。

#### RSNA 真实 split audit

- 新增 `scripts/audit_rsna_split.py`，生成本地忽略的 `results/data_audits/rsna_split_audit_v1.json` 和 26,683 行全量 SHA-256 manifest。
- train/val/test PNG 数为 18,677/5,336/2,670；positive/nonempty 为 4,198/1,207/607；legal negative/empty 为 14,479/4,129/2,063；当前 no-txt background 均为 0。缺 txt 的图片在工具和测试中只计为合法背景候选，不判错。
- 所有当前 PNG basename 均能映射原始 metadata `patientId`；跨 split basename/patient/exact-content overlap 均为 0，结果 PASS。
- train 内存在一个 exact duplicate group（2 个阴性 patientId，1 个 excess），不构成跨 split leakage，本任务未改划分。原始 metadata 的 26,684 个 patientId 中，已知阴性 `6227d915-6ca0-430d-8905-34a61e29e074` 不在当前 26,683 个冻结转换样本中；不重新加入或随机划分。

#### TDD、检查与实际结果

- 初始 RED：目标测试因缺少 `scripts.audit_rsna_split` 收集失败；首次真实 CLI 还暴露 direct-script import 失败，随后做最小兼容修复。
- 新增 overlap/alias/negative/artifact/intra-split duplicate 测试，补 formal current-manifest guard、显式 objective 参数和 resume total-budget 测试；P3/P4/P5 与原 smoke 回归继续通过。
- 预最终 `.venv\Scripts\python.exe -m pytest -v`：73 passed in 9.03s。
- coverage：73 passed，TOTAL 86%，`scripts/audit_rsna_split.py` 82%，`src/optimize_lcbam.py` 86%。
- 真实全量数据 audit：PASS。只执行 formal preflight、未 create study、未运行 trial：`FORMAL_PREFLIGHT=PASS`，protocol SHA-256 `f2e3afb93422ca821206379d7b57897120af6f8ee86cd34d7e3efe8370acc611`。
- 最终 `.venv\Scripts\python.exe -m pytest -v`：74 passed in 9.31s；`git diff --check/stat/status` 结果以本次最终交付汇报为准。

#### 文档、限制与下一步

- 新增 `docs/experiments/lcbam_structure_search_v1.md` 和本轮 TDD evidence；同步 README、PROJECT_CONTEXT、EXPERIMENT_PROTOCOL 与机器可读 config。
- Test 在 architecture/hyperparameter search 中保持 untouched；只有最终结构和训练协议全部选择、冻结并完成后，才允许单次最终 test evaluation。
- TPE 允许重复离散结构；报告必须区分 attempted trials 和 unique structures。SQLite 不持久化 sampler RNG state，因此跨进程 resume 不声称与 uninterrupted suggestion sequence 完全相同，restart boundary 必须记录。
- 本次没有启动正式 Optuna、长训练或 test evaluation，没有产生性能或最佳结构结论。下一步只能在单独授权任务中执行冻结的 20-trial 命令。
