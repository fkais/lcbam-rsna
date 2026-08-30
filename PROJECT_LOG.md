# LCBAM-RSNA 项目持续记忆

本文件只记录项目实际上发生的事情：当前状态、重要历史、已确认结论、待验证假设、已否定方案、实验记录和下一步。`AGENTS.md` 负责规定“应该怎么开发”，本文件负责提供跨任务持续记忆，避免重复工作或把旧状态误当成当前状态。

## 当前状态（2026-08-31 00:35 +08:00）

- 当前阶段：正式实验前的代码、数据与 CUDA smoke 验证均已完成，正在进行阶段性 Git 收尾。
- LCBAMv2 模块已实现；YOLOv8 注册与集成、自定义模型 YAML 和兼容预训练权重迁移均已完成。插入位置仍为 backbone SPPF 后，本阶段未修改 LCBAMv2 结构。
- RSNA 固定数据路径已确认为 `E:/paperdata/meddet/datasets/rsna`，不重新随机划分；唯一孤立的 0 字节空标签已处理。
- RSNA 数据验证已重新执行并返回 `PASS`：train/val/test 均无孤立标签、无未标注的受支持图片、无无效标签行。
- Windows smoke training 入口默认使用 `workers=0`，并默认以 `fraction=0.02` 抽取 2% 训练数据；二者仅用于轻量链路验证，不是正式实验超参数。
- CUDA 环境已配置成功：PyTorch 2.12.1+cu132 可见 NVIDIA GeForce RTX 3050 Laptop GPU。
- Baseline CUDA smoke 已完成：`runs/detect/train-3`，1 epoch，`workers=0`、`fraction=0.02`，产出完整结果表和 checkpoint。
- LCBAMv2 CUDA smoke 已完成：`runs/detect/train-4`，1 epoch，`workers=0`、`fraction=0.02`，产出完整结果表和 checkpoint。
- 完整 pytest 已重新执行：`13 passed in 5.00s`。
- 正式训练、多随机种子对比、性能评估和效率 benchmark 均为 **NOT STARTED**。Smoke training 指标只证明训练链路可运行，不得用于判断模型优劣。
- 下一步：固定四组模型的正式公平对比配置后，才可开始正式实验。

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

- LCBAMv2 已完成 Ultralytics 注册、模型 YAML、项目构建入口和完整 forward 验证，插入位置为 YOLOv8n backbone 的 SPPF 后。
- RSNA 真实本地路径为 `E:/paperdata/meddet/datasets/rsna`；当前 train/val/test 划分固定，不重新随机划分。
- Baseline 使用 Ultralytics 官方 YOLOv8n 预训练权重；LCBAMv2 通过 SPPF 后层索引偏移映射迁移形状兼容权重。
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

## 下一步

1. 核验 CBAM 和原始 LCBAM 的实现、插入位置及复现配置。
2. 固定 Baseline、CBAM、原始 LCBAM 与 LCBAMv2 的正式公平对比配置。
3. 配置冻结并完成训练前复核后，再单独启动正式长训练；当前任务不得启动。
