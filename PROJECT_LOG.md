# PROJECT_LOG

## 2026-08-30 — LCBAMv2 的 YOLOv8 集成验证

### 前置检查

- 本次开始时仓库根目录没有 `AGENTS.md`，Git `HEAD` 中也未跟踪该文件，因此没有可读取的仓库级代理约定。
- 本次开始时仓库根目录没有 `PROJECT_LOG.md`，本文件为首次创建。
- `configs/rsna.yaml` 是 RSNA 数据集配置，不是模型结构配置，不能且不应直接传给 `YOLO(...)` 构建模型。

### 发现并修复的集成问题

- `LCBAMv2` 原先只存在于 `src/models/lcbam_v2.py` 和独立单元测试中。
- Ultralytics 的 YAML 解析命名空间中没有 `LCBAMv2`，仓库也没有当前版 LCBAMv2 的模型 YAML 或项目构建入口。
- 新增 `register_lcbamv2()`，在模型构建前将项目模块显式注册到 Ultralytics 解析命名空间。
- 新增 `build_lcbamv2_yolov8()`，统一执行注册并通过 Ultralytics `YOLO(...)` 构建模型。
- 新增 `configs/models/yolov8n-lcbamv2.yaml`，在 YOLOv8n backbone 的 SPPF 后接入一个 `LCBAMv2`；未修改 LCBAMv2 设计或正式实验超参数。

### 代码已验证

- TDD RED：新增集成测试首次运行时，因 `src.models` 缺少 `build_lcbamv2_yolov8` 而按预期失败。
- TDD GREEN：目标集成测试 `2 passed`。
- 模型解析：构建后的网络中恰好包含一个 `LCBAMv2` 实例。
- 完整 forward：CPU 输入形状为 `(1, 3, 64, 64)`，主预测输出形状为 `(1, 5, 84)`，并返回三个检测尺度特征图。
- 完整测试：`python -m pytest -v` 最终复跑为 `6 passed in 2.06s`。
- 覆盖率：`python -m pytest --cov=src --cov-report=term-missing -q` 最终复跑为 `94%`、`6 passed in 2.98s`。
- 验证环境：Python 3.11.9、PyTorch 2.13.0+cpu、Ultralytics 8.4.135。

### Smoke training 未执行

- `torch.cuda.is_available()` 为 `False`，CUDA 设备数为 `0`。
- `configs/rsna.yaml` 指向的 `E:/meddet/datasets/rsna` 不存在；预期的 images/labels train/val/test 子目录均不存在。
- 因同时缺少 GPU 与数据集，本次没有执行 smoke training，也没有生成或声称任何训练结果。

### 正式实验尚未验证

- 尚未完成数据完整性检查与固定划分验证。
- 尚未验证预训练权重加载策略以及 Baseline / CBAM / LCBAMv2 三组配置的严格公平性。
- 尚未在 CUDA 环境执行 smoke training。
- 尚未执行任何正式长训练、多随机种子实验、性能评估或效率 benchmark。

结论：LCBAMv2 的代码级 YOLOv8 注册、YAML 解析、模型构建和 forward 链路已经验证；当前仍不具备开始正式实验的完整条件。
