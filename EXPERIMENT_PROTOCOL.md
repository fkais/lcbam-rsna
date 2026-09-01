# LCBAM-RSNA 实验协议

本文件只规定正式实验规则。除非维护者明确修订并在 `PROJECT_LOG.md` 记录原因，否则它是实验口径的最高优先级文档。源码决定实现事实，经过核验的 run artifacts / logs 决定实验事实；聊天记录、README 计划和 smoke 指标都不能替代它们。

## 1. Experiment Status

每次运行必须标记为 `debug`、`smoke`、`formal`、`failed` 或 `invalid`。只有遵守本协议且完成预定训练的 `formal` run 才能进入正式结果候选；只有经过重复性与最终 test 流程的结果才能支持性能结论。

## 2. Main Comparison

正式基础比较至少包括：

1. YOLOv8n baseline。
2. YOLOv8n + CBAM。
3. YOLOv8n + LCBAMv2 default。

BO-LCBAMv2 是 LCBAMv2 的结构优化组，不是独立注意力模块，应在基础公平比较之后报告。Legacy 原始 LCBAM 如被纳入，应作为单独的历史复现组，先核验其实现、位置和配置，不能与当前 LCBAMv2 混写。

原始 LCBAM 的历史复现是项目目标，但当前 primary comparison 固定为上述三组。原始 LCBAM 必须完成一次可复核的历史复现；在实现和配置核验前不进入 primary comparison。若后来把它放入同一主结果表，则必须遵守与三组相同的预算和公平性规则，并通过协议修订明确升级为第四个主组。

所有主比较组必须尽可能保持以下条件一致：

- 固定 dataset split；
- imgsz、batch、epochs；
- optimizer 与 learning-rate policy；
- augmentation；
- pretrained initialization policy；
- seed policy；
- early stopping / best-versus-last checkpoint policy；
- evaluation metrics 和阈值；
- hardware class、软件环境和资源预算（可行时）。

不同模型不得获得未披露的额外 epoch、trial、调参次数或其他明显不同的训练预算。若条件无法完全一致，必须事前定义差异并在报告中披露。

## 3. Phase-1 Optuna Structure Search

第一阶段只搜索：

- `channel_kernel`: `[3, 5, 7]`，映射到 LCBAMv2 构造参数 `kernel_size`；
- `spatial_kernel`: `[3, 5, 7]`；
- `insert_position`: `[p3, p4, p5]`。

正式 v1 条件已经冻结在 `docs/experiments/lcbam_structure_search_v1.md` 和 `configs/experiment_protocol.yaml`。该版本必须使用其中唯一的 dataset、split、20-trial 总预算、imgsz、batch、epochs、显式 AdamW 与 learning-rate policy、augmentation、training/sampler seed、pretrained identity、early-stopping policy、dilation=3、gamma_init=0.0、num_lcbam=1，以及软件/硬件环境；启动后不得就地修改。

第一阶段不得同时搜索 learning rate、momentum、weight decay、optimizer、batch 或大量 augmentation 参数。目标是尽量隔离 LCBAMv2 结构变化与训练超参数变化。当前代码中的 CLI 默认值、`optimizer="auto"` 和 Ultralytics defaults 只是已记录的实现状态；正式搜索启动前仍须明确接受或替换并冻结，不能从 smoke 自动继承。

仅对 LCBAMv2 做大量结构调参后，不得把 BO-LCBAMv2 与完全未调参的 baseline 描述为绝对公平的基础比较。应同时报告未优化 LCBAMv2 的基础比较，并将 BO-LCBAMv2 标为进一步优化。

## 4. Validation and Test Separation

Validation 可以用于 model selection、Optuna objective、结构/超参数选择和协议允许的 early stopping。

Test 只能在模型、训练方案、选择规则和报告指标全部冻结后，用于最终独立评估。禁止使用 test 进行：

- trial scoring 或 best-trial selection；
- 搜索空间或超参数调整；
- early model selection；
- 根据 test 表现返回修改模型或训练协议。

正确流程是：`train -> validation selection -> freeze model and protocol -> final test evaluation`。如果研究过程中查看 test 后又修改方案，必须记录为 protocol violation，或明确重定义并重新建立一个真正 untouched 的 final test；原 test 不得继续称为完全独立最终测试。

split 路径互异只防止明显路径别名，不证明患者或内容独立。正式 v1 前已完成 patient/content-level leakage 审计并保存全量 SHA-256 manifest / PASS 报告；正式入口会重新核对当前磁盘数据与该 manifest。

## 5. Smoke and Debug Runs

任何 1 epoch、fraction data、tiny subset、`smoke` 或 `debug` run 的指标都禁止进入：

- 正式论文/报告结果表；
- 模型优劣或泛化结论；
- README 性能宣传；
- BO 最佳结构结论。

它们只能证明指定工程链路是否成功，并必须带清晰的 smoke/debug 标记。Code validated、smoke validated、experiment validated 和 result validated 不得互相替代。

## 6. Recommended Execution Order

1. Phase 0：数据完整性与 patient/content-level split leakage 审计。
2. Phase 1：YOLOv8n baseline。
3. Phase 2：CBAM。
4. Phase 3：LCBAMv2 default。
5. Phase 4：LCBAMv2 ablation。
6. Phase 5：Optuna structure search。
7. Phase 6：best LCBAMv2 full training。
8. Phase 7：multi-seed stability。
9. Phase 8：final frozen-model test evaluation。
10. Phase 9：parameters / FLOPs / latency benchmark。
11. Phase 10：results aggregation / figures / report。

Phase 1～3 的基础比较配置必须先冻结。Phase 4～6 使用 validation 做选择；Phase 8 前不得使用 test 反馈修改前面阶段。

## 7. Reproducibility Record

每个 formal run 至少保存以下机器可读信息：

- Git commit 和 dirty-worktree 状态；
- model config 与 dataset config（最好附内容哈希）；
- seed / sampler seed；
- epochs、batch、imgsz、workers；
- optimizer、初始学习率和完整 learning-rate policy；
- augmentation 与 early stopping policy；
- pretrained initialization policy；
- device、GPU、hardware class；
- Python、Ultralytics、PyTorch、CUDA 版本；
- validation/test 指标定义；
- best.pt / last.pt 选择规则；
- 唯一 run directory 和 run name；
- 开始/结束时间、状态、失败原因（如适用）。

不得只依赖聊天记录恢复实验。正式配置变更必须产生新配置快照或新 study / run identity。

## 8. Artifact Integrity

- 每个正式实验和 study 必须使用唯一、可追踪的名称。
- 不得覆盖不同 run / study 的原始训练目录、trial 记录、数据库或 checkpoint。同一 study 续跑时，允许按设计刷新其派生 `trials.csv` 和 `best_params.yaml`，但必须可从不可变 trial / SQLite 记录重建，且不得用它覆盖另一 study 的摘要。
- 成功、失败、取消和 invalid trial 必须可区分；失败原因应持久化。
- 恢复 study 前必须核对搜索空间、总 trial 预算、formal/smoke 类型、固定条件、output / results roots、storage、split audit/manifest、源码/配置/权重身份；协议变化应使用新的 study 名称和数据库。正式 v1 入口已把这些身份纳入协议指纹，并拒绝 stale `RUNNING` trial。
- 不得删除或改写支持既有结论的原始产物；若必须迁移，保留校验信息与迁移记录。

## 9. Preconditions for Formal GPU Work

正式训练或正式 Optuna 前，必须同时满足：

- patient/content-level leakage 审计完成；
- baseline / CBAM / LCBAMv2 的实现、插入位置和初始化策略核验完成；
- 主比较与 Optuna 的全部非搜索参数冻结；
- seed、checkpoint、early stopping、指标与 test 使用规则冻结；
- GPU 机器、磁盘、依赖版本、唯一输出目录和失败恢复方案记录完成；
- import、build、forward、相关 tests 和最小 smoke 均通过。

未满足这些条件时，只能做代码验证或明确标记的 smoke/debug，不能启动正式实验并事后补协议。
