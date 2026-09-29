# 纯 VLM kernel list（2026-09-28）

最终产物：[evidence/captures/vlm_capture002](../evidence/captures/vlm_capture002/README.md)。八卡训练完成 10 次更新，returncode 0；使用 GQA 图像问答替代媒体路径不可用的 Grounding100K，采集 step8–10。224 种计算 kernel 名称完整保留，CSV 比原始 trace 少 4 次调用，详见产物目录中的覆盖差异审计。以下保留配置适配和重采过程。

## 参考与范围

- 用户指定训练参考：`BAAI-PWM/cosmos-framework-pwm` 的 `ybai_0828/understanding_verification` 分支，`docs/understanding_training_review.md`。
- 2026-09-28 fetch 后分支 SHA：`9d8008581008c6209c7d3cb3bae5887ad65a49da`。
- 独立 detached checkout：`/share/project/eai_pwm/home/hcr/worktrees/cosmos-vlm-kernel-20260928`，不改已有框架工作树。
- 源配方：`examples/understanding_review/grounding100k_four_node_ga3.toml`。
- 参考配方使用 Qwen3-VL-8B-Instruct HF 初始化、LM + ViT 训练、Grounding100K 数据、36,000 token packing 上限、GA3、BF16、full AC、compile disabled、sample-mean/accumulation-window loss。最终保留模型和训练设置，数据替换原因见下文。
- 原四机 FSDP32 适配到本机八张 H100 的 FSDP8/replicate1/CP1。每次更新为 24 个 pack，不能当作历史四机训练的等价收敛复现。
- 初始混合数据尝试从 step0 开始，wait20/warmup2/active3/repeat1，计划 rank0 采集完成 step23–25；最终 GQA 运行改为 wait5/warmup2/active3，forward/backward/optimizer 为 9/9/3。
- 原历史审计回调和协调器不复用。关闭 W&B、上传、validation、样本预览、nsys；使用本仓库现有 PyTorch profiler 与固定 FlagScale 导出器。
- 最终交付目录：`evidence/captures/vlm_capture.../`，与 T2V `evidence/captures/resumed_capture001/` 并列。

## 环境与检查

默认 `/opt/venv/bin/python` 当前无 Torch；使用已有隔离训练环境：
`/share/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train/bin/python`。
Python3.13.14 / Torch2.10.0+cu130 / CUDA13.0 / torchvision0.25.0+cu130 / transformers4.57.6 / FA3 1.0.3+cu130.torch210。

- 初始 GPU 检查：8 张 H100 80GB 全部空闲。
- `validation/vlm-cuda-probe-001`：实际 CUDA profiler 探针通过，24 次计算 kernel，四文件正常导出。
- `validation/vlm-attention-001`：框架 Attention causal/noncausal 编译前后向与 FP32 参考核验通过，最大输出绝对误差 0.0082207/0.0044644。
- 原有 10 项 CPU 检查、新增 5 项 VLM 验收检查通过。
- `vlm_preflight001`：参考分支不包含 T2V 分支的 `trainer.compile_cache` 配置，预检拒绝该覆盖；删除此不适用配置，用新 run-id 重试。
- `vlm_preflight002`：Grounding100K 配方的配置预检成功，确认 FSDP8、36k、GA3、LM+ViT、HF 初始化覆盖正确。

## 数据路径适配

Grounding100K Parquet 存在，但图片引用 `/home/baiyu/chenbangpu/grounding100k/releases/wild_source_v4/media/`，该机器本地路径在当前容器不可用；共享盘上的另一份 Grounding100K source pool 也引用相同路径。已向用户询问可用媒体目录，同时继续执行纯 VLM 采集准备。

为了完成纯 VLM kernel list 目标，本次选择同分支已有 `examples/toml/sft_config/qwen3vl_v2_0a_understanding_nano8b.toml` 的 20 个数据源及其原比例。仅替换数据源列表，保留 Review 配方的模型、训练、packing 上限、GA、优化器和 loss 设置。`--dataset-recipe` 将此差异写入生成的 recipe 和 metadata；不修改共享数据和源 TOML。此运行应标记为 V2 reference 数据上的纯 VLM 采集，不能称为原 Grounding100K 历史训练复现。

- `validation/vlm-data-preflight.json`：20 个数据源各检查 3 条样本，共 60 条、358 个媒体路径，无缺失；每个数据源首张图片也通过 PIL 验证。
- `vlm_preflight003`：替换数据源列表后的配置预检成功。
- `vlm_capture001`：单机八卡混合数据尝试，torchrun PID1962102；未完成，诊断材料归档于 `evidence/runs/vlm_capture001`。

## 混合数据运行未完成与图像 VLM 重采

`vlm_capture001` 完成 step1–23，全部 rank 的已完成更新 loss 有限。step24 出现数分钟无完成记录，期间 rank3 的预取数据加载器报告一个视频解码 120 秒超时并按框架逻辑跳过该样本。已保留日志、参数/批次记录、代码快照及 `incomplete_run.json`，没有将该次运行列为正式交付。

2026-09-28 10:57 UTC 向本任务自己的 torchrun PID1962102 发 SIGTERM，最终 returncode1、GPU进程释放。结束时各 rank 记录为71个完成microbatch、23次优化器更新，说明中断过程前后还有少量推进；中断栈处于 backward/梯度归一化等待，根因尚未确认。混合图像/视频分别调用FSDP视觉塔的潜在调用顺序问题只是排查假设，不作为已证实结论。

下一次采集选择同一 V2 数据列表中的 `gqa_train_balanced_reference`，保持参考模型、LM+ViT全参数训练、36k上限、GA3、BF16、full AC、compile关闭。GQA 与原 Grounding100K 一样是图像问答输入，可避免混合媒体路径。使用 wait5/warmup2/active3，共10次更新，采集完成step8–10。增加 `microbatch_start` 元数据和可选 SIGUSR2 Python 栈转储，便于区分数据、前向、反向等待；不改变模型计算和导出器。

```bash
$PY launch_vlm_profile.py --run-id vlm_capture002 --wait 5 \
  --dataset-recipe "$COSMOS_REPO/examples/toml/sft_config/qwen3vl_v2_0a_understanding_nano8b.toml" \
  --dataset-name gqa_train_balanced_reference
```

## 启动与离线验收

```bash
export COSMOS_REPO=/share/project/eai_pwm/home/hcr/worktrees/cosmos-vlm-kernel-20260928
export LD_LIBRARY_PATH=''
PY=/share/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train/bin/python
$PY launch_vlm_profile.py --run-id vlm_preflight_new --dryrun
$PY launch_vlm_profile.py --run-id vlm_capture_new
$PY validate_vlm_run.py runs/vlm_capture_new
```

使用此次 V2 reference 数据列表时，两个启动命令都增加：

```bash
--dataset-recipe "$COSMOS_REPO/examples/toml/sft_config/qwen3vl_v2_0a_understanding_nano8b.toml"
```

复现最终 GQA 采集还需增加 `--dataset-name gqa_train_balanced_reference --wait 5`，并使用新的 run-id。

## 最终验证与覆盖边界

`vlm_capture002` 于 2026-09-28 11:13:26 UTC 正常退出，最终 checkpoint 为 step10。八个 rank 均完成 30 个 microbatch、10 次优化器更新并写入结束标记，所有更新 loss 有限。rank0 active 窗口包含 9 个 pack、432 个 GQA 图像问答样本；实际 pack 长度为 10,344–12,128，并非每包填满 36k。750 个参数张量全部可训练，其中 351 个属于视觉塔。没有视频或生成分支输入。

三份 CSV 内部一致性通过，224 种计算 kernel、72 个算子分组、238 条映射、3,920 个 shape/dtype 变体。原始 trace 为 203,696 次计算 kernel 调用，CSV 为 203,692 次。完整 kernel 计数验收因此为 **incomplete**，不能称为全项通过。差额 4 次、85.599 微秒，涉及的三个 kernel 名称都已有其他调用进入报告。逐项多重集核对和候选事件见 `coverage_gap_investigation.json`，未手工补表或修改固定 FlagScale 导出器；上游事件关联缺口的确切原因未确认。

`validate_vlm_run.py` 默认仍拒绝计数不一致；`--report-coverage-gaps` 仅允许继续检查训练完整性，并在输出中明确保留 `kernel_count_coverage: incomplete` 和差异明细。离线复核此次产物：

```bash
$PY validate_vlm_run.py evidence/captures/vlm_capture002 --report-coverage-gaps
PYTHONPATH=. $PY evidence/captures/vlm_capture002/audit_coverage_gap.py evidence/captures/vlm_capture002
```

17 项 CPU 测试及相关 Python 文件的 Ruff 检查通过。归档分别保存采集时 `implementation/` 和采集后验收使用的 `postprocess/`，通过指纹区分版本。checkpoint、模型、数据和缓存未复制到交付目录。

启动器设置 RLIMIT_NOFILE=65536，保存配方、配置、受控环境、命令、代码指纹和两个 Git SHA。VLM 审计回调通过现有 trainer hooks 保存各 rank 的批次元数据、参数训练状态、每次更新 loss/监督权重和结束标记。张量内容不写入批次记录，shape/dtype 只读元数据；每次更新读取 loss 标量用于验收，不把此次采集当作无侵入吞吐测试。
