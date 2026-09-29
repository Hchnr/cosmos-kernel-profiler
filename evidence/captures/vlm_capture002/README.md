# 纯图像 VLM kernel list：vlm_capture002

2026-09-28 八卡训练成功退出（returncode 0），训练完整性与 CSV 内部一致性通过。**224 种计算 kernel 名称全部覆盖；CSV 相对 trace 少 4 次调用，完整计数覆盖验收为 incomplete。** 差异保留在报告和审计文件中，没有手工补表。

## 四份交付文件

| 文件 | 字节 | 用途 |
| --- | ---: | --- |
| [rank-0.json.gz](reports/rank-0.json.gz) | 68,026,119 | 原始 CPU/CUDA trace，包含通信和内存活动 |
| [rank-0_kernel_details_report.csv](reports/rank-0_kernel_details_report.csv) | 2,470,552 | 算子/kernel/shape/dtype 变体明细 |
| [rank-0_kernel_summary.csv](reports/rank-0_kernel_summary.csv) | 90,158 | kernel 调用次数、时间和占比 |
| [rank-0_operator_list.csv](reports/rank-0_operator_list.csv) | 87,345 | 算子与 kernel 去重映射 |

完整清单与 SHA256 见 [manifest.json](manifest.json)。与此前 [T2V/USR](../resumed_capture001/README.md) 并列保存。

## 训练负载

- 参考用户指定 `ybai_0828/understanding_verification` 分支训练 Review，框架 SHA `9d8008581008c6209c7d3cb3bae5887ad65a49da`。原文与源 TOML、命令快照在 [reference/](reference/)。框架源码未修改。
- Qwen3-VL-8B-Instruct HF 初始化；LM + ViT 全参数训练，750 个参数张量可训练，其中视觉塔 351 个；BF16、full activation checkpointing、compile disabled。
- **数据替换：原 Grounding100K 的媒体引用不可访问，最终使用同分支 V2 配方中的 `gqa_train_balanced_reference`。** 这是 GQA 图像问答上的纯 VLM 采集，不是原 Grounding100K 历史训练复现，也不代表混合视频数据覆盖。
- 单机 8 张 H100 80GB，FSDP8 / replicate1 / CP1；原配方四机 FSDP32 作了单机适配。GA3，packing 上限 36k，保留 sample_mean / accumulation_window loss、优化器及原 25-step scheduler 设置。
- 共完成 step1–10，rank0 使用 wait5/warmup2/active3/repeat1，采集完成 step8–10。8 个 rank 各完成 30 个 microbatch，active 各 9 个；全部更新 loss 有限，保存最终 checkpoint10。
- rank0 active 有 432 个不同 catalog ID、432 个图像输入，无视频、生成视觉或 action 输入。9 个 pack 长度：10823、11011、12128、11697、10359、10551、10894、10344、11769。36k 是上限，不是实际固定长度。
- 最后三步 rank0 的末 microbatch loss：0.255281、0.462860、0.361954；窗口归一化 loss：0.371873、0.426935、0.319594。具体统计见 [training_validation.json](training_validation.json)、[workload_audit.json](workload_audit.json)。
- 使用 PyTorch profiler CPU+CUDA / Kineto 采集，通过 `profiler.events().kernels` 和固定 FlagScale 导出器生成 CSV；未启用 nsys。固定导出器 SHA256 为 `a147184fe5c172eb06e26c18825db89b60f2316c7d1c2c133aeb44e027d8dd29`，见仓库 [来源说明](../../../references/README.md)。

## 结果与边界

| 指标 | 结果 |
| --- | ---: |
| 算子分组 | 72 |
| 算子/kernel 映射 | 238 |
| 不同计算 kernel 名称 | 224 |
| shape/dtype 变体 | 3,920 |
| CSV 计算 kernel 调用 | 203,692 |
| 原始 trace 计算 kernel 调用 | 203,696 |
| 计算调用计数覆盖率 | 99.998036% |
| forward/backward/optimizer 阶段 | 9 / 9 / 3 |
| active 编译事件 | 0 |

- 原始 trace 另有 1,837 次通信 kernel、69,978 次内存活动；按固定导出器规则从三份 CSV 排除。
- **4 次调用差额合计 85.599 微秒，涉及三个已出现在 CSV 中的 kernel 名称。** 对导出器过滤后的事件快照与原始 trace 按名称和时长做多重集核对，差额已存在于 profiler 事件输入侧。候选事件中一个及其 runtime launch 缺少 External id，三个在 trace 中能关联到 cuDNN CPU 算子；确切上游原因未确认。同名同耗时事件无法仅凭多重集区分身份。证据见 [coverage_gap_investigation.json](coverage_gap_investigation.json)，可用 [audit_coverage_gap.py](audit_coverage_gap.py) 复核。
- 六类代表性算子（视觉卷积、GEMM、FA3 前后向、log-softmax、AdamW）的 External id、算子/kernel 对和输入维度与 CSV 匹配，见 [attribution_audit.json](attribution_audit.json)。这是抽查，不是逐事件归属的完整证明。视觉塔另外使用 cuDNN SDPA，不能将所有 Attention 都称为 FA3。
- 18 个变体行缺少 dtype，涉及 7,110 次调用（3.4906%）、1,020,930.531 微秒（2.1103% 计算时间）。保留空值，没有按 BF16 配置填造。TensorList/标量的 shape 可能仅有空维度占位，`input_shapes` 外层非空不等于每个输入维度完整。
- `custom_operator=null` 表示固定上游边界名单未识别 Cosmos 自定义父算子，不影响实际 execution_operator/kernel 名称保留。
- 百分比分母为 CSV 中计算 kernel 累计时间 48,378,913.034 微秒，不能直接视为端到端训练耗时占比。仅覆盖 rank0 的短窗口；不同数据、packing、compile 设置会改变 kernel 及其变体，不能与此前 T2V 直接作吞吐比较。

## 复核与复现

配置、命令和版本见 [metadata.json](metadata.json)、[config.yaml](config.yaml)、[runtime.json](runtime.json)、[launch.sh](launch.sh)。复现启动需使用新的 run-id，完整说明见 [docs/vlm_execution.md](../../../docs/vlm_execution.md)。模型、数据、checkpoint 和缓存未放入交付目录。

在仓库根目录、已有 Python 环境中执行（无需 GPU）：

```bash
python validate_vlm_run.py evidence/captures/vlm_capture002 --report-coverage-gaps
PYTHONPATH=. python evidence/captures/vlm_capture002/audit_coverage_gap.py evidence/captures/vlm_capture002
```

默认不加 `--report-coverage-gaps` 时，验收器会拒绝此次 4-call 计数差异。此参数仅使训练检查继续，输出仍明确写出 `kernel_count_coverage: incomplete`；不能解读为完整覆盖通过。[report_quality.json](report_quality.json) 为采集当时的原始记录，保留原 runs 路径。[implementation/](implementation/) 对应启动时代码指纹，[postprocess/](postprocess/) 为采集后验证代码；更改仅增加明确报告覆盖缺口的诊断模式。

此前混合图像/视频尝试未完成，诊断材料另存 [evidence/runs/vlm_capture001](../../runs/vlm_capture001/README.md)。不与此次已完成的图像 VLM 报告混用。
