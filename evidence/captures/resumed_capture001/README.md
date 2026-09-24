# 正式USR kernel list：resumed_capture001

**状态：训练成功退出（returncode 0），四文件与完整训练验收通过。** 2026-09-24执行。以下四个文件可交给算子组；不是早期小型探针结果。

## 四份交付文件

| 文件 | 大小 | 用途 |
| --- | ---: | --- |
| [rank-0.json.gz](reports/rank-0.json.gz) | 11,483,570 bytes | 原始CPU/CUDA trace，包含通信和拷贝 |
| [rank-0_kernel_details_report.csv](reports/rank-0_kernel_details_report.csv) | 990,433 bytes | 算子/kernel/shape/dtype变体明细 |
| [rank-0_kernel_summary.csv](reports/rank-0_kernel_summary.csv) | 69,077 bytes | kernel调用次数、时间及占比 |
| [rank-0_operator_list.csv](reports/rank-0_operator_list.csv) | 67,387 bytes | 算子与kernel去重映射列表 |

文件大小和SHA256见 [manifest.json](manifest.json)。四文件按原运行字节复制，没有手工改表或丢弃低频算子。

## 负载与窗口

- 用户原始配置：`vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml`；原文件未改动。
- 单机8张H100 80GB，DP shard=8、replicate=1、CP=1；UND3072、GEN98304、GA2。
- 保留BF16、two_way、language static compile/full AC、原29份USR catalog及权重、480p/offline VAE、EMA关闭、原CFG和prefix条件采样。
- 完整恢复model/optimizer/scheduler/trainer到12600，新增25步并保存最终checkpoint12625。
- profiler为rank0，wait20/warmup2/active3/repeat1；active完成step12623–12625，共6个microbatch。
- Python3.13.14、Torch2.10.0+cu130、CUDA13.0、FA3 1.0.3+cu130.torch210。环境详情见 [runtime.json](runtime.json)，实际覆盖/命令见 [metadata.json](metadata.json)，完整配置见 [config.yaml](config.yaml)。

## 验证结论

| 指标 | 结果 |
| --- | ---: |
| 算子分组（operator_id） | 87 |
| 算子/kernel映射行 | 182 |
| 不同kernel名称 | 175 |
| shape/dtype变体明细行 | 1,534 |
| 报告中的计算kernel调用 | 35,496 |
| 原始trace中的计算kernel调用 | 35,496 |
| kernel名称对应的次数差异/超额归属 | 0 / 0 |
| forward/backward/optimizer阶段计数 | 6 / 6 / 3 |
| active窗口编译事件 | 0 |
| 各rank完整训练microbatch | 50（active各6） |

8个rank的active loss全部有限；rank0三步分别0.1815、0.1772、0.0873。表格字段、调用次数、时间、百分比、operator ID及原始trace覆盖均通过核验。重点抽查GEMM、FA3前后向、Triton融合计算和AdamW的External id关联，结果见 [attribution_audit.json](attribution_audit.json)。

rank0实际采到33个样本，T2V15、I2V13、V2V5，全部offline VAE；样本ID前缀分别为robomind2 14、lightwheel 11、agibot-world 6、egopro_lw2030 1、egostandard_lw2030 1。批级dataset标签与逐样本数量分别统计，没有把6个batch标签当成6个样本。完整数据见 [training_validation.json](training_validation.json)。

## 统计边界

- 三份CSV沿用固定版本FlagScale过滤规则，排除通信和内存拷贝；trace另含727次通信kernel和11,635次内存活动。
- `percent`的分母是报告中计算kernel累计时间127,866,900.223微秒，不是训练端到端耗时或考虑并行重叠后的关键路径。
- 12个变体行缺少dtype，涉及2,065次调用（5.82%），累计394,208.599微秒（0.3083%计算kernel时间）。集中在chunk/cat、foreach_copy和split操作；保留空值，没有按BF16配置补造类型。
- `input_shapes`外层非空不表示每个输入的尺寸都完整。标量及TensorList参数可能只有空维度占位，尤其fused AdamW的TensorList没有展开每个元素的尺寸。
- `custom_operator=null`表示上游Megatron/TE边界名单未识别Cosmos自定义父边界，不影响实际执行算子和kernel名称的保留。
- 本次覆盖单rank的实际6个microbatch，不声称遍历全部29个数据源或模型全部变体。日志的Warmup x/50来自原iter_speed统计回调，包含步末同步；本次结果用于算子筛选，不作为无侵入吞吐基准。

## 离线复核

在新仓库根目录执行，不需要GPU：

```bash
mkdir -p validation
python validate_reports.py evidence/captures/resumed_capture001/reports \
  --output validation/archived_quality.json
python validate_run.py evidence/captures/resumed_capture001
```

[report_quality.json](report_quality.json) 是采集时的校验记录，其中文件路径保留原runs目录以供审计。归档文件的相对路径及校验和以manifest为准。`rank-0_report_events.json.gz`另存上游导出器所用事件属性，方便后续离线重导出；该快照和其他验证材料不属于算子组的四份核心交付。
