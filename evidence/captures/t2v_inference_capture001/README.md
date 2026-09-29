# T2V 推理 kernel list：WorldArena Eval25

状态：推理完成，16/16 视频验证通过，CSV 与原始 trace 的全部计算 kernel 名称和逐名称调用次数一致。2026-09-29，物理 GPU 4–7，4×H100 80GB；仅 rank0（GPU 4）记录 profiler。

## 交付文件

[纯 kernel 名称列表](reports/kernel_names.txt) 共 **144 行**，可直接交给算子组。

| 校验项 | 结果 |
| --- | ---: |
| 不同计算 kernel 名称 | 144 |
| 计算 kernel 调用（CSV / trace） | 295,739 / 295,739 |
| 算子分组 / 算子-kernel 映射 | 64 / 153 |
| shape/dtype 变体 | 928 |
| 逐 kernel 调用计数差异 / 超额归属 | 0 / 0 |
| 正式窗口编译事件 | 0 |
| 原始 trace 通信 kernel 调用 | 3,654 |
| 原始 trace 内存事件 | 59,780 |

8 个变体缺失 dtype，涉及 36,348 次调用、3,725,280.368 微秒；保留空值，没有按 BF16 配置补造类型。明细中的 TensorList/标量输入仍可能只有空维度占位。原始 trace 独立审计确认包含 FA3 forward 和 VAE 卷积，没有 backward，见 [trace_audit.json](trace_audit.json)。

- [kernel 汇总](reports/rank-0_kernel_summary.csv)
- [算子/kernel/shape/dtype 明细](reports/rank-0_kernel_details_report.csv)
- [算子映射](reports/rank-0_operator_list.csv)
- [原始 CPU/CUDA trace](reports/rank-0.json.gz)

## 真实评测入口与配置

入口为框架 `cosmos_framework/scripts/inference.py:35`，实际调用 `cosmos_framework/inference/inference.py:1495` 的 `OmniInference.generate_batch`；`:1683` 进入采样，`:1711` 解码 VAE。框架源码未修改。外部 runner 包装一次完整预热与一次完整采集，使用已有 `profiler_reports.py` 和 `validate_reports.py` 原样导出、校验。

参考用户指定的 `outputs/worldarena_eval25_480p_steps50_transfer/README.md:9` 和 `run_usr_v2_21k.sh:49`；原文快照在 [reference/worldarena/](reference/worldarena/)。使用该评测的 USR v2 **iter21000 online DCP** 及其原 `config.yaml`，不是最初探查时的发布版 baseline。具体路径见 [launch.sh](launch.sh)、[reference/checkpoint_config.yaml](reference/checkpoint_config.yaml)。

- UniPC，50 steps，CFG 5，shift 10，seed 2026090900；480p 桶、4:3、实际 736×544、30 FPS。
- 保持原 prompt，`native_prompt_upsampling=false`，音频和 action 分支关闭，guardrails 关闭；不生成音频。
- BF16；FSDP shard4、DP replicate1、CP1、CFGP1，每卡 batch4。参考评测是 8 卡 shard4×replicate2，本次按用户限定四卡适配为 shard4×replicate1。
- 保留 language dynamic compile，关闭 CUDA graphs；fixed sequence length 关闭，不把推理 padding 成训练长度。
- 旧 checkpoint 配置缺少当前代码新增的 `diffusion_expert_config.vision_spatial_position_mode`，显式补成当前默认 `canvas`（框架 `cosmos_framework/configs/base/defaults/model_config.py:58`），见启动命令。

## 输入和采集窗口

直接读取原 WorldArena 25 条 Parquet，选择全部 **16 条纯 T2V 且原长度不超过 245 帧**的样本，排除 5 条 TI2V 和 4 条超长 T2V；prompt、fps、帧数和行内其他内容原样保留。完整选择规则、源行序号及 ID 见 [input_selection.json](input_selection.json)，实际输入见 [worldarena_t2v16.parquet](worldarena_t2v16.parquet)。这是代表性子集采集，不是重新跑完整 Eval25。

| Rank | 原始帧数 | 每卡打包视觉 tokens（每个 CFG 分支） |
| --- | --- | ---: |
| 0，正式采集 | 177、153、117、105 | 55,131 |
| 1 | 149、93、221、221 | 68,034 |
| 2 | 165、145、245、145 | 69,598 |
| 3 | 213、221、133、213 | 77,418 |

视觉 tokens 按 `((frames-1)/4+1) × (736/32) × (544/32)` 计算；VAE 压缩 4×16×16、空间 patch2，不含文本 tokens。实际 tensor/kernel shapes 在明细 CSV（45 个明细变体观察到 rank0 的 55,131 维度，例子见 `observed_vision_shape_examples.json`），逐 rank 的最终参数在 [batches/](batches/)。没有用合成超短输入替代评测输入。

完整 50 步采样和 VAE 解码先预热一次，随后采集相同输入与 seed 的完整 batch：CFG 条件/无条件计算、所有去噪步、VAE 解码及输出保存均在请求窗口内。checkpoint 加载、预热、输入 catalog 读取和 batch 初始准备不在 trace 内。其余 rank 执行相同完整流程但不采集 profiler，导出后通过 barrier 同步。

CSV 沿用原导出器的计算 kernel 口径，排除通信与 memcpy/memset；原始 trace 保留通信和内存事件。时间百分比是计算 kernel 累计时间占比，不能当作端到端耗时或无 profiler 性能基准。仅覆盖 rank0 的这一个真实 batch，不代表全部数据或所有 batch/长度组合。

## 复现与核验

[launch.sh](launch.sh) 保存完整四卡命令；重跑时使用新的 `capture_root`，并复制 `worldarena_t2v16.parquet` 到新目录，避免覆盖已采集结果。实现快照见 [implementation/](implementation/)，环境见 [environment_packages.json](environment_packages.json)。早期尚未进入采样的探查失败留在 `attempts/`，不属于正式 capture。最初的 canonical T2V 输入方案已由用户指定的 WorldArena 方案替换。

```bash
python validate_reports.py evidence/captures/t2v_inference_capture001/reports --output validation/t2v_inference_quality.json
python validate_inference_run.py evidence/captures/t2v_inference_capture001
```

[report_quality.json](report_quality.json) 记录表格与 trace 覆盖；[inference_validation.json](inference_validation.json) 确认全部 16 个视频的帧数、尺寸、FPS、无音轨及各 rank 批量均通过；[manifest.json](manifest.json) 保存归档文件校验和。
