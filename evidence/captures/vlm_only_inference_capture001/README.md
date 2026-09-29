# vlm_only 推理 kernel list

完成时间：2026-09-29。GPU 5，单卡 H100 80GB。完整预热一次后采集一次真实图像问答请求，模型加载和预热不在 trace 中。

## 交付

[纯 kernel 名称列表](reports/kernel_names.txt) 可直接用于检索和比对（104 行）。

- [kernel 汇总](reports/rank-0_kernel_summary.csv)：104 个不同计算 kernel，73,663 次调用。
- [shape/dtype 明细](reports/rank-0_kernel_details_report.csv)：374 个变体。
- [算子映射](reports/rank-0_operator_list.csv)：46 个算子分组，111 个算子/kernel 映射。
- [原始 CPU/CUDA trace](reports/rank-0.json.gz)。

[report_quality.json](report_quality.json) 确认三表内部一致，全部 kernel 名称和逐名称调用次数与 trace 一致，没有缺失或超额归属。6 个变体缺少 dtype，涉及 4,760 次调用、16,982.808 微秒；保留缺失值，未按 BF16 配置推测填写。百分比是计算 kernel 累计时间占比，不是端到端延迟占比。

## 入口与负载

- 独立 VLM 评测入口：框架 `cosmos_framework/scripts/reasoner/eval_videophy2.py:240` 加载 HF `Qwen3VLForConditionalGeneration`，`:278` 使用 processor chat template，`:291` 调用 `model.generate`。本 runner 复用该调用方式，将视频评测输入替换成仓库内真实图像描述输入；并未运行整个 VideoPhy2 指标评测。
- 权重：`/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct`，与已有 `vlm_capture002` 训练采集初始化相同。LM+ViT，共 8,767,123,696 参数，BF16，Transformers 默认 SDPA，KV cache 开启，无 compile / CUDA graph。
- 图片：框架 `inputs/prompt_upsampler/image_inputs/humanoid_robot.jpg`。问题采用 `inputs/reasoner/reasoner_image.json:3` 的单句图片描述问题。精确输入见 [input.json](input.json)。
- Batch=1；输入序列 1,529 tokens，其中图像经 processor 得到 grid `[1,58,104]`，合并后 1,508 个视觉 tokens；不做训练式 packing/padding 到固定长序列。
- `max_new_tokens=128`，greedy，自然 EOS 实际生成 32 tokens。采集含视觉编码 1 次、prefill 1 次、decode 31 次，均在 trace 中标注。预热与采集生成 token 完全相同，见 [generation.json](generation.json)。
- 只代表这次图像理解请求，不代表视频理解或所有长度/批量组合；没有 backward 或 optimizer。

## 复核与复现

[launch.sh](launch.sh) 保存完整命令，正式运行使用 GPU 5；用户指定后四卡之前的首次权重加载已停止，日志留在 `pre_migration.log`，未形成 capture。重跑时请为脚本的 output 指定新目录，避免覆盖既有报告。

实现快照在 [implementation/](implementation/)，源码引用快照在 [reference/](reference/)。框架版本、配置、环境见 [metadata.json](metadata.json)、[runtime.json](runtime.json)、[model_runtime.json](model_runtime.json)、[batch_shapes.json](batch_shapes.json)。[manifest.json](manifest.json) 保存文件 SHA256。

```bash
python validate_reports.py evidence/captures/vlm_only_inference_capture001/reports --output validation/vlm_only_inference_quality.json
```
