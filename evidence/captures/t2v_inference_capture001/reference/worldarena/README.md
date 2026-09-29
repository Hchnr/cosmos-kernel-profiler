# WorldArena Eval25：Cosmos3-Nano T2V Checkpoint 对比评测（480p / 50 steps）

状态：已完成。共二十八路、700 个正式展示视频。2026-09-28 新增线上 USR v2 的 18k/20k/21k checkpoint 原 Prompt 评测，三路均为 25/25 条成功结果。纠正规则前生成的 13k/14k 扩写 Prompt 结果保留供追溯，不纳入正式对比；页面按训练 step 递增，12k FP8 与常规 12k 相邻。

12k FP8 W8A8 的量化范围、性能和复现说明见 [FP8_12K.md](FP8_12K.md)。Reasoner 扩写 Prompt 的适用规则、baseline 复评和历史误跑说明见 [REASONER_UPSAMPLED.md](REASONER_UPSAMPLED.md)，页面默认显示 baseline 原 Prompt 与扩写 Prompt 两路。

## 实验协议

- 输入：`/share/project/liyue/data/eval/worldarena-eval25-640x480-20260909/worldarena-eval25.parquet`
- Reasoner 扩写 Prompt 输入：只允许用于发布版 baseline，不用于任何训练 checkpoint（包括 13k/14k）
- Baseline checkpoint：`/share/project/eai_pwm/models/Cosmos3-Nano`
- 原始 Nano checkpoint：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/cosmos3-t2v/nano-128gpu/nano-t2v-action100m-koala-128gpu-ga2-hsdp8x16-r480-fixed-static-seqlen64k-v1/checkpoints/iter_000014000`
- 自研 VLM v1 iter-1000 checkpoint：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v1/cosmos3-usr/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga4-cfg01-cond731/checkpoints/iter_000001000`
- 自研 VLM v1 iter-2600 checkpoint：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga4-cfg01-cond731/checkpoints/iter_000002600`
- 自研 VLM v1 iter-4200 checkpoint：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000004200`
- 自研 VLM v1 约 4.5k checkpoint（实际 iter-4600）：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000004600`
- 自研 VLM v1 iter-6600 checkpoint：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000006600`
- 自研 VLM v2 iter-1000/2000/3000/4000/4800/6000/7000/8000/9000/10000/11000/12000 checkpoints：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_{000001000,000002000,000003000,000004000,000004800,000006000,000007000,000008000,000009000,000010000,000011000,000012000}`
- 自研 VLM v2 iter-13000/14000/15000/18000/20000/21000 checkpoints（USR v2，R7-1056 理解权重）：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v2/cosmos3-t2v/nano-128gpu/nano-usr-v2-r7-1056-resume12600-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541/checkpoints/iter_{000013000,000014000,000015000,000018000,000020000,000021000}`
- Fine-tuned 配置：对应实验目录的 `config.yaml`
- 样本：25 条；20 条 T2V、5 条 TI2V
- TI2V：`conditioning_latent_frames=1`，以 `gen_video` 的第 0 帧为条件
- seed：各路逐条相同，均为 `2026090900`
- 分辨率参数：`resolution=480`、`aspect_ratio=4,3`
- 实际输出：H.264、736×544、30 FPS，帧数随样本为 93～477
- 采样：UniPC、50 steps、CFG 5、shift 10
- 运行资源：8×NVIDIA H100 80GB，`throughput`；iter-1000、iter-2600 与 iter-4200 unbatched 使用 `max_num_seqs=1`，其余 DCP 结果使用 `max_num_seqs=4`
- 后续 WorldArena 正式评测必须显式使用真正的 batched infer：`--max-num-seqs=4`；8×H100 推荐同时使用 `--dp-shard-size=4 --dp-replicate-size=2 --use-torch-compile --compiled-region=language --no-use-cuda-graphs`
- `max_num_seqs=1` 仅用于 unbatched 正确性或显存诊断。加载 25 条输入不等于自动 batching；该参数默认值就是 1
- Eval25 启动后应在 `debug.log` 看到 1 个 local/global batch，并在 `launch.log` 只看到 1 轮 50-step `Sampling`；出现 4 轮时立即停止并检查 `max_num_seqs`
- guardrails 与 fixed sequence length 均关闭；原十五路关闭 torch compile，iter-8000～21000 十一路使用 language dynamic compile
- 所有正式 DCP 结果均显式加载 online (`net.*`) 权重；原始 Nano iter-14000 虽保留 EMA，也使用 online 权重以保持横向口径一致

## 最终结果

| 目录 | 权重 | 状态 |
| --- | --- | --- |
| `nano_base_480p_steps50_cfg5` | 原始 Cosmos3-Nano | 25/25 |
| `nano_base_reasoner_upsampled_480p_steps50_cfg5` | 原始 Cosmos3-Nano · Reasoner 扩写 Prompt | 25/25 |
| `nano_t2v_iter14000_original_vlm_online_480p_steps50_cfg5_batched4` | 原始 Nano VLM，T2V iter-14000 online | 25/25 |
| `vlm_v1_t2v_iter1000_online_480p_steps50_cfg5` | 自研 VLM v1 · T2V SFT iter-1000 · online | 25/25 |
| `vlm_v1_t2v_iter2600_online_480p_steps50_cfg5` | 自研 VLM v1 · T2V SFT iter-2600 · online | 25/25 |
| `vlm_v1_t2v_iter4200_online_480p_steps50_cfg5_batched4` | 自研 VLM v1 · T2V SFT iter-4200 · online · batched=4 | 25/25 |
| `vlm_v1_t2v_iter4200_online_480p_steps50_cfg5_unbatched` | 自研 VLM v1 · T2V SFT iter-4200 · online · unbatched | 25/25 |
| `nano_usr_iter4600_qwen8b1684_online_480p_steps50_cfg5_batched4` | 自研 VLM v1 · T2V SFT iter-4600（约 4.5k）· online | 25/25 |
| `nano_usr_iter6600_qwen8b1684_online_480p_steps50_cfg5_batched4` | 自研 VLM v1 · T2V SFT iter-6600 · online | 25/25 |
| `vlm_v2_t2v_iter1000_online_480p_steps50_cfg5_batched4` | 自研 VLM v2 · T2V SFT iter-1000 · online | 25/25 |
| `vlm_v2_t2v_iter2000_online_480p_steps50_cfg5_batched4` | 自研 VLM v2 · T2V SFT iter-2000 · online | 25/25 |
| `vlm_v2_t2v_iter3000_online_480p_steps50_cfg5_batched4` | 自研 VLM v2 · T2V SFT iter-3000 · online | 25/25 |
| `vlm_v2_t2v_iter4000_online_480p_steps50_cfg5_batched4` | 自研 VLM v2 · T2V SFT iter-4000 · online | 25/25 |
| `nano_usr_iter4800_qwen3vl8b_instruct_online_480p_steps50_cfg5_batched4` | 自研 VLM v2 · T2V SFT iter-4800 · online | 25/25 |
| `vlm_v2_t2v_iter6000_online_480p_steps50_cfg5_batched4` | 自研 VLM v2 · T2V SFT iter-6000 · online | 25/25 |
| `vlm_v2_t2v_iter7000_online_480p_steps50_cfg5_batched4` | 自研 VLM v2 · T2V SFT iter-7000 · online | 25/25 |
| `vlm_v2_t2v_iter8000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · T2V SFT iter-8000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter9000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · T2V SFT iter-9000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter10000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · T2V SFT iter-10000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter11000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · T2V SFT iter-11000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter12000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · T2V SFT iter-12000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter12000_online_480p_steps50_cfg5_fp8_w8a8_dp8_compile_language_batched4` | 自研 VLM v2 · T2V SFT iter-12000 · online · FP8 W8A8 对照 | 25/25 |
| `vlm_v2_t2v_iter13000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · USR v2 R7-1056 · T2V SFT iter-13000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter14000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · USR v2 R7-1056 · T2V SFT iter-14000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter15000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · USR v2 R7-1056 · T2V SFT iter-15000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter18000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · USR v2 R7-1056 · T2V SFT iter-18000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter20000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · USR v2 R7-1056 · T2V SFT iter-20000 · online · 最优推理配置 | 25/25 |
| `vlm_v2_t2v_iter21000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | 自研 VLM v2 · USR v2 R7-1056 · T2V SFT iter-21000 · online · 最优推理配置 | 25/25 |

每个样本目录均保留最终 `vision.mp4` 以及用于复核的 `sample_args.json`、`sample_outputs.json`。直接用浏览器打开本目录的 `index.html` 可逐条同步对比二十八路正式视频；页面不依赖 HTTP 服务。页面支持按训练 step 勾选 Checkpoint，默认显示发布版 baseline 的原 Prompt 和 Reasoner 扩写 Prompt，支持全选、清空及选择记忆，未选视频不加载。原始和扩写后的 Prompt 均可查看。下载改为按 Checkpoint 独立打包，见 [增量下载说明](./DOWNLOAD.md)。

新 Prompt baseline 已生成结果的单次运行墙钟为 1,011 秒，4 轮 50 步采样合计 881 秒，500 ms 轮询记录的逐卡最高显存为 38,389 MiB。这次误用了 `max_num_seqs=1`，属于 unbatched 运行，结果保留但耗时不得作为 batched infer 性能基准；按要求不重新生成。正确配置与审计见 [REASONER_UPSAMPLED.md](./REASONER_UPSAMPLED.md)。

## iter-8000～21000：最优推理配置

十一路按 [推理优化测试结论](../worldarena_eval25_480p_steps50_infer_test/summary.md) 使用 `--dp-shard-size=4 --dp-replicate-size=2 --max-num-seqs=4 --use-torch-compile --compiled-region=language --no-use-cuda-graphs`，保持 dynamic compile。输入、online 权重、Wan2.2 VAE、seed 和采样质量参数均沿用本页协议；完整命令、日志及 500 ms GPU 遥测保留在各结果目录。旧十五路关闭 compile，因此跨旧结果的画面差异同时包含 checkpoint 与编译路径差异；iter-8000～21000 十一路之间的推理配置一致。

| checkpoint | 成功 | 总墙钟 | 50 步采样日志 | 逐卡最高采样显存 | 运行记录 |
| --- | ---: | ---: | ---: | ---: | --- |
| iter-8000 | 25/25 | 549 s | 428 s | 42,628 MiB | [命令](./vlm_v2_t2v_iter8000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter8000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-9000 | 25/25 | 555 s | 428 s | 42,628 MiB | [命令](./vlm_v2_t2v_iter9000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter9000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-10000 | 25/25 | 550 s | 428 s | 42,628 MiB | [命令](./vlm_v2_t2v_iter10000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter10000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-11000 | 25/25 | 550 s | 428 s | 42,628 MiB | [命令](./vlm_v2_t2v_iter11000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter11000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-12000 | 25/25 | 550 s | 428 s | 42,628 MiB | [命令](./vlm_v2_t2v_iter12000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter12000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-13000 | 25/25 | 556 s | 428 s | 42,628 MiB | [命令](./vlm_v2_t2v_iter13000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter13000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-14000 | 25/25 | 563 s | 435 s | 42,632 MiB | [命令](./vlm_v2_t2v_iter14000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter14000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-15000 | 25/25 | 552 s | 428 s | 42,073 MiB | [命令](./vlm_v2_t2v_iter15000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter15000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-18000 | 25/25 | 554 s | 428 s | 42,073 MiB | [命令](./vlm_v2_t2v_iter18000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter18000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-20000 | 25/25 | 553 s | 428 s | 42,073 MiB | [命令](./vlm_v2_t2v_iter20000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter20000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |
| iter-21000 | 25/25 | 553 s | 428 s | 42,073 MiB | [命令](./vlm_v2_t2v_iter21000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt) · [指标](./vlm_v2_t2v_iter21000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) |

可用本目录的 [`run_vlm_v2_new_ckpt.sh`](./run_vlm_v2_new_ckpt.sh) 复跑 8k～12k，13k/14k 分别使用 [`run_usr_v2_13k.sh`](./run_usr_v2_13k.sh) 和 [`run_usr_v2_14k.sh`](./run_usr_v2_14k.sh)，18k/20k 使用 [`run_usr_v2_18k_20k.sh`](./run_usr_v2_18k_20k.sh)，21k 使用 [`run_usr_v2_21k.sh`](./run_usr_v2_21k.sh)；推理会跳过已成功生成的样本。上表是运行与资源指标，不是感知质量分数；画面及文本遵循度请在 `index.html` 中逐条查看。

Reasoner 扩写 Prompt 只使用 [`run_nano_base_reasoner_upsampled.sh`](./run_nano_base_reasoner_upsampled.sh) 在发布版 baseline 上评测；该 runner 已改为 batched 推荐配置。运行与审计结果见 [REASONER_UPSAMPLED.md](./REASONER_UPSAMPLED.md)。原 13k/14k 扩写 Prompt runner 已禁用。

## 2026-09-25：USR v2 14k / 15k 评测

来源均为当前线上 USR v2 训练：`/share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v2/cosmos3-t2v/nano-128gpu/nano-usr-v2-r7-1056-resume12600-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541/checkpoints/iter_{000014000,000015000}`。

- **14k**：与已有正式评测的 checkpoint 路径一致，model 文件最新修改时间早于旧评测；复用原有结果，本次重新核对 25/25 条样本协议、online 权重、单批推理日志并完整解码全部视频。保留原有运行时间和压缩包，不将复用记录标为新推理。
- **15k**：使用原始 WorldArena Eval25 Parquet，20 条 T2V + 5 条 TI2V，online 权重、480p、UniPC 50 steps、CFG 5、shift 10、seed 2026090900。8×H100，FSDP4×DP2、`max_num_seqs=4`、language dynamic compile；日志确认 1 个 local/global batch、1 轮 50-step Sampling。未使用 Reasoner 扩写或 FP8。
- 15k 总墙钟 **552 s**，采样 **428 s**，500 ms 轮询记录的逐卡最高显存 **42,073 MiB**；25/25 条输出成功。
- 联合审计 14k/15k：**50/50** 视频完整解码通过，H.264、736×544、30 FPS、帧数及所有协议字段一致，每组 25 个不同的 MP4 SHA-256。
- 首次 15k 启动因环境缺少 `iopath` 而在模型加载前退出；按仓库锁文件恢复推理所需的 `train` extra 和 `cu130-train` 依赖后重试（跳过 T2V/TI2V 不使用的 lerobot；完整环境与命令见 15k 的 `environment.json`）。首次失败记录保存在 15k 的 `attempts/missing_training_dependencies/`；成功运行使用 `uv run --no-sync` 保持环境。

复现：`bash run_usr_v2_15k.sh`（使用脚本中的绝对路径，可从任意目录执行）。审计：`LD_LIBRARY_PATH='' python audit_usr_v2_14k_15k.py`。

记录：[联合审计](./audit_usr_v2_14k_15k_summary.json) · [checkpoint 来源](./usr_v2_14k_15k_checkpoint_provenance.json) · [15k 指标](./vlm_v2_t2v_iter15000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) · [15k 命令](./vlm_v2_t2v_iter15000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/command.txt)。这些检查验证协议与文件完整性，不代表感知质量分数；画质与文本遵循度请在同一 `index.html` 中勾选 14k/15k 对比。

## 2026-09-28：USR v2 18k / 20k 评测

18k 与 20k 使用原始 WorldArena Eval25 Parquet，不使用 Reasoner 扩写 Prompt。两路均加载 online、非量化权重，使用 8×H100、FSDP4×DP2、`max_num_seqs=4` 和 language dynamic compile；日志均确认 1 个 local/global batch 及 1 轮 50-step Sampling。

- **18k**：25/25 条成功，总墙钟 **554 s**，采样 **428 s**，500 ms 轮询记录的逐卡最高显存 **42,073 MiB**。
- **20k**：25/25 条成功，总墙钟 **553 s**，采样 **428 s**，500 ms 轮询记录的逐卡最高显存 **42,073 MiB**。
- 联合审计：**50/50** 视频完整解码通过，H.264、736×544、30 FPS、帧数及全部协议字段一致；每组 25 个 MP4 SHA-256 均不重复。
- GPU 忙时 runner 会等待连续两分钟空闲；正式运行中若检测到额外计算进程，会中止并归档该次尝试，待空闲后重跑。

复现：`bash run_usr_v2_18k_20k.sh`。验收：`LD_LIBRARY_PATH='' python finalize_usr_v2_18k_20k.py`。

记录：[联合审计](./audit_usr_v2_18k_20k_summary.json) · [checkpoint 来源](./usr_v2_18k_20k_checkpoint_provenance.json) · [18k 指标](./vlm_v2_t2v_iter18000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json) · [20k 指标](./vlm_v2_t2v_iter20000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json)。

## 2026-09-28：USR v2 21k 评测

21k 使用原始 WorldArena Eval25 Parquet，不使用 Reasoner 扩写 Prompt。运行加载 online、非量化权重，使用 8×H100、FSDP4×DP2、`max_num_seqs=4` 和 language dynamic compile；日志确认 1 个 local/global batch 及 1 轮 50-step Sampling。运行期间没有检测到其他 GPU 计算进程。

- 25/25 条成功，总墙钟 **553 s**，采样 **428 s**，500 ms 轮询记录的逐卡最高显存 **42,073 MiB**。
- **25/25** 个视频完整解码通过；H.264、736×544、30 FPS、帧数及全部协议字段一致，25 个 MP4 SHA-256 均不重复。
- 复现：`bash run_usr_v2_21k.sh`。验收：`LD_LIBRARY_PATH='' python finalize_usr_v2_21k.py`。

记录：[审计](./audit_usr_v2_21k_summary.json) · [checkpoint 来源](./usr_v2_21k_checkpoint_provenance.json) · [21k 指标](./vlm_v2_t2v_iter21000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4/metrics.json)。

## 审计结果

- 正式二十八路 `sample_args.json`、`sample_outputs.json`、`vision.mp4` 各 25 个
- 正式 `sample_outputs.json`：700/700 为 `success`
- 每路模态均为 20 条 `text2video`、5 条 `image2video`；TI2V 均有且仅有 1 个条件帧索引
- 原十四路对比结果与 baseline 的 25 组 prompt、model mode、seed、分辨率、帧数和采样参数逐条一致，差异数均为 0；18k/20k/21k 与 iter-7000 逐条比对同一协议，差异数均为 0
- 视频：700/700 均为 H.264、736×544、30 FPS，实际帧数与 `sample_args.json` 一致
- 完整 ffmpeg 解码：700/700 通过；新增 18k/20k 的 50 个 MP4 见 [`audit_usr_v2_18k_20k_summary.json`](./audit_usr_v2_18k_20k_summary.json)，新增 21k 的 25 个 MP4 见 [`audit_usr_v2_21k_summary.json`](./audit_usr_v2_21k_summary.json)
- MP4 SHA-256：每路 25/25 唯一
- 新增 baseline 一路：25/25 条输出成功、25/25 个视频完整解码、Prompt 与新 Parquet 逐条一致，且 checkpoint 标记为发布版 Cosmos3-Nano；详见 [`audit_baseline_reasoner_upsampled_summary.json`](./audit_baseline_reasoner_upsampled_summary.json)
- 先前 13k/14k 扩写 Prompt 结果及 [`audit_reasoner_upsampled_summary.json`](./audit_reasoner_upsampled_summary.json) 保留供追溯，不计入上述正式统计
- 3 条视频超过模型建议的 300 帧上限（325、341、477 帧），已按测试集原始长度生成，可能存在长视频质量退化

上述检查验证了数据接入、混合模态 batch、TI2V 条件路径、权重选择、参数一致性、输出数量和文件完整性；本页面用于人工感知质量对比，不代表自动感知质量评分。

## DCP 复现路径映射

原十四路 DCP 评测共用下方命令的采样参数与 overrides；不同结果按上文记录使用 `max_num_seqs=1` 或 `4`。iter-8000～21000 十一路共用采样参数与 overrides，并叠加上节记录的优化参数：

| 模型 | 输出目录 | Checkpoint | Config |
| --- | --- | --- | --- |
| 原始 Nano | `nano_t2v_iter14000_original_vlm_online_480p_steps50_cfg5_batched4` | `outputs/cosmos3-t2v/nano-128gpu/nano-t2v-action100m-koala-128gpu-ga2-hsdp8x16-r480-fixed-static-seqlen64k-v1/checkpoints/iter_000014000` | 同 run 的 `config.yaml` |
| 自研 VLM v1 iter-1000 | `vlm_v1_t2v_iter1000_online_480p_steps50_cfg5` | `outputs/usr_v1/cosmos3-usr/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga4-cfg01-cond731/checkpoints/iter_000001000` | 同 run 的 `config.yaml` |
| 自研 VLM v1 iter-2600 | `vlm_v1_t2v_iter2600_online_480p_steps50_cfg5` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga4-cfg01-cond731/checkpoints/iter_000002600` | 同 run 的 `config.yaml` |
| 自研 VLM v1 iter-4200 batched | `vlm_v1_t2v_iter4200_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000004200` | 同 run 的 `config.yaml` |
| 自研 VLM v1 iter-4200 unbatched | `vlm_v1_t2v_iter4200_online_480p_steps50_cfg5_unbatched` | 同上 iter-4200 | 同 run 的 `config.yaml` |
| 自研 VLM v1（约 4.5k） | `nano_usr_iter4600_qwen8b1684_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000004600` | 同 run 的 `config.yaml` |
| 自研 VLM v1 iter-6600 | `nano_usr_iter6600_qwen8b1684_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen8b1684-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000006600` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-1000 | `vlm_v2_t2v_iter1000_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000001000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-2000 | `vlm_v2_t2v_iter2000_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000002000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-3000 | `vlm_v2_t2v_iter3000_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000003000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-4000 | `vlm_v2_t2v_iter4000_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000004000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-4800 | `nano_usr_iter4800_qwen3vl8b_instruct_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000004800` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-6000 | `vlm_v2_t2v_iter6000_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000006000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-7000 | `vlm_v2_t2v_iter7000_online_480p_steps50_cfg5_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000007000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-8000 | `vlm_v2_t2v_iter8000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000008000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-9000 | `vlm_v2_t2v_iter9000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000009000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-10000 | `vlm_v2_t2v_iter10000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000010000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-11000 | `vlm_v2_t2v_iter11000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000011000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-12000 | `vlm_v2_t2v_iter12000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | `outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000012000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-13000 (USR v2 R7-1056) | `vlm_v2_t2v_iter13000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | `outputs/usr_v2/cosmos3-t2v/nano-128gpu/nano-usr-v2-r7-1056-resume12600-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541/checkpoints/iter_000013000` | 同 run 的 `config.yaml` |
| 自研 VLM v2 iter-14000 (USR v2 R7-1056) | `vlm_v2_t2v_iter14000_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4` | `outputs/usr_v2/cosmos3-t2v/nano-128gpu/nano-usr-v2-r7-1056-resume12600-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541/checkpoints/iter_000014000` | 同 run 的 `config.yaml` |

## 自研 VLM v2 完整复现命令（以 iter-4800 为例）

```bash
ulimit -n "$(ulimit -Hn)"
LD_LIBRARY_PATH='' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --nproc-per-node=8 \
  -m cosmos_framework.scripts.inference \
  -i /share/project/liyue/data/eval/worldarena-eval25-640x480-20260909/worldarena-eval25.parquet \
  -o /share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm/outputs/worldarena_eval25_480p_steps50/nano_usr_iter4800_qwen3vl8b_instruct_online_480p_steps50_cfg5_batched4 \
  --checkpoint-path /share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/checkpoints/iter_000004800 \
  --config-file /share/project/eai_pwm/repos/cosmos-framework-pwm/outputs/usr_v1/cosmos3-t2v/nano-128gpu/nano-usr-v1-qwen3vl8b-instruct-hybrid-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond731/config.yaml \
  --no-use-ema-weights \
  --parallelism-preset=throughput --max-num-seqs=4 \
  --sampler=unipc --resolution=480 --num-steps=50 --guidance=5 --shift=10 \
  --seed=2026090900 --no-use-torch-compile --no-guardrails \
  --experiment-overrides \
    "model.config.fixed_sequence_length.enabled=false" \
    "model.config.fixed_sequence_length.understanding_tokens=0" \
    "model.config.fixed_sequence_length.und_tokens=0" \
    "model.config.fixed_sequence_length.gen_tokens=0" \
    "model.config.load_vision_tokenizer=true" \
    "model.config.vision_vae_mode=online" \
    "model.config.tokenizer.vae_path=/share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth" \
    "model.config.tokenizer.bucket_name=''"
```

## 页面维护

页面由仓库中的 `tools/build_worldarena_viewer.py` 和 `tools/worldarena_viewer/` 模板生成。在仓库根目录执行：

```bash
LD_LIBRARY_PATH='' python tools/build_worldarena_viewer.py
```

脚本读取结果目录的 `viewer_data.json`，同步更新结果目录和 transfer 目录的 `index.html`，无需重新推理或打包。Baseline 优先，其余按实际训练 step 递增；实验名拆分为统一颜色的模型、Prompt 和配置标签，完整名称仍可查看。支持搜索、系列筛选、仅看已选和选择记忆。

固定发布目标：`ks3://baai-eai/eval/worldarena_eval25_480p_steps50_transfer/`。页面或 transfer 更新后，执行 `/share/project/eai_pwm/home/hcr/app/sync_worldarena_eval25_480p_steps50_transfer.sh` 并校验云端内容。具体命令、endpoint 与校验步骤见 [金山云发布说明](./DOWNLOAD.md#金山云发布固定目标)。
