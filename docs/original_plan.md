> 历史快照：记录迁移前的目录、命令及已移除的本地提交。当前用法和状态以仓库根目录 README.md 为准。

# v0.2.3c：训练 profiling 与算子列表导出计划

状态：2026-09-24 用户已授权实施和验证；实际采用 USR v2 配方、单机8卡、DP shard=8。执行进展见 record.md。

## 1. 目标与范围

对当前 Cosmos 训练负载做一次有预热、有限窗口的 PyTorch profiling，按 FlagScale 的输出约定生成 **1 个原始 trace + 3 个 CSV**，供算子组筛选可用算子。保留实际执行的算子名、kernel 名、输入形状/类型、调用次数和时间；本任务不替算子组判断适配结果。

- 工作目录：`/share/project/eai_pwm/home/hcr/worktrees/cosmos-framework-pwm-load-balance`。
- 本计划：`plans/v0.2.3c_kernel_list.md`。
- 全过程记录：`plans/v0.2.3c_kernel_list/record.md`，实施时持续追加命令、阶段、结果和问题。
- 后续新增脚本、派生配置、日志、缓存、训练产物和交付文件全部放在 `plans/v0.2.3c_kernel_list/` 内；不修改 `cosmos_framework/`、`examples/`、环境依赖文件或其他项目代码。
- 本轮以“代表性训练窗口中实际观测到的 GPU 计算算子”为清单范围，不声称覆盖模型全部任务、所有输入形状或纯 CPU 算子。通信和内存拷贝保留在 trace，CSV 过滤口径与参考实现一致。

## 2. 已核实的参考实现与本仓库接入点

### FlagScale

固定参考版本为 `0b7b638547c87b8129baec0499f859d96332db8e`，避免执行时 `main` 更新改变字段或过滤规则：

- [training.py:3876](https://github.com/flagos-ai/FlagScale/blob/0b7b638547c87b8129baec0499f859d96332db8e/flagscale/train/megatron/training/training.py#L3876)：`on_trace_ready` 先保存 `rank-{rank}.json.gz`，再将同一 profiler 的 `p.events()` 和 trace 路径交给报告函数。
- [profiler_reports.py:357](https://github.com/flagos-ai/FlagScale/blob/0b7b638547c87b8129baec0499f859d96332db8e/flagscale/train/megatron/training/profiler_reports.py#L357)：`export_kernel_reports()` 生成三个 CSV；第四个文件来自上述 trace 导出。
- [profiler_reports.py:245](https://github.com/flagos-ai/FlagScale/blob/0b7b638547c87b8129baec0499f859d96332db8e/flagscale/train/megatron/training/profiler_reports.py#L245)：从事件所关联的 kernels 聚合统计，并结合 trace 补充 dtype 和自定义算子的父子关系。

### Cosmos

| 位置 | 已确认行为及对本计划的影响 |
| --- | --- |
| `cosmos_framework/scripts/train.py:249`、`:303` | 正式入口接收 `--sft-toml` 和末尾覆盖项；包装器应继续使用此入口。 |
| `cosmos_framework/configs/toml_config/sft_config.py:1000` | TOML 经校验、Hydra 合并后构造训练配置；不能把仅存在于运行时的字段随意塞进结构化 TOML。 |
| `cosmos_framework/trainer/__init__.py:17`、`:304` | Trainer 通过模块内已导入的 `maybe_enable_profiling` 进入 profiler；临时替换必须作用于该绑定。 |
| `cosmos_framework/trainer/__init__.py:358`、`:373` | 梯度累积完成后才调用一次 `profiler.step()`；采集窗口以 optimizer step 计数。 |
| `cosmos_framework/utils/profiling.py:42` | 现有 handler 只写 Chrome trace，未导出所需 CSV。 |
| `cosmos_framework/utils/profiling.py:59` | 现有调度为 `wait = profile_freq - warmup - active`，未设置 `repeat=1`。 |
| `cosmos_framework/utils/profiling.py:50`、`:64` | 原实现的 `target_ranks` 仅控制导出，所有 rank 仍创建 profiler；本次包装器可仅在目标 rank 开启采集。 |
| `cosmos_framework/utils/config.py:387` | 实际配置字段为 `record_shape`（单数），传入 PyTorch 才是 `record_shapes`。 |
| `cosmos_framework/utils/callback.py:703` | operator 级 `emit_nvtx` 与 Torch profiler 不能同时开启。 |

## 3. 四份交付文件与统计口径

首轮采集 rank 0，在同一运行目录内生成：

| 文件 | 用途 / 字段 |
| --- | --- |
| `rank-0.json.gz` | 完整 Chrome/Kineto trace，保留 CPU op、CUDA kernel、通信、拷贝及关联信息，供复核和定位。 |
| `rank-0_kernel_details_report.csv` | `custom_operator, execution_operator, kernel_name, variant_index, mapping_status, input_shapes, input_dtypes, candidate_operators, kernel_event_count, kernel_time_us` |
| `rank-0_kernel_summary.csv` | `custom_operator, execution_operator, kernel_name, kernel_call_count, kernel_time_us, percent` |
| `rank-0_operator_list.csv` | `operator_id, custom_operator, execution_operator, operator_kind, kernel_name` |

要求：

1. 三份 CSV 采用参考实现的字段顺序、UTF-8 BOM 编码和 CSV quoting；shape/dtype 字段使用 JSON 数组表示。不要用字符串按逗号切分 CSV。
2. details 按自定义算子、执行算子、kernel、shape、dtype 聚合；summary 去掉 shape/dtype 维度并按 kernel 累计耗时降序排列。
3. list 去重的是 `(custom_operator, execution_operator, kernel_name)`；同一 `(custom_operator, execution_operator)` 可有多个 kernel 行并共用 `operator_id`。该 ID 只在本报告内有意义，不是跨运行的固定编号。
4. `kernel_time_us` 为 GPU kernel duration 的累计值；`percent` 的分母是本报告**经过滤且成功归属的计算 kernel 累计时间**。它不是端到端训练耗时占比，也不表示考虑重叠后的关键路径占比。
5. 沿用参考实现对 communication、memcpy/memset、非计算事件和用户 annotation 的过滤；不额外删除低频算子，也不只导出 Top-K。
6. 不根据模型 BF16 配置填充所有算子的 dtype。缺失 shape/dtype 时保持参考实现的空值表示，在质量记录中单列缺失数量；`operator_shape_matched` 是参考脚本的标签，不能据此宣称 dtype 完整或形状已独立验证。
7. 参考脚本的自定义边界名单主要针对 Megatron/TransformerEngine。Cosmos 未识别的边界保留 `custom_operator=null` 和可观测的执行算子/kernel；后端名称的推断也要与真实 ATen 名称区分。若后续补 Cosmos 规则，修改仅放在任务目录内，并记录规则差异与原始证据。
8. 保留 compile/Triton/fused kernel 的实际名称，不能凭名称反推成一组未经观测的 ATen 算子。没有 CPU 归属的 GPU kernel 可能不进入三份 CSV，应在质量记录中列出并保留 trace；不能把 CSV 当作无遗漏的 GPU kernel 清单。

## 4. 用户指定的实际训练负载（2026-09-24 更新）

基准固定为 `examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml:8`。直接读取原 TOML，必要变更通过 CLI 覆盖，保留原文件内容。

| 项目 | 实际选择 |
| --- | --- |
| 模型/精度/attention | Cosmos3-Nano，BF16，two_way |
| 并行 | 单机8张 H100 80GB；FSDP shard=8、replicate=1、CP=1 |
| 长度/GA | UND=3072、GEN=98304、总长度101376、GA=2 |
| Compile/AC | language static compile、mix_order_reduction_split_size=32、full AC |
| 数据 | USR v2 的29份 catalog，保留全部原始采样权重 |
| 条件 | 480p、offline VAE、CFG dropout=0.1、prefix frame 权重5:4:1；不启动在线 VAE，EMA保持关闭 |
| 权重和状态 | 保留原配方的 R7-1056/USR iter12600 assembled DCP，load_training_state=true，恢复 optimizer/scheduler/trainer |
| 窗口 | 从12600恢复，新增25步；等待20步、profiler warmup2步、active3步；采集全局完成 step12623–12625 |
| 输出 | rank0 的四份文件；所有辅助脚本、环境补充包、日志和缓存放任务目录 |

checkpoint全状态恢复12600已实测成功，tokenizer与29份catalog/首行latent均已核验。GPU占用会变化，启动器在每轮开始前检查空闲。29个数据源不可能保证在6个microbatch内全部出现，只记录实际采到的来源/条件，不改变采样权重强行覆盖。

保留训练状态的原因：本轮采集生产配置的实际计算路径，避免重新初始化 optimizer 和重启 LR warmup。profiler schedule 使用本次调用的局部 step，不把12600写入内部 step_num；全局 step 另存 metadata。该决定替代初版从0开始的候选方案。

## 5. 无项目源码修改的接入方式

在任务目录内实现以下独立文件，并分阶段提交实现和记录：

```text
plans/v0.2.3c_kernel_list/
  record.md
  run_profile.py                 # torchrun 启动包装器
  profile_hook.py                # 单次 profiler 上下文、handler
  profiler_reports.py            # 固定版本的报告实现，保留来源信息
  validate_reports.py            # 四文件一致性及覆盖情况核验
  launch_profile.py             # 原始 USR TOML 上的必要覆盖项与启动记录
  requirements-overlay.txt      # 任务环境依赖，不更新系统或项目依赖
  probe_attention.py            # compiled packed varlen前后向数值核验
  references/                    # 上游版本、来源、文件 SHA256
  runs/<run_id>/
    launch.sh
    train.log
    metadata.json
    runtime.json                 # Python/Torch/CUDA/cuDNN/Attention版本
    cache/
    job/                         # config.yaml、框架日志、AOT、最终 checkpoint
    reports/                     # 通过验收的四份文件
    report_quality.json          # 内部质量记录，不增加交付文件种类
```

包装器实施顺序：

1. 确保当前仓库在 `sys.path` 中优先，并核验实际导入的 `cosmos_framework` 来自此工作树。保留系统Torch/CUDA栈，缺失依赖按锁定版本仅补充到任务目录env。实际发现packed varlen需要NATTEN/FA3，已补官方Torch2.12/CUDA13.2对应NATTEN0.21.7，加载checkpoint前检查支持标志并保存runtime.json。
2. 导入 `cosmos_framework.trainer`，保存原始函数绑定；仅在当前进程中把 `trainer.maybe_enable_profiling` 替换成任务目录里的 context manager。仅改 `utils.profiling` 的同名函数不够，因为 Trainer 已经 `from ... import ...`。
3. 通过 `runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")` 进入原 CLI，原样转交 `--sft-toml`、`--dryrun` 和覆盖项；退出时用 `finally` 恢复函数绑定。
4. hook 保留 `config, global_step` 接口。确认本轮从 iteration 12600 恢复；非目标 rank 返回 `None`，目标 rank 开启 CPU+CUDA profiler。Trainer 仍按原逻辑执行 `step()`，不修改训练循环或梯度累积。
5. `on_trace_ready` 先保存 `.json.gz`，再当场调用 `export_kernel_reports(prof.events(), ..., trace_path)`。**必须在 profiler 事件仍可访问时导出；旧 nsys CSV 或仅有 Chrome JSON 不能直接作为该函数的 events 入参。**
6. 四文件先写到该 run 的临时报告目录，校验完整后移至 `reports/`。导出失败保留日志和已生成 trace，记录失败并使该次运行验收失败，不能只留下三个空表而报成功。
7. 禁止目标 rank 的导出分支单独调用 distributed barrier/all-reduce。如需同步导出结束，所有 rank 必须在同一退出位置参与，并保证异常可传播，避免其他 rank 等待超时。

此方案的进程内替换必须在执行前验证命中实际 Trainer；若 recipe 改用其他 Trainer，重新查找其引用位置。原框架导出函数及训练算法均保持原文件内容。

## 6. 单次采集窗口

使用新 job，从原配方 checkpoint 的 iteration 12600 恢复；profiler 独立从局部 step 0 开始：

| 配置 | 值 |
| --- | --- |
| `trainer.profiling.enable_profiling` | `true` |
| `trainer.profiling.enable_nsys` / `enable_memory_snapshot` | `false` / `false` |
| `trainer.profiling.profile_freq` | `25` |
| `trainer.profiling.profile_warmup` / `profile_active` | `2` / `3` |
| `trainer.profiling.target_ranks` | `[0]` |
| `trainer.profiling.record_shape` | `true` |
| `profile_memory` / `with_stack` / `with_modules` | `false` / `false` / `false` |
| 包装器的 schedule | `wait=20, warmup=2, active=3, repeat=1` |
| `trainer.max_iter` | `12625` |

以本次新增的 optimizer step 从1开始计数（全局12601起）：1–20为训练预热，21–22为 profiler warmup，23–25为正式记录（全局12623–12625）。GA=2 时正式记录包含 rank 0 的6个 microbatch，以及3次 optimizer update。日志/注解中的从0开始 iteration 和此处从1开始 step 必须显式对应；不能只把 `prof.step_num` 当作窗口起点。

20步预热是起始方案，不保证任何硬件或新输入都已稳定。若正式窗口仍发生 compile/AOT 初始化、遗漏数据源或 profiler 映射失效，保留失败运行，调整 wait/active 后使用新 run_id 重跑；不覆盖旧证据。保持模型和算法设置，不能为获得更漂亮的算子名默认关闭 compile。确有必要采 eager 辅助清单时另存并标明不同负载。

## 7. 执行步骤

### A. 运行前检查与配置冻结

1. 记录当前 Git SHA、已有工作区变化、训练环境解释器路径、PyTorch/CUDA/driver/CUPTI 可用性、GPU 型号及可用资源。当前调研基线 SHA 为 `872f6bf2888c8899c1599dda8c722e1c551684fa`；执行时重新采集。
2. 解析并核验 USR TOML 中的 assembled DCP、USR v2 29份catalog、离线latent以及本地 tokenizer 路径；离线模式不要求在线VAE权重。记录路径、资产标识和数据配置，不在日志输出 token 或凭据。缺少路径时先补齐；不改用随机数据冒充正式训练负载。
3. 核验8卡资源和 shard/replicate/CP 配置可用，记录实际 world size。不要根据 job 名中的128gpu推断运行规模，也不要因资源不足直接缩成1卡、改变算子分布。
4. 直接读取用户指定 USR TOML，保存第4节覆盖项；独立 job 名和输出目录，避免自动恢复旧 profiling 作业。保留基座权重加载，保留 `checkpoint.load_training_state=true` 并核验实际起始 iteration=12600。
5. 所有写路径指向本次 `runs/<run_id>/`：设置 `IMAGINAIRE_OUTPUT_ROOT`，覆盖 `trainer.compile_cache.path`，把 Triton/Torch 扩展/临时目录导向内部 cache；设置 `PYTHONDONTWRITEBYTECODE=1`。现有模型和数据目录只读复用。VAE 默认把 AOT 产物写到 job 目录（`cosmos_framework/callbacks/compile_tokenizer.py:138`），也应检查外部 `COSMOS_TOKENIZER_AOT_*` 环境变量是否另行指定写路径。
6. 设置 `LD_LIBRARY_PATH=''` 后运行 Python，遵循根目录 `AGENTS.md` 的环境要求。W&B、训练样本预览、validation、nsys 和 operator 级 `emit_nvtx` 关闭；模型计算、优化器、EMA 和 AC 路径保持配方设置。
7. 保存最终 resolved config，核验模型/loader 分辨率、GA、固定长度、compile、checkpoint、profiler 参数及路径。先完成 CLI `--dryrun`，再启动正式训练。
8. 给最终 checkpoint 预留空间。`checkpoint.save_iter` 设大只能排除窗口内定期保存，Trainer 结束仍会保存最后一个 checkpoint（`cosmos_framework/trainer/__init__.py:384`），本计划保留该行为并将其写入内部 job 目录；它不属于交付文件。不要用同时跳过权重加载的 dummy checkpointer 规避保存（`cosmos_framework/checkpoint/dummy.py:15`、`:28`）。

### B. 包装器和报告逻辑验证

在正式分布式任务前，用轻量验证检查：schedule 的一次性回调、非目标 rank 不创建 profiler、正确替换 Trainer 绑定、四文件命名/编码、shape variant 聚合、通信过滤和空报告报错。以 mock 事件验证报表的关键统计关系；这些检查代码只放任务目录内。正式运行前完成训练环境中的 profiler/CUDA 兼容性检查。

### C. 已实现的启动命令

在仓库根目录使用容器解释器执行；原始TOML、环境路径及全部覆盖由启动器冻结到metadata.json。详细环境安装见任务README。

```bash
export LD_LIBRARY_PATH=''
export PYTHONDONTWRITEBYTECODE=1
PROFILE_ROOT="$PWD/plans/v0.2.3c_kernel_list"
/opt/venv/bin/python "$PROFILE_ROOT/launch_profile.py" --run-id preflight_new --dryrun
/opt/venv/bin/python "$PROFILE_ROOT/wait_and_launch.py" --run-id capture_new --probe-attention
```

run_id必须唯一。启动器设置短TMPDIR符号链接解决AF_UNIX路径长度限制，实际临时文件仍在任务run/cache/tmp；结束移除短链接。每rank的Inductor/Triton缓存分开，避免8卡同时写一个编译缓存。`launch.sh`为审计快照，复现应重新运行启动器创建短链接。

### D. 运行过程记录

在 `record.md` 追加：启动时间、run_id、命令/配置位置、主进程及 rank PID、设备分配、checkpoint 加载结果、初始化/AOT 完成情况、正式采集起止 step、导出耗时、训练结束/退出码。每到一个阶段或出现异常及时记录；长时间运行补充当前 iteration 和日志位置。

发生 OOM、CUPTI 缺失、CUDA events 为空、导出失败或数据不足时，写清错误原文、失败阶段和修复动作；不得修改任务目录外代码。若需要调整窗口或环境，保留原 run 并用新 run_id。只管理本次任务创建的进程。

## 8. 验收与交付

1. **负载真实**：基座权重加载成功、loss 有限、预定窗口跑满，forward/backward/optimizer 及配方启用的路径（本次offline VAE、EMA关闭）有运行证据；核对正式窗口的实际数据源和输入形状。不能只以训练退出码为0验收。
2. **四文件有效**：同一 run、rank、active 窗口；gzip 可解压、JSON 含非空 CPU/CUDA 事件；CSV 非空且表头、编码正确。
3. **统计一致**：按 `(custom_operator, execution_operator, kernel_name)` 汇总 details，调用次数与 summary 严格一致，耗时允许三位小数舍入误差；operator list 的映射集合与 summary 一致且无重复三元组。percent 按参考分母复算，容许格式化误差和 `<0.001%` 表示。
4. **归属质量**：记录 trace 中 GPU kernel 数量/耗时、通信与拷贝过滤、已归属计算 kernel、未归属 kernel、缺失 shape/dtype 数量及占比。审计未归属或疑似重复归属的热点，抽查 GEMM、attention、elementwise、VAE/optimizer kernel 与 CPU op 的对应关系。不要求原始 GPU kernel 数与过滤后的 CSV 数天然相等。
5. **缺失可解释**：若 CUDA 活动为空、主要热点未被清单覆盖、窗口包含初始化或丢失预期训练阶段，本轮不标记成功；先修复采集或有依据的映射规则再导出。编译后无法恢复的内层语义明确写为覆盖边界，不补造算子或 shape。
6. **可复现**：保存 resolved config、命令、版本、种子、并行度、输入配置、rank/window、参考脚本版本、四文件大小和 SHA256；质量指标写入 `report_quality.json`，摘要写入 `record.md`。
7. **改动边界**：与运行前 Git 状态对比，新增修改仅在指定计划文件和任务目录；已有无关工作区变化保持原样。
8. **交付**：向用户提供通过验收的四份文件路径及其 workload/rank/window/过滤口径说明，由用户交给算子组。暂不自动发送外部消息，也不提前筛掉算子组可能需要的低频计算算子。

## 9. 执行清单

- [x] 阅读仓库入口、profiler 和 FlagScale 参考，确认四文件定义。
- [x] 编写本计划，建立 `record.md`。
- [x] 确定实际 recipe、训练环境、GPU、数据/checkpoint/tokenizer 路径（GPU准入动态检查）。
- [x] 仅在任务目录内实现包装器、报告导出和校验脚本。
- [x] 完成配置 dryrun、7项调度/报告逻辑检查及小型 CUDA profiler 兼容性检查。
- [ ] 运行一次有效的正式 profiling，持续记录中间过程。
- [ ] 验证四份文件、归属质量及配置覆盖范围，交付路径与摘要。
