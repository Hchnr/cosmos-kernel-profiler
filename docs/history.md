> 历史快照：记录迁移前的目录、命令及已移除的本地提交。当前用法和状态以仓库根目录 README.md 为准。

# v0.2.3c kernel list 过程记录

## 当前状态

2026-09-24 实施中：独立包装器、四文件导出、离线验收、7项CPU检查、小型CUDA探针及NATTEN compiled前后向数值检查均已通过；真实USR已成功恢复step12600。已修复checkpoint路径pickle、IPC长路径和varlen后端缺失。usr_capture005因准入后出现另一组8卡作业而停止，己方进程已全部清理；waiter PID396928等待连续5分钟空闲后启动usr_capture006。**正式USR四份交付文件尚未生成；小型CUDA探针不作为最终交付。**

计划：`plans/v0.2.3c_kernel_list.md:1`。
工作目录：`/share/project/eai_pwm/home/hcr/worktrees/cosmos-framework-pwm-load-balance`。
允许新增/修改范围：上述计划文件和 `plans/v0.2.3c_kernel_list/`。

## 2026-09-24：调研与方案编写

- 读取根目录 `AGENTS.md`，使用 `.agents/skills/cosmos3-codebase-nav/SKILL.md` 核对训练调用链。
- 调研时仓库 HEAD：`872f6bf2888c8899c1599dda8c722e1c551684fa`。
- 初始 `git status --short` 只有 `?? .venv-odc-v010`。该条目是原有 symlink，指向 `/tmp/cosmos-fsdp-odc-v010-8af4599f`；本次未修改。`readlink -e` 未解析出有效路径，因此后续不能直接假定该环境可用。
- 只读访问用户提供的 FlagScale 页面，并查询上游 `main` 的 SHA：`0b7b638547c87b8129baec0499f859d96332db8e`。计划按该 SHA 固定来源。
- 上游 [training.py:3876](https://github.com/flagos-ai/FlagScale/blob/0b7b638547c87b8129baec0499f859d96332db8e/flagscale/train/megatron/training/training.py#L3876) 先导出 trace，再调用报告函数；[profiler_reports.py:357](https://github.com/flagos-ai/FlagScale/blob/0b7b638547c87b8129baec0499f859d96332db8e/flagscale/train/megatron/training/profiler_reports.py#L357) 只写三个 CSV。合计四份，已在计划列明完整文件名和表头。
- 本仓库 `cosmos_framework/utils/profiling.py:42` 只有 trace 导出；`cosmos_framework/trainer/__init__.py:304` 使用模块内导入的 profiler 上下文，`:373` 每个 optimizer step 推进一次。
- 方案：任务目录内的启动包装器临时替换 Trainer 的 profiler 绑定，调用原训练 CLI；在 `on_trace_ready` 内同时使用 `prof.events()` 与 trace 生成三个 CSV，不改项目源码。
- 默认候选为 `examples/toml/sft_config/vision_sft_nano_t2v_8gpu_fixed_static.toml:17`。当前固定长度已是1086/64450、GA2、online VAE、language static compile。其文件名/job 名不能证明实际分辨率或并行规模；基础模型配置为720，TOML 的 VAE 预热为256。计划显式派生480p配置，并要求 dryrun 核验最终模型和 loader。
- 初始窗口拟定 `wait=20, warmup=2, active=3, repeat=1`，总25个 optimizer steps，仅采 rank 0；正式窗口为完成 step 23–25，GA2 对应6个 microbatch。
- 确认结束时仍会写最终 checkpoint（`cosmos_framework/trainer/__init__.py:384`），计划把该产物放到任务 run 的内部 job 目录，并预留空间。
- 调研用到只读 `rg`、`nl`、`git status/log/rev-parse`、`curl`、网页读取等；没有导入训练环境做运行验证，没有进行 GPU profiling。
- 文档检查通过：代码引用的文件/行号存在，Markdown 代码围栏闭合，启动模板通过 `bash -n` 语法检查。此检查不代表训练命令已运行或环境已验证。
- 两份文档已落盘；`plans/` 命中现有 Git ignore 规则，因此普通 `git status` 不列出它们。本次未更改 ignore 规则或强制加入暂存区，项目跟踪文件无 diff。

## 待执行前核实

- 实际目标 recipe 是否采用计划中的 Nano T2V 候选，以及8卡资源和并行拓扑。
- 可用训练 Python 环境、PyTorch/CUDA/CUPTI 兼容性。
- checkpoint、VAE、tokenizer、两份 catalog 的实际可读路径。
- 最终配置、所有输出/缓存目录、数据源覆盖与 compile/AOT 稳态条件。

## 后续每次运行记录模板

### `<UTC时间> / <run_id> / <阶段>`

- 目的与当前状态：
- Git SHA / 环境 / GPU / PID：
- 完整命令或 `launch.sh` 路径：
- 配置、数据、checkpoint 及其标识：
- 日志位置与当前 iteration：
- 正式窗口 / rank / 实际 microbatch 数：
- 错误、处理和是否重跑：
- 四文件路径 / 大小 / SHA256：
- CSV 一致性、缺失元数据与未归属 kernel：
- 退出码 / 最终验收结论 / 仍有的覆盖边界：

## 2026-09-24：实施启动与保守决策

- 用户指定 USR v2 `vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml`，授权必要覆盖、dp shard=8、自主实施验证和频繁提交。分支 `hcr_0924/kernel_list`，初始跟踪文件无diff。
- 8张H100 80GB均无计算进程。默认checkpoint、Qwen3-VL-8B-Instruct tokenizer及USR v2根目录均存在；共享盘约22TB可用。
- 保留offline VAE、EMA=false、3072/98304、GA2、compile和29份原采样权重；覆盖为单机shard8/replicate1/CP1及有限采集窗口。
- 保留full-state resume12600；max_iter12625；局部profiler schedule20/2/3，全局active12623–12625。避免改变optimizer/LR状态。
- 系统Python为/opt/venv/bin/python，Torch 2.12.0a0+0291f960b6.nv26.04.48445190 / CUDA13.2，可见8卡；缺少hydra-core。使用env-troubleshoot技能，补充依赖仅安装到任务目录env，不更新系统Torch或项目依赖文件。
- plans目录原被忽略；用户已要求提交，因此仅对本任务实现/文档逐个git add -f，原ignore规则和无关symlink保持原样，大型运行产物不加入Git。

### 独立包装器与报告实现

- 已固定并原样保存 FlagScale 导出脚本，SHA256=a147184fe5c172eb06e26c18825db89b60f2316c7d1c2c133aeb44e027d8dd29，来源和许可证保留在references。
- 新增run_profile/profile_hook/phase_audit/launch_profile/validate_reports，分别负责训练入口绑定、单窗口导出、阶段注解、隔离启动、表格及原始trace覆盖核验。额外保存轻量events快照，便于原样离线复算，不依赖重跑GPU。
- 使用原有LoadBalanceTrace回调记录实际microbatch/条件/形状，关闭其CUDA timing以避免额外同步；只有rank0开启Torch profiler，其他rank正常训练。
- CPU unittest 3项通过：shape variant与通信过滤、损坏调用次数拒绝、空CUDA trace拒绝。脚本经定向Ruff修正和格式化；未调用项目pytest GPU fixture。
- 已按uv.lock在任务目录隔离安装Hydra/OmegaConf/Transformers4.57.6/Diffusers0.39.0等所需包。配置导入先后发现portalocker及multi-storage-client缺失，继续补齐；不更新项目外环境。

### CUDA兼容性与资源/目录检查

- 小型真实CUDA探针通过，位于validation/cuda-probe-001：恢复起点12600，局部5步中的2步active，成功导出trace和3 CSV。24次报告计算kernel与原始trace24次一致，16种shape/operator/kernel变体、9个operator分组；4次kernel调用的dtype缺失被明确记录。
- 阶段核验只数CPU user_annotation，避免把Kineto的GPU镜像annotation重复计数；此探针实际forward/backward/optimizer各2次。
- 新增4项CPU hook检查全部通过：局部20/2/3单次schedule、非目标rank不创建profiler、错误恢复step拒绝、非法窗口拒绝。加上报告检查共7项通过。
- USR全部29个catalog首个Parquet元数据检查通过，五列schema齐全，采样权重和4246463；明细validation/catalogs.json。
- preflight001失败于系统huggingface-hub1.9.2与Transformers4.57.6不兼容，任务环境补齐锁定0.36.2；preflight002继续到catalog构建时发现LeRobot缺失，按uv.lock补齐0.4.4。所有失败日志保留在各run目录。

### 正式启动前配置验收

- preflight003返回0，原训练CLI dryrun成功；resolved config已核验shard8/replicate1/CP1、UND3072/GEN98304、480p、offline VAE、EMA=false、GA2、full-state resume和max_iter12625。摘要resolved_settings.json。
- 所有29数据源及原权重保留；本次不要求6个microbatch覆盖全部来源，仅审计实际窗口。环境overlay共19个包已锁定requirements-overlay.txt，CUDA基础栈未更换。
- 将启动usr_capture001：先20步训练预热，再2步profiler warmup，采集全局完成step12623–12625，rank0。日志与metadata均在runs/usr_capture001。

### usr_capture001：checkpoint元数据兼容性失败及修复

- 首次8卡任务在加载model之前失败，未完成任何训练步，返回码1；torchrun已清理全部rank，nvidia-smi确认无残留计算进程。
- Torch DCP对read_metadata异常做了捕获，外层表现为AssertionError metadata is None。单独读取揭示原因为Python3.12无法导入checkpoint pickle中的pathlib._local（较新Python的Path模块位置）。
- 新增任务目录checkpoint_compat.py：仅在Python<3.13、该模块不存在时为当前进程安装pathlib._local→pathlib别名，并在退出时恢复；不改checkpoint、系统Python或项目源码。
- CPU实读model/optim/scheduler/trainer四份metadata成功，分别804/6075/6/769项。另用DCP实读trainer.iteration确认12600，pickle路径往返与别名恢复通过。
- 新增validate_run.py，将在训练结束时核验8个rank完整50microbatch、采集窗口6个microbatch、6/6/3阶段注解、active loss有限以及实际来源/条件。
- 修复后使用新目录usr_capture002重跑，保留首次失败日志。

### usr_capture002：恢复成功，修复IPC路径长度

- 完整model/optimizer/scheduler/trainer已成功加载，日志明确iteration12600。随后DataLoader worker共享tensor时AF_UNIX path too long，尚无完整训练step。
- 根因是严格放入任务目录的TMPDIR绝对路径超过Linux UNIX socket上限。修复为/tmp/ckl-<随机名>/tmp符号链接到run/cache/tmp；仅创建短路径入口，临时文件仍在指定任务目录，结束时移除短链接。
- 真实AF_UNIX Listener通过短链接绑定验证成功，地址36字节；代码未改动项目其他位置。launch.sh是命令审计快照，其中临时短链接需由launch_profile.py重新创建，复现使用新run_id。
- 精确核对torchrun PID286466及进程组后SIGTERM；记录89个后代的PID和创建时间，退出后全部清理，无残留CUDA上下文。stop_record.json保留原因及清理结果；未处理其他进程。
- 新run_id=usr_capture003，模型和采样配置保持不变。

### usr_capture003：资源准入拦截，继续离线核验

- 07:58–08:01 UTC，8卡出现其他PID namespace中的计算进程，显存从约6.7GB增长到38GB；本容器已无任何run_profile.py进程，前一任务记录的89个后代全部退出。未干扰外部进程。
- usr_capture003在启动torchrun前被GPU占用检查拒绝，没有启动GPU任务；metadata标为resources_busy/launched=false。修复准入先于临时链接创建，并清理该次遗留的短链接。
- 在等待期间继续CPU只读验证：29份catalog首条样本的离线latent文件全部可读，safetensors tensor_key存在且header shape与catalog声明一致。证据validation/latent_headers.json。
- 验收增加active窗口中Dynamo编译区间检查，避免把冷编译当作稳态算子清单；复现README已写入。

### 08:03 UTC：资源释放，自动恢复执行

- 新增wait_and_launch.py，以30秒间隔只读观察GPU进程，连续两次空闲才交给原启动器再检查并启动，期间不分配CUDA资源。观察日志validation/resource_wait.jsonl。
- 08:03:21 UTC检测到首次完全空闲，随后继续第二次检查；下一次正式run为usr_capture004。资源问题已自行解除，无需更换节点。

### usr_capture004：8卡已恢复并进入计算

- 08:03:51 UTC经两次空闲观察和启动器复核开始，torchrun PID324388；原USR TOML、dp shard8和full-state设置保持一致。
- checkpoint完整恢复12600，online VAE初始化跳过；8卡已进入首步计算/compile，IPC长路径错误未再出现。正式采集仍在等待20+warmup2之后。

### usr_capture004：变长Attention后端缺失

- 08:05 UTC首步失败，返回码1；首个microbatch未完成。异常来自`cosmos_framework/model/generator/mot/attention.py:257`的causal varlen调用，没有兼容后端。torchrun已退出并清理rank。
- 实测容器Torch2.12.0a0/CUDA13.2/cuDNN92100，FA2和cuDNN可导入，但项目`flash2/checks.py`及`cudnn/checks.py`都拒绝varlen；FA3/NATTEN未安装。不是cuDNN版本不足，也不需要关闭compile。
- 按[NATTEN官方wheel索引](https://whl.natten.org/)隔离安装`natten-0.21.7+torch2120cu132-cp312-cp312-linux_x86_64.whl`，未修改Torch/系统环境。导入成功，`NATTEN_SUPPORTED=True`。版本及URL加入requirements-overlay.txt；安装日志validation/install-natten.log。
- 新增probe_attention.py：对BF16/GQA、causal等长及noncausal异长的packed varlen调用执行fullgraph compile前后向，并与逐样本FP32显式softmax参考比较输出与梯度；随后检查缓存路径。先验证原生扩展兼容性，避免反复加载完整checkpoint。
- 当前8卡再次出现外部任务；waiter只读等两次空闲，再执行数值探针和usr_capture005，不触碰外部进程。仍保留UND3k/GEN96k、GA2和原有编译设置。

### 环境预检与文档同步

- run_profile.py在加载checkpoint前检查FA3/NATTEN支持；rank0保存runtime.json，记录Python、Torch、Triton、NATTEN、FA2、Transformers、HFhub、CUDA和cuDNN版本。
- preflight004再次完整CLI dryrun成功（returncode0），确认上述环境记录实际生成；7项CPU检查再次通过。
- 计划中的占位启动模板已替换为真实launcher/waiter用法，修正TMPDIR短链接、配置落盘路径、任务内依赖安装和已完成清单，避免review时照旧模板运行。

### NATTEN编译数值验证通过

- 在GPU7空余显存>32GiB时，执行仅检查正确性、不采性能数据的微型probe；PyTorch张量allocator上限设为显存2.5%。此例共享GPU是为了推进环境验证，正式训练仍要求8卡空闲。
- `validation/attention-001/result.json`：causal packed GQA和noncausal异长packed GQA的fullgraph编译前向、反向及缓存路径均通过。与FP32显式softmax参考相比，前向最大绝对误差分别0.0082207和0.0044644；三路梯度均通过atol/rtol0.04检查。
- probe退出码0。08:15 UTC左右外部任务也已释放全部GPU，waiter将继续两次空闲确认并恢复正式运行。

### usr_capture005：依赖修复后的正式重跑

- 08:15 UTC仅捕获一次空闲，第二次观察前外部8卡任务再次启动，waiter未抢占。资源观测持续保存在resource_wait.jsonl。
- 08:19:12/08:19:42 UTC连续两次空闲后，独占状态的Attention数值探针再次通过（validation/usr_capture005-attention/result.json），继而启动正式torchrun PID379132。
- 使用提交c8dc76c8对应实现，NATTEN0.21.7，其余训练和采样设置保持不变；日志runs/usr_capture005/train.log。

### usr_capture005：准入后出现另一组8卡作业，主动止损

- 本次checkpoint成功完整恢复12600；约08:20:50 UTC的nvidia-smi出现16个GPU计算进程，说明另一组8卡作业在本次准入后启动。正式训练尚未完成首步。
- 08:21:43 UTC核对torchrun379132命令行及其129个后代的PID/创建时间，记录stop_record.json后只对本次torchrun发送SIGTERM；等待框架退出及清理，未终止其他作业。
- 增加waiter的`--idle-observations`参数。下一轮usr_capture006要求11次、每30秒的连续空闲观察（首尾间隔5分钟），避开外部作业反复启动的短间隙；数值探针通过后仍由launcher重新检查资源。
- 这是资源冲突导致的中止，不能计为一次完整模型训练通过。四文件正式交付仍待完成。
- usr_capture005最终returncode1（SIGTERM）；记录的130个进程均退出，宽限期后无需强杀。日志中的DataLoader Terminated/InductorError都出现在主动停止之后，不作为新增训练根因。08:22 UTC启动usr_capture006等待器PID396928。

### 用户要求：plans内容全部保留本地，不进入Git提交

- 按最新指示，将本次14个仅涉及plans/的本地提交从当前分支历史移除，分支回到任务开始前的872f6bf2888c8899c1599dda8c722e1c551684fa。使用mixed reset，未删除工作区文件。
- 对此前跟踪的22个文件逐一比较SHA256，重置前后全部一致；git ls-files plans/为空，既有plans忽略规则生效。核验记录validation/git_untrack.json。
- 后续plans下的脚本、计划、运行产物和过程记录均只保留本地，不再git add或commit。先前记录中的提交号仅作为历史执行审计信息。
- 本次Git整理不停止已授权的资源等待任务。
- 状态复核：此前等待器已于5分钟空闲后进入usr_capture006的Attention预检，但随后因ModuleNotFoundError: No module named 'torch'退出，未启动正式训练；日志validation/usr_capture006-attention.log。Git整理没有重启作业。
