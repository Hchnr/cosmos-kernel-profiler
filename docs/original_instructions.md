> 历史快照：记录迁移前的目录、命令及已移除的本地提交。当前用法和状态以仓库根目录 README.md 为准。

# USR 算子列表采集

本目录通过原训练入口读取用户指定的 USR TOML，在8卡、DP shard=8下运行一次有限窗口的PyTorch profiler。项目训练代码没有文件修改；实现细节及问题记录见 [record.md](record.md)。

## 复现

在仓库根目录执行。基础环境为容器Torch 2.12.0a0 / CUDA13.2 / cuDNN9.21；`requirements-overlay.txt`补充Python依赖及该ABI的官方NATTEN0.21.7 wheel，不安装或替换Torch。USR packed varlen路径需要NATTEN或FA3，现有FA2/cuDNN在项目中不支持此路径。

```bash
export LD_LIBRARY_PATH=''
export PYTHONDONTWRITEBYTECODE=1
PROFILE_TASK="$PWD/plans/v0.2.3c_kernel_list"
UV_CACHE_DIR="$PROFILE_TASK/env-cache" uv pip install \
  --target "$PROFILE_TASK/env" --no-deps -r "$PROFILE_TASK/requirements-overlay.txt"

python "$PROFILE_TASK/launch_profile.py" --run-id preflight_new --dryrun
python "$PROFILE_TASK/launch_profile.py" --run-id capture_new
```

资源被占用时可使用`wait_and_launch.py --run-id capture_new --probe-attention`：连续两次空闲后，先执行compiled varlen Attention的前向/梯度数值检查，再启动8卡训练；证据保存在`validation/<run_id>-attention/`。

若有其他作业反复启动，可加`--idle-observations 11`要求连续5分钟空闲。空闲检测不能提供集群独占锁；正式运行期间若出现另一组作业，应记录冲突并停止本次自己的进程，避免把争用窗口作为性能统计交付。

每次必须使用新的run_id。启动器拒绝覆盖旧目录，正式运行前检查当前8卡无其他计算进程；实际命令、受控环境变量、源TOML和脚本SHA256保存在该run的`metadata.json`及`launch.sh`中。

- 基准：`examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml:8`。
- 保留：USR29份catalog及其权重、offline VAE、480p、CFG 0.1、条件权重5:4:1、EMA关闭、UND3072/GEN98304、GA2、language static compile/full AC、full-state resume12600。
- 覆盖：shard8/replicate1/CP1、关闭外部日志上传、隔离输出/缓存、25个新增optimizer steps、rank0 profiler20/2/3窗口。
- 兼容处理：`checkpoint_compat.py:9`在Python3.12进程内解析较新Python写入的Path pickle模块名；不修改原checkpoint。
- `--wait`、`--warmup`、`--active`可调整窗口，max_iter随之更新；恢复起点固定12600并由hook断言。要换checkpoint须显式修改实现/配置并重新验收，不能误把其他checkpoint的清单标成当前结果。

## 验证

CPU局部检查使用unittest，避免仓库pytest fixture自动占用GPU：

```bash
LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR="$PROFILE_TASK/validation" python "$PROFILE_TASK/test_reports.py" -v
LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 \
  python "$PROFILE_TASK/test_hook.py" -v
```

在没有训练运行时可先执行小型CUDA探针：

```bash
python "$PROFILE_TASK/probe_profiler.py" "$PROFILE_TASK/validation/cuda_probe_new"
```

正式任务结束后，独立重读落盘文件验收：

```bash
PROFILE_CAPTURE="$PROFILE_TASK/runs/capture_new"
python "$PROFILE_TASK/validate_reports.py" "$PROFILE_CAPTURE/reports" \
  --output "$PROFILE_CAPTURE/report_quality.json"
python "$PROFILE_TASK/validate_run.py" "$PROFILE_CAPTURE"
```

`validate_reports.py`检查CSV字段、聚合次数/时间/百分比、operator ID，以及原始trace的kernel覆盖；`validate_run.py`核验所有rank的起止step、GA microbatch数量、阶段注解及日志中的有限loss，并汇总实际采到的数据源、条件和latent形状。**表格验证通过并不自动表示完整训练或归属质量通过**，仍须阅读两个质量文件及记录中的验收结论。

## 交付口径

每个有效run的`reports/`只包含：

1. `rank-0.json.gz`
2. `rank-0_kernel_details_report.csv`
3. `rank-0_kernel_summary.csv`
4. `rank-0_operator_list.csv`

三份CSV由固定版本FlagScale导出，来源及许可证见 [references/README.md](references/README.md)。CSV排除通信/拷贝等非计算事件，原始trace保留这些事件。`percent`以已归属计算kernel累计时间为分母；shape/dtype无法取得时留空并计入质量记录，不能从模型BF16设置推填。单rank、短窗口只代表实际执行到的变体。

内部还保留原始events简化快照、resolved config、每rank microbatch记录、日志、缓存和最终checkpoint；这些不是算子组的四份核心交付。大型运行产物未加入Git，最终报告摘要和manifest用于review。
