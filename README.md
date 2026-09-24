# Cosmos Kernel Profiler

Cosmos Kernel Profiler 是一个面向 Cosmos 训练任务的算子与 kernel 分析工具集。它围绕训练过程中的性能数据采集、结果整理和质量验证展开，帮助使用者了解实际执行了哪些计算 kernel，以及这些 kernel 对应的算子、形状和数据类型等信息。

这个仓库独立于 Cosmos 框架源码工作：通过外部框架 checkout 和启动配置运行目标任务，并在进程内接入 profiler。因此，分析工具和历史结果可以单独维护，不需要把实验性改动直接放进训练框架。

## 能力概览

- 在受控训练窗口中采集计算 kernel 和原始 trace。
- 标记训练阶段，区分预热、采集和其他运行区间。
- 将 profiler 输出整理为便于查看和后处理的报告文件。
- 检查报告覆盖范围、字段质量和训练运行完整性。
- 提供 checkpoint 兼容处理及 Attention、CUDA 等局部探针。
- 保存配置、运行元数据和验证证据，支持复核分析过程。

## 项目结构

- `launch_profile.py`、`run_profile.py`：准备并启动 profiling 任务。
- `profile_hook.py`、`phase_audit.py`、`profiler_reports.py`：采集、阶段标记和报告生成的核心逻辑。
- `validate_reports.py`、`validate_run.py`：报告和运行结果的验证工具。
- `probe_*.py`、`test_*.py`：局部运行探针和自动化检查。
- `checkpoint_compat.py`、`workspace.py`：运行环境和 checkpoint 相关辅助逻辑。
- `docs/`：详细流程、迁移记录和历史说明。
- `evidence/`：经过选择的运行结果、配置和验证证据。
- `references/`：上游工具来源与许可证信息。

## 基本工作方式

项目通常依赖一个外部 Cosmos 源码 checkout，以及与目标训练任务匹配的 Python、PyTorch、CUDA 和扩展环境。启动器通过 `COSMOS_REPO` 定位外部源码，运行前生成独立的运行目录，随后将采集结果和元数据写入该目录。

```bash
export COSMOS_REPO=/path/to/cosmos-framework
python launch_profile.py --run-id <run-id> --dryrun
```

完成环境和资源确认后，再按目标任务启动正式采集。详细参数、环境要求、报告格式和验收规则见 [详细采集流程](docs/profiling-workflow.md)。

## 当前状态

已完成原USR配置、DP shard=8的正式采集 `resumed_capture001`，并通过训练完整性、四文件一致性及kernel覆盖验收。结果和四份文件见 [正式采集结果](evidence/captures/resumed_capture001/README.md)。历史小型探针仍保留在evidence/validation，勿与正式报告混用。

## 说明

模型权重、数据集、训练 checkpoint、编译缓存和完整运行环境不随仓库分发。复现实验需要准备相应资产，并根据机器环境调整外部源码路径和启动配置。
