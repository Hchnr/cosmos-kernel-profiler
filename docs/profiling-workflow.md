# Cosmos Kernel Profiler：详细采集流程

独立保存 Cosmos 训练算子清单的采集实现、验证工具和历史证据。该仓库是本地 Git 仓库，没有配置远程，也未发布。原 Cosmos 工作树和 `plans/` 文件保留原样。

**当前状态：正式 USR 采集尚未完成，不能把 evidence 中的小型探针报告作为正式交付。用户已要求等待 GPU 资源通知，本次整理不启动训练、GPU 检查或资源等待器。**

## 内容

- `launch_profile.py`、`run_profile.py`、`workspace.py`：读取外部 Cosmos checkout 的原始 TOML，以覆盖参数启动8卡训练；通过进程内替换接入 profiler，不修改框架源码。
- `profile_hook.py`、`phase_audit.py`、`profiler_reports.py`：单窗口采集、训练阶段标记、FlagScale 四文件导出。
- `validate_reports.py`、`validate_run.py`：表格统计、trace 覆盖及完整训练窗口验收。
- `checkpoint_compat.py`：Python3.12读取较新Python写入的Path checkpoint元数据。
- `probe_profiler.py`、`probe_attention.py`、`test_*.py`：小型CUDA探针、编译Attention数值验证和CPU单元检查。
- `wait_and_launch.py`：资源等待工具，只有手动执行才会运行；空闲检测不等于独占资源预留。
- `docs/`：迁移前的计划、过程记录和启动说明历史快照。
- `evidence/`：已选取的中间结果、失败日志、配置、来源清单及本次迁移核验，详见 [证据说明](evidence/README.md)。

模型权重、数据集、安装环境、编译缓存与训练checkpoint没有复制进仓库。今后的 `env/`、`env-cache/`、`runs/`、`validation/` 默认忽略；需要归档的结果经选择后放入 `evidence/`。

## 目标负载

外部原始配方：`examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml`。

保留Nano/BF16/two_way、UND3072/GEN98304、GA2、480p/offline VAE、29份USR数据及原权重、language static compile/full AC、完整训练状态恢复12600。必要覆盖为单机8卡、DP shard8/replicate1/CP1、关闭外部上传和验证、隔离输出，以及20步等待+2步warmup+3步active。正式采集目标为rank0、完成step12623–12625。

历史验证基础栈为Python3.12、容器Torch2.12.0a0/CUDA13.2/cuDNN9.21、NATTEN0.21.7对应wheel。最后一次等待器的预检因环境中无法导入Torch而退出；环境可能发生变化，恢复采集前必须重新核验解释器和依赖。本次迁移的CPU验证结果单独记录，不能替代训练/GPU验收。

## 使用（收到GPU可用通知后再启动）

脚本通过 `COSMOS_REPO` 定位框架，不再依赖本仓库位于 `plans/` 下。以下命令在本仓库根目录执行，Python应选择与原生Attention扩展匹配的训练环境。

```bash
export COSMOS_REPO=/share/project/eai_pwm/home/hcr/worktrees/cosmos-framework-pwm-load-balance
export LD_LIBRARY_PATH=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/env:$PWD:$COSMOS_REPO"

# 仅安装隔离overlay；不会替换基础Torch，不能直接用于其他Torch/CUDA ABI。
UV_CACHE_DIR="$PWD/env-cache" uv pip install --target "$PWD/env" --no-deps -r requirements-overlay.txt

python launch_profile.py --run-id preflight_new --dryrun
python launch_profile.py --run-id capture_new
# 或收到资源通知后手动启动等待器：
# python wait_and_launch.py --run-id capture_new --probe-attention --idle-observations 11
```

每次必须使用新的run_id。运行输出放本仓库 `runs/<run_id>/`，资源等待及独立探针放 `validation/`。启动器保留原模型/数据的共享路径；在其他机器复现需要相同资产或明确修改相应配置。metadata同时记录框架和采集工具两份Git SHA，记录脚本指纹、原配方指纹、命令和受控环境变量。短TMPDIR链接只用于满足UNIX socket路径限制，临时文件仍位于本仓库run目录。

CPU检查及报告验收：

```bash
mkdir -p validation
TMPDIR="$PWD/validation" python test_reports.py -v
python test_hook.py -v
python validate_reports.py evidence/validation/cuda-probe-001/reports --output validation/toy_quality.json
# 正式训练成功退出后：
# python validate_reports.py runs/capture_new/reports --output runs/capture_new/report_quality.json
# python validate_run.py runs/capture_new
```

## 四文件与边界

有效run的 `reports/` 包含 `rank-0.json.gz`、`rank-0_kernel_details_report.csv`、`rank-0_kernel_summary.csv`、`rank-0_operator_list.csv`。上游FlagScale来源、固定SHA及许可证在 [references/README.md](references/README.md)。

CSV排除通信和拷贝等事件；trace保留原始活动。时间百分比以已归属计算kernel累计时间为分母，并非端到端训练耗时占比。缺失shape/dtype保持空值，编译融合后的kernel不能凭名称还原未经观测的ATen操作。单rank短窗口仅代表实际执行到的变体。只有表格、训练完整性和归属质量均核验后才能标记为正式交付。

## 恢复执行的环境变更（2026-09-24）

当前基础解释器已切换到Python3.13/Torch2.10.0+cu130/CUDA13.0，使用 `requirements-overlay-cu130-torch210.txt`，其中FA3为框架uv.lock固定的1.0.3+cu130.torch210 ABI3 wheel。不要在此环境安装上文旧的Python3.12/Torch2.12 NATTEN wheel。继续执行的记录见 [resumed_execution.md](resumed_execution.md)。基础环境不同，需先重新检查ABI和局部数值验证，再启动正式训练。
