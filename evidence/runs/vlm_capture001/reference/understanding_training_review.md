# Understanding 训练代码 Review 指南

代码分支：`ybai_0828/understanding_verification`。

## 代码入口

- 训练命令入口：[`cosmos_framework/scripts/train.py`](../cosmos_framework/scripts/train.py)。通过 `--sft-toml` 指定配置，额外的 Hydra 覆盖参数放在 `--` 之后。
- TOML 配置定义与转换：[`sft_config.py`](../cosmos_framework/configs/toml_config/sft_config.py) 和 [`toml_config_helper.py`](../cosmos_framework/configs/toml_config/toml_config_helper.py)。
- Loss 计算语义：[Understanding loss 归一化说明](understanding_loss_normalization.md)。
- 完整梯度累积窗口的归一化：[`trainer/loss_normalization.py`](../cosmos_framework/trainer/loss_normalization.py)。
- Packing 与 map-style 数据恢复：[`data/generator/dataflow/loader.py`](../cosmos_framework/data/generator/dataflow/loader.py)。
- HF 权重加载：[`safetensors_loader.py`](../cosmos_framework/model/generator/utils/safetensors_loader.py)。

## 生产训练参考（历史存档，不是可直接迁移的启动脚本）

成功完成的 Grounding100K 训练对应以下存档：

- [实际使用的正式训练 TOML](../examples/understanding_review/grounding100k_four_node_ga3.toml)。
- [实际使用的 node-0 启动命令与环境变量](../examples/understanding_review/grounding100k_node0_command.json)。

启动命令包含必要的 HF 初始化覆盖参数：仅将 DCP `load_path` 留空，并不能保证加载预期的初始权重。本次训练从原始 Qwen3-VL-8B-Instruct 初始化，同时训练 LM 和 ViT，使用按原始样本等权、跨完整梯度累积窗口归一化的 sample-mean loss。四机 × 每机八个 rank × GA3，对应每次更新 96 个 pack，而不是 96 条原始样本。

这些快照为方便 review，保留了历史绝对路径、运行标识和输出位置，**请勿原样执行**。外部审计／覆盖统计回调 `night8_callback.Formal`、其 run plan 以及协调器依赖未包含在这份参考中。该回调在确认达到样本覆盖边界后请求安全结束训练。此实验的 scheduler 计划预算为 25 步，实际在第 24 步结束。框架中的 epoch 监督限制负责屏蔽后续 epoch 对齐填充样本的监督；仅靠此限制，不能证明覆盖完整，也不能确定正确的停止预算。

启动新实验时，必须使用新的配置、输出目录、W&B 标识、rendezvous ID 和端口，并核对初始化覆盖参数、数据版本以及对应拓扑的训练预算。每个节点都需要设置自己的 node rank。保留已验证的启动环境，包括 `RLIMIT_NOFILE=65536`。没有配套且经过审计的 run plan 时，不要直接复用存档中的回调路径。

## 本次 Review 范围

功能提交按以下类别拆分：权重加载与转换、数据处理与单遍监督、loss／GA 归一化与安全收尾。历史实验调度脚本和配置仍保留在本地，未批量纳入本次代码 review。本次提交不包含数据集、媒体文件、checkpoint 或评测产物。
