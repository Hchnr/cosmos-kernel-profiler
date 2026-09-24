# 恢复执行记录

## 2026-09-24：新仓库继续

- 用户通知继续，在独立仓库执行；不再更新原plans目录。开始时工作区干净，HEAD为fa07a4b，保留已有README更新。
- 当前8卡无计算进程；基础环境已变为Python3.13.14、Torch2.10.0+cu130、CUDA13.0、cuDNN9.15.1、Triton3.6.0。此前Python3.12/Torch2.12的NATTEN wheel不能复用。
- 新增requirements-overlay-cu130-torch210.txt，沿用Python依赖锁定版本，Attention改用外部框架uv.lock中的flash-attn-3-nv1.0.3+cu130.torch210 cp39-abi3官方wheel，隔离安装到新仓库env；保留系统Torch。
- probe_attention改为记录可用FA3/NATTEN包版本，不再强制导入NATTEN。数值计算仍调用框架自动选择的真实Attention前端，以验证当前后端。
- 继续保持原USR TOML、UND3k/GEN96k、GA2、DP shard8、20/2/3窗口，计划恢复12600到12625。

## 恢复环境的局部验证

- FA3 causal packed GQA和noncausal异长packed GQA的fullgraph编译前向/反向、缓存路径均通过，与FP32参考的最大输出绝对误差0.0082207/0.0044644；validation/attention-fa3-001/result.json。
- Torch2.10的真实CUDA小型profiler探针通过，四文件导出24次计算kernel，validation/cuda-probe-torch210-001。7项CPU检查再次通过。
- resumed_preflight001因新基础环境缺少python-dateutil失败；检查multi-storage-client的运行依赖后，按框架uv.lock在overlay补齐16个包（含wandb等训练导入依赖），未更换系统包。第二次预检使用新run_id保留旧日志。
- resumed_preflight002继续暴露boto3缺失。新容器缺少原容器预装的训练库，按锁定版本补齐对象存储、Parquet/数据处理、Qwen图像处理和日志的必要导入依赖；Torchvision固定0.25.0+cu130以匹配Torch2.10。安装清单追加至当前环境requirements文件，所有包仅落在env目录。

## resumed_preflight004通过，启动正式运行

- resumed_preflight003在wandb导入时缺少protobuf；已按锁定版本补齐6.33.5。预检004返回0，完整配置构建成功，配置保存在runs/resumed_preflight004/job/kernel-list/v0.2.3c/profile/config.yaml。
- 接下来使用新目录resumed_capture001运行8卡。已通过当前环境的FA3数值检查和四文件CUDA探针，保持原始USR长度/采样/GA/compile/full-state，仅按计划覆盖并行度和窗口。
