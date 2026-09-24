# 独立仓库迁移记录

2026-09-24，按用户要求建立独立本地仓库cosmos-kernel-profiler，未设置远程、未推送。原项目Git HEAD仍为872f6bf2888c8899c1599dda8c722e1c551684fa，plans保持被忽略。

复制94份实现、文档和精选中间证据，共约1.9MB。排除安装环境、编译缓存、模型、数据、训练checkpoint；原文件未移动或删除，逐项源SHA256检查全部通过。

独立仓库新增workspace.py，用COSMOS_REPO显式定位外部框架，调整launcher、wrapper与waiter的目录解析。输出留在新仓库runs/validation，metadata分别记录框架与工具Git SHA。修复新目录下waiter首次运行时validation目录不存在的问题，并统一本仓库脚本格式；保留FlagScale许可证；后续来源校验发现迁移格式化触及其排版，已恢复上游原字节并验证三个CSV可逐字节重导出，详见正式采集证据中的upstream_reexport_validation.json。

本次验证：7项CPU单元检查、历史小型探针四文件离线验证、外部工作树定位及缺失路径拒绝、生成的启动命令/环境/shard8检查、短IPC链接清理、全部脚本语法检查、Ruff检查和格式检查均通过。证据位于evidence/migration/。这些检查没有启动训练、CUDA数值探针或资源等待器。

当前正式USR采集仍未完成。等待用户通知GPU资源可用后，从新仓库继续：先核验当时的Torch/CUDA/NATTEN ABI与本地数据资产，再执行配置预检和8卡采集；不要直接假定历史容器依赖仍然有效。
