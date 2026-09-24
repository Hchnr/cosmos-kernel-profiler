# 中间证据

这些文件是迁移时的历史快照，内含原机器路径和已从原分支移除的提交号；它们用于解释当时的执行，不是新仓库的可执行启动命令。

- `import_manifest.json`：94份复制文件的来源、大小及原始SHA256。源文件迁入后，部分脚本进行了独立仓库路径适配，docs增加历史标记；其余证据保留原字节。
- `validation_summary.json`：此前7项CPU检查及小型CUDA探针结果。
- `validation/cuda-probe-001/`：**玩具矩阵运算与优化器探针**，不是USR训练。四文件有效，24次计算kernel调用、16种变体、9个operator分组，4次调用缺失dtype。
- `validation/attention-001/`、`validation/usr_capture005-attention/`：两种packed GQA的fullgraph编译前后向数值检查通过，包含共享设备与空闲设备两次验证。
- `validation/catalogs.json`、`validation/latent_headers.json`：29个catalog及首行离线latent的只读检查。
- `runs/preflight*/`：配置预检过程；003、004成功，001、002保留依赖问题证据。
- `runs/usr_capture001/`：checkpoint Path pickle兼容性失败，已实现兼容适配。
- `runs/usr_capture002/`：恢复12600成功，随后IPC路径过长，已用短链接解决。
- `runs/usr_capture003/`：GPU资源准入拒绝，未启动训练。
- `runs/usr_capture004/`：首步缺少varlen Attention后端，后续补齐NATTEN并完成小型数值验证。
- `runs/usr_capture005/`：恢复成功，因另一组8卡作业启动而主动中止；本次进程已清理。
- `validation/usr_capture006-attention.log`：等待资源后，预检无法导入Torch；未启动正式训练。

**没有一次正式USR active窗口完成，因而本仓库尚无最终交付清单。** 大型环境、缓存、权重、数据和checkpoint仍在原位置，没有打包或提交。
