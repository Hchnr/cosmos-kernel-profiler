#!/usr/bin/env bash
set -euo pipefail
ulimit -n 65536
cd /share/project/eai_pwm/home/hcr/worktrees/cosmos-vlm-kernel-20260928
# Audit snapshot: regenerate the temporary IPC alias with a fresh run-id.
export LD_LIBRARY_PATH=''
export COSMOS_REPO=/share/project/eai_pwm/home/hcr/worktrees/cosmos-vlm-kernel-20260928
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler:/share/project/eai_pwm/home/hcr/worktrees/cosmos-vlm-kernel-20260928
export WANDB_MODE=disabled
export COSMOS_TRAINING=true
export IMAGINAIRE_OUTPUT_ROOT=/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/runs/vlm_capture002/job
export COSMOS_KERNEL_CACHE_ROOT=/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/runs/vlm_capture002/cache
export COSMOS_KERNEL_REPORT_DIR=/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/runs/vlm_capture002/reports
export COSMOS_KERNEL_START_STEP=0
export COSMOS_KERNEL_STACK_DIR=/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/runs/vlm_capture002/stacks
export TMPDIR=/tmp/vkl-l1y2p3ee/tmp
export TMP=/tmp/vkl-l1y2p3ee/tmp
export TEMP=/tmp/vkl-l1y2p3ee/tmp
export HF_HOME=/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/runs/vlm_capture002/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_MODEL_PATH=/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
/share/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 /share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/run_profile.py --sft-toml=/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/runs/vlm_capture002/recipe.toml -- job.project=kernel-list job.group=vlm job.name=vlm_capture002 job.wandb_mode=disabled upload_reproducible_setup=false model.config.parallelism.data_parallel_shard_degree=8 model.config.parallelism.data_parallel_replicate_degree=1 model.config.parallelism.context_parallel_shard_degree=1 +model.config.vlm_config.model_instance.config.include_visual=true model.config.vlm_config.pretrained_weights.enabled=true model.config.vlm_config.pretrained_weights.backbone_path=/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct model.config.vlm_config.tokenizer.pretrained_model_name=/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct checkpoint.load_training_state=false checkpoint.only_load_scheduler_state=false checkpoint.save_iter=1000000 checkpoint.save_to_object_store.enabled=false trainer.max_iter=10 trainer.seed=42 trainer.run_validation=false trainer.save_zero_checkpoint=false trainer.profiling.enable_profiling=true trainer.profiling.enable_nsys=false trainer.profiling.enable_memory_snapshot=false trainer.profiling.profile_freq=10 trainer.profiling.profile_warmup=2 trainer.profiling.profile_active=3 'trainer.profiling.target_ranks=[0]' trainer.profiling.record_shape=true trainer.profiling.profile_memory=false trainer.profiling.with_stack=false trainer.profiling.with_modules=false '~trainer.callbacks.every_n_sample_reg=null' '~trainer.callbacks.every_n_sample_ema=null' +trainer.callbacks.kernel_phases._target_=phase_audit.PhaseAudit +trainer.callbacks.kernel_vlm_audit._target_=vlm_audit.VLMAudit +trainer.callbacks.kernel_vlm_audit.output_dir=/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/runs/vlm_capture002/batches
