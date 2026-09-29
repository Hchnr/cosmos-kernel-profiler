#!/usr/bin/env bash
set -euo pipefail
cd /share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm
export LD_LIBRARY_PATH=''
export CUDA_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm:/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler
/share/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train/bin/python /share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/run_inference_profile.py \
 --mode vlm_only --output /share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/evidence/captures/vlm_only_inference_capture001 \
 --checkpoint /share/project/eai_pwm/models/Qwen3-VL-8B-Instruct \
 --input /share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/evidence/captures/vlm_only_inference_capture001/input.json
