#!/usr/bin/env bash
set -euo pipefail
cd /share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm
export LD_LIBRARY_PATH=''
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm:/share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler
export TORCHINDUCTOR_COMPILE_THREADS=4
/share/project/eai_pwm/home/hcr/venv/cosmos-framework-pwm-cu130-train/bin/python /share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/run_inference_profile.py \
 --mode t2v --output /share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/evidence/captures/t2v_inference_capture001 \
 -i /share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/evidence/captures/t2v_inference_capture001/input.json \
 -o /share/project/eai_pwm/home/hcr/repos/cosmos-kernel-profiler/evidence/captures/t2v_inference_capture001/outputs \
 --checkpoint-path /share/project/eai_pwm/models/Cosmos3-Nano \
 --experiment-overrides model.config.sound_tokenizer.bucket_name= model.config.sound_tokenizer.avae_path=/share/project/eai_pwm/models/Cosmos3-Nano/sound_tokenizer/avae_48k_noncausal_25hz_64ch.ckpt model.config.tokenizer.bucket_name= model.config.tokenizer.vae_path=/share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth \
 --seed 42 --no-guardrails --no-use-cuda-graphs
