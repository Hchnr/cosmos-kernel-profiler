#!/usr/bin/env bash
set -euo pipefail

result_root=/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm/outputs/worldarena_eval25_480p_steps50
repo_root=/share/project/eai_pwm/repos/cosmos-framework-pwm
run_root=${repo_root}/outputs/usr_v2/cosmos3-t2v/nano-128gpu/nano-usr-v2-r7-1056-resume12600-128gpu-ga2-hsdp8x16-r480-fixed-static-und3k-gen96k-cfg01-cond541
input_path=/share/project/liyue/data/eval/worldarena-eval25-640x480-20260909/worldarena-eval25.parquet

gpus_are_free() {
  local app_count
  local memory_used
  app_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
  (( app_count == 0 )) || return 1
  while IFS= read -r memory_used; do
    (( memory_used < 2048 )) || return 1
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
}

wait_for_gpus() {
  local free_checks=0
  while (( free_checks < 12 )); do
    if gpus_are_free; then
      ((free_checks += 1))
      echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') GPUs free (${free_checks}/12)"
    else
      free_checks=0
      echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') GPUs occupied; waiting"
    fi
    sleep 10
  done
}

run_checkpoint() {
  local iter=$1
  local ckpt_name
  local out_dir
  printf -v ckpt_name 'iter_%09d' "${iter}"
  out_dir=${result_root}/vlm_v2_t2v_iter${iter}_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4

  test -d "${run_root}/checkpoints/${ckpt_name}"
  test -f "${run_root}/config.yaml"
  test -f "${input_path}"
  mkdir -p "${out_dir}"
  rm -f "${out_dir}/concurrent_gpu_use_detected.txt"
  cd "${repo_root}"
  ulimit -n "$(ulimit -Hn)"

  local cmd=(
    uv run --no-sync torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference
    -i "${input_path}" -o "${out_dir}"
    --checkpoint-path "${run_root}/checkpoints/${ckpt_name}"
    --config-file "${run_root}/config.yaml"
    --no-use-ema-weights
    --parallelism-preset=throughput --max-num-seqs=4
    --sampler=unipc --resolution=480 --num-steps=50 --guidance=5 --shift=10
    --seed=2026090900 --no-use-cuda-graphs --no-guardrails
    --dp-shard-size=4 --dp-replicate-size=2
    --use-torch-compile --compiled-region=language
    --experiment-overrides
      model.config.tower_structure=two_tower
      model.config.action_tower_preset=full
      model.config.action_tower_init=checkpoint
      model.config.action_tower_projection_mode=routed
      model.config.action_tower_execution=native
      model.config.action_tower_attention_backend=auto
      model.config.action_tower_und_short_tokens=0
      model.config.fixed_sequence_length.action_tokens=0
      model.config.fixed_sequence_length.enabled=false
      model.config.fixed_sequence_length.understanding_tokens=0
      model.config.fixed_sequence_length.und_tokens=0
      model.config.fixed_sequence_length.gen_tokens=0
      model.config.load_vision_tokenizer=true
      model.config.vision_vae_mode=online
      model.config.tokenizer.vae_path=/share/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth
      "model.config.tokenizer.bucket_name=''"
  )

  printf '%q ' "${cmd[@]}" > "${out_dir}/command.txt"
  printf '\n' >> "${out_dir}/command.txt"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "${out_dir}/started_utc.txt"

  (
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits -lms 500 > "${out_dir}/gpu_telemetry.csv" &
    monitor_pid=$!
    inference_pid=
    watcher_pid=
    cleanup_processes() {
      if [[ -n "${watcher_pid}" ]]; then
        kill "${watcher_pid}" 2>/dev/null || true
        wait "${watcher_pid}" 2>/dev/null || true
      fi
      if [[ -n "${inference_pid}" ]]; then
        kill -TERM -- "-${inference_pid}" 2>/dev/null || true
      fi
      kill "${monitor_pid}" 2>/dev/null || true
      wait "${monitor_pid}" 2>/dev/null || true
    }
    trap cleanup_processes EXIT

    start_seconds=$SECONDS
    set +e
    setsid env LD_LIBRARY_PATH='' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      "${cmd[@]}" > "${out_dir}/launch.log" 2>&1 &
    inference_pid=$!
    (
      consecutive_conflicts=0
      while kill -0 "${inference_pid}" 2>/dev/null; do
        if ! app_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
          | sed '/^[[:space:]]*$/d' | wc -l); then
          app_count=999
        fi
        if (( app_count > 8 )); then
          ((consecutive_conflicts += 1))
        else
          consecutive_conflicts=0
        fi
        if (( consecutive_conflicts >= 2 )); then
          date -u +'%Y-%m-%dT%H:%M:%SZ concurrent GPU workload detected; aborting this attempt' \
            > "${out_dir}/concurrent_gpu_use_detected.txt"
          kill -TERM -- "-${inference_pid}" 2>/dev/null || true
          exit 0
        fi
        sleep 3
      done
    ) &
    watcher_pid=$!
    wait "${inference_pid}"
    status=$?
    set -e

    printf '%s\n' "${status}" > "${out_dir}/exit_code.txt"
    printf '%s\n' "$((SECONDS - start_seconds))" > "${out_dir}/wall_seconds.txt"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${out_dir}/finished_utc.txt"
    exit "${status}"
  )
}

archive_concurrent_attempt() {
  local iter=$1
  local out_dir=${result_root}/vlm_v2_t2v_iter${iter}_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4
  local attempt=${out_dir}/attempts/concurrent_gpu_use_$(date -u +'%Y%m%dT%H%M%SZ')
  mkdir -p "${attempt}"
  find "${out_dir}" -mindepth 1 -maxdepth 1 ! -name attempts -exec mv -t "${attempt}" -- {} +
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') archived concurrent-use attempt for iter ${iter}"
}

for iter in 21000; do
  while true; do
    wait_for_gpus
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') starting iter ${iter}"
    if run_checkpoint "${iter}"; then
      break
    fi
    out_dir=${result_root}/vlm_v2_t2v_iter${iter}_online_480p_steps50_cfg5_fsdp4_dp2_compile_language_batched4
    if [[ -f "${out_dir}/concurrent_gpu_use_detected.txt" ]]; then
      archive_concurrent_attempt "${iter}"
      continue
    fi
    echo "iter ${iter} failed without a concurrent GPU workload" >&2
    exit 1
  done
  echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') finished iter ${iter}"
done
