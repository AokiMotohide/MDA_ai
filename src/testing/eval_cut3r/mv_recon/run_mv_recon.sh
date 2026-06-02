#!/bin/bash

# set -e

workdir='.'

# Model checkpoints are resolved by src/testing/utils/model_choice.py from the
# --model_name (download them first: hf download sy000/MDA --local-dir checkpoints/MDA).
model_names=('mda_mog_sky_l2')
PYTHON=/nfs/turbo/coe-jungaocv/siyuanb/miniconda3/envs/da3/bin/python

# --- Pass 2: score (margin + gt_cam_output) ---
for model_name in "${model_names[@]}"; do
    output_dir="${workdir}/eval_results/mv_recon_gpu/${model_name}"
    echo "=== Score (margin): $output_dir ==="
    $PYTHON src/testing/eval_cut3r/mv_recon/launch.py \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --margin --gt_cam_output 1
done

# --- Pass 3: per-view (margin + gt_cam_output) ---
for model_name in "${model_names[@]}"; do
    output_dir="${workdir}/eval_results/mv_recon_gpu/${model_name}"
    echo "=== Per-view (margin): $output_dir ==="
    $PYTHON src/testing/eval_cut3r/mv_recon/launch_perview.py \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --margin --gt_cam_output 1
done


# --- Pass 1: score (no margin) + sideview vis ---
for model_name in "${model_names[@]}"; do
    output_dir="${workdir}/eval_results/mv_recon_gpu/${model_name}"
    echo "=== Score (no margin): $output_dir ==="
    $PYTHON src/testing/eval_cut3r/mv_recon/launch.py \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
         --gt_cam_output 1

done