#!/bin/bash

workdir='.'
# Boundary-blur ablation. Checkpoints resolve from --model_name via
# src/testing/utils/model_choice.py (hf download sy000/MDA --local-dir checkpoints/MDA).
model_names=('mda_mog_sky_l2')
PYTHON=python

for model_name in "${model_names[@]}"; do
    output_dir="${workdir}/eval_results/mv_recon_blur/${model_name}"
    echo "=== Boundary-blur ablation: $model_name ==="
    $PYTHON src/testing/eval_cut3r/mv_recon/launch_blur.py \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --blur_factors 1 2 4 8 \
        --gt_cam_output 1 \
        --save_vis
done


