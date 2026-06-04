#!/bin/bash

# set -e

workdir='.'

# Model checkpoints are resolved by src/testing/utils/model_choice.py from the
# --model_name (download them first: hf download sy000/MDA --local-dir checkpoints/MDA).
model_names=('mda_mog_sky_l2')
PYTHON=python

datasets=('sintel' 'bonn' 'kitti' 'diode')

for data in "${datasets[@]}"; do
    for model_name in "${model_names[@]}"; do

        output_dir="${workdir}/eval_results/video_depth_final/${data}_${model_name}"
        echo "Running dataset=${data}, model=${model_name}"
        echo "Output dir: $output_dir"

        # --- Step 1: Inference + save depth maps & vis data ---
        echo "=== Score: $output_dir ==="
        $PYTHON src/testing/eval_cut3r/video_depth/launch_final_score.py \
            --output_dir "$output_dir" \
            --eval_dataset "$data" \
            --model_name "$model_name" \
            --size 512 \
            --gt_cam_output 0

        # --- Step 2: Depth evaluation ---
        echo "=== Eval depth: $output_dir ==="
        $PYTHON src/testing/eval_cut3r/video_depth/eval_depth.py \
            --output_dir "$output_dir" \
            --eval_dataset "$data" \
            --align "scale" \
            --gt_cam_output 0

    done
done
