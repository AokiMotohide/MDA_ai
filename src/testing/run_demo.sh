#!/usr/bin/env bash
#
# Demo wrapper around run_inference_video.py. Takes an image folder or a video
# file (frames extracted with ffmpeg), runs MDA inference, writes to --output_dir.
#
# Download checkpoints first: hf download sy000/MDA --local-dir checkpoints/MDA
# See src/testing/utils/model_choice.py for available --model_name values.
#
# Usage:
#   bash src/testing/run_demo.sh <input_path> [--output_dir DIR] [--model_name NAME]
#                                             [--size N] [--fps N] [--max_chunk N]

set -euo pipefail

MODEL_NAME="mda_mog_sky_l2"
OUTPUT_DIR=""
SIZE=512
FPS=15
MAX_CHUNK=48
PYTHON="${PYTHON:-python}"

INPUT_PATH="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --model_name)  MODEL_NAME="$2";  shift 2 ;;
        --size)        SIZE="$2";        shift 2 ;;
        --fps)         FPS="$2";         shift 2 ;;
        --max_chunk)   MAX_CHUNK="$2";   shift 2 ;;
        *)             shift ;;
    esac
done

: "${OUTPUT_DIR:=eval_results/demo/$(basename "${INPUT_PATH%.*}")}"

# A file is treated as a video: extract frames to a temp dir. A directory is
# used as an image folder directly.
if [[ -f "$INPUT_PATH" ]]; then
    FRAME_DIR="$(mktemp -d)"
    trap 'rm -rf "$FRAME_DIR"' EXIT
    ffmpeg -hide_banner -loglevel error -y -i "$INPUT_PATH" -vf "fps=${FPS}" "$FRAME_DIR/frame_%05d.png"
    INPUT_PATH="$FRAME_DIR"
fi

mkdir -p "$OUTPUT_DIR"

$PYTHON src/testing/run_inference_video.py \
    --model_name "$MODEL_NAME" \
    --img_path   "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --size       "$SIZE" \
    --max_chunk  "$MAX_CHUNK"

echo ">> Done. Results written to: $OUTPUT_DIR"
