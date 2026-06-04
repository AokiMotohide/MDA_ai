"""
MDA demo: run multi-view inference on an image folder (or video file) and save
per-frame outputs in the layout expected by the bundled viser viewer
(`view.py`):

    <output_dir>/<model_name>/
        depth/000000.npy      (H, W) float32 depth
        color/000000.png      (H, W, 3) uint8 RGB
        camera/000000.npz     pose (4x4 C2W, OpenCV) + intrinsics (3x3)
        sky/000000.npy        (H, W) bool sky mask (only when the model detects sky)

All configuration lives in the `DemoConfig` dataclass below; every field is
also exposed as a CLI flag.

Examples:
    python demo.py path/to/video.mp4 --fps 5
    python demo.py path/to/image_folder --no-viewer
    python demo.py path/to/image_folder --model_name vggt_mog_l2
    python demo.py assets/examples/dolomiti
    python demo.py assets/examples/redbull
    python demo.py assets/examples/game
    python demo.py assets/examples/diode_indoor  # unordered indoor stills (DIODE)
    python demo.py assets/examples/mono/painting/painting.jpeg  # single-image (mono) inference
    python demo.py assets/examples/mono/animal/animal.jpg       # single-image (mono) inference
"""

import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.testing.utils.model_choice import choose_model  # noqa: E402
from src.testing.run_inference_video import (  # noqa: E402
    _concat_frame_dim,
    _preds_to_cpu,
    prepare_views,
)
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DemoConfig:
    # --- I/O ---
    input_path: str = ""  # image folder, or video file (frames extracted with ffmpeg)
    output_dir: str = ""  # default: eval_results/demo/<input_basename>
    fps: int = 15  # frame-extraction fps when input_path is a video file
    image_stride: int = 1  # downsample rate for image inputs: keep every Nth image

    # --- Model ---
    model_name: str = "mda_mog_sky_l2"  # see src/testing/utils/model_choice.py
    size: int = 512  # inference image size (long edge)
    max_chunk: int = 48  # max frames per forward pass (chunking breaks cross-chunk attention)

    # --- model.inference() flags (mirrors run_inference_video.py defaults) ---
    crop_center_112: bool = False
    cam_inp: bool = False
    gt_cam_output: bool = False
    output_normalize: bool = False
    use_sky_mask: bool = True  # predict sky and save sky/*.npy masks for the viewer

    # --- Viewer (bundled view.py) ---
    viewer: bool = True  # launch view.py after inference (disable: --no-viewer)
    view_script: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "view.py"
    )  # viser viewer script (defaults to the bundled view.py)
    viewer_port: int = 8080
    viewer_up_direction: str = "-y"  # +x/-x/+y/-y/+z/-z

    # ImageNet de-normalization for saving color frames.
    rgb_mean: tuple = field(default=(0.485, 0.456, 0.406), repr=False)
    rgb_std: tuple = field(default=(0.229, 0.224, 0.225), repr=False)


def parse_args() -> DemoConfig:
    """Build an argparse CLI from DemoConfig fields and return the filled config."""
    defaults = DemoConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_path", type=str, help="Image folder or video file")
    for f in dataclasses.fields(DemoConfig):
        if f.name in ("input_path", "rgb_mean", "rgb_std"):
            continue
        default = getattr(defaults, f.name)
        if f.type is bool or isinstance(default, bool):
            parser.add_argument(
                f"--{f.name}", action=argparse.BooleanOptionalAction, default=default
            )
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    args = parser.parse_args()
    return DemoConfig(**{f.name: getattr(args, f.name, getattr(defaults, f.name))
                         for f in dataclasses.fields(DemoConfig)})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def extract_video_frames(video_path: str, frame_dir: str, fps: int) -> None:
    """Extract frames from a video with ffmpeg (same as run_demo.sh)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; needed to extract video frames")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-vf", f"fps={fps}",
            os.path.join(frame_dir, "frame_%05d.png"),
        ],
        check=True,
    )


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(img_dir: str, stride: int = 1):
    """Sorted list of image files in a folder, keeping every `stride`-th image."""
    files = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )
    return files[:: max(1, stride)]


def merge_chunk_predictions(chunks):
    """Concatenate per-chunk predictions along the frame axis (dim=1)."""
    if len(chunks) == 1:
        return chunks[0]
    merged = dict(chunks[0])
    for key in ("depth", "depth_conf", "images", "pose_enc", "sky_mask"):
        if all(c.get(key) is not None for c in chunks):
            merged[key] = _concat_frame_dim([c[key] for c in chunks])
    return merged


def run_inference(cfg: DemoConfig, filelist):
    """Load the model, run chunked multi-view inference, return merged predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {cfg.model_name}")
    loaded = choose_model(cfg.model_name)
    model = loaded.model.to(device).eval()
    is_mog = "mog" in cfg.model_name.lower()

    views = prepare_views(filelist, cfg.size, loaded.patch_size, loaded.img_norm)

    num_views = len(filelist)
    max_chunk = max(1, cfg.max_chunk)
    chunk_predictions = []
    for chunk_start in range(0, num_views, max_chunk):
        chunk_end = min(chunk_start + max_chunk, num_views)
        print(f">> Inference on frames [{chunk_start}:{chunk_end}] of {num_views}")
        with torch.no_grad():
            chunk_pred = model.inference(
                views[chunk_start:chunk_end],
                device,
                is_mog=is_mog,
                use_sky_mask=cfg.use_sky_mask,
                crop_center_112=cfg.crop_center_112,
                cam_inp=cfg.cam_inp,
                gt_cam_output=cfg.gt_cam_output,
                output_normalize=cfg.output_normalize,
            )
        chunk_predictions.append(_preds_to_cpu(chunk_pred))
        del chunk_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return merge_chunk_predictions(chunk_predictions)


def cameras_c2w_from_predictions(predictions, L, H, W):
    """Recover per-view (c2w 4x4, intrinsics 3x3) from predictions.

    Mirrors Depth-Anything-3's run_inference_video._cameras_c2w_from_predictions:
      1. A valid pose encoding (>=7-dim): pose_encoding_to_extri_intri returns
         camera-to-world (c2w) DIRECTLY for the MoG / cam-decoder models (the
         net does output.extrinsics = affine_inverse(c2w)) — do NOT invert.
      2. Fallback: raw_preds extrinsics are world->cam and must be inverted.
    """
    pose_enc = predictions.get("pose_enc")
    if (pose_enc is not None and getattr(pose_enc, "ndim", 0) == 3
            and pose_enc.shape[1] >= L and pose_enc.shape[-1] >= 7):
        with torch.no_grad():
            c2w_34, K = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(H, W))
        c2w_34 = c2w_34[0, :L].cpu().numpy().astype(np.float64)  # (L, 3, 4) cam->world
        K = K[0, :L].cpu().numpy().astype(np.float64)
        c2w = np.tile(np.eye(4, dtype=np.float64), (L, 1, 1))
        c2w[:, :3, :] = c2w_34
        return c2w, K

    raw = predictions.get("raw_preds")
    ext = raw.get("extrinsics") if hasattr(raw, "get") else None
    K_t = raw.get("intrinsics") if hasattr(raw, "get") else None
    if ext is not None and K_t is not None:
        ext = ext.detach().cpu().numpy()
        K_np = K_t.detach().cpu().numpy()
        if ext.ndim == 4:
            ext = ext[0]
        if K_np.ndim == 4:
            K_np = K_np[0]
        ext = ext[:L].astype(np.float64)  # (L, 3|4, 4) world->cam
        K_np = K_np[:L].astype(np.float64)
        w2c = np.tile(np.eye(4, dtype=np.float64), (L, 1, 1))
        w2c[:, : ext.shape[1], : ext.shape[2]] = ext
        return np.linalg.inv(w2c), K_np

    raise RuntimeError("no usable pose_enc / raw_preds extrinsics in predictions")


def save_viewer_outputs(cfg: DemoConfig, predictions, run_dir: str) -> None:
    """Write depth/, color/, camera/ (+ optional sky/) per-frame files for view.py."""
    import imageio.v2 as iio

    for sub in ("depth", "color", "camera"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    depth = predictions["depth"].squeeze(0).squeeze(-1) \
        if predictions["depth"].ndim == 5 else predictions["depth"].squeeze(0)
    depth = depth.cpu().numpy().astype(np.float32)  # (N, H, W)
    num_views = depth.shape[0]

    # Sky mask (depth there is already pushed to 2x scene max by the wrapper);
    # the viewer uses it to hide sky by default and for its "Show sky" toggle.
    sky_mask = predictions.get("sky_mask")
    sky = None
    if sky_mask is not None:
        sky = sky_mask.squeeze(0).cpu().numpy().astype(bool)[:num_views]
        if sky.any():
            os.makedirs(os.path.join(run_dir, "sky"), exist_ok=True)
            print(f"Sky mask: {int(sky.sum())} sky pixels")
        else:
            sky = None

    # Colors: undo ImageNet normalization.
    imgs = predictions["images"].squeeze(0)[:num_views].permute(0, 2, 3, 1).cpu().numpy()
    mean = np.array(cfg.rgb_mean, dtype=np.float32)
    std = np.array(cfg.rgb_std, dtype=np.float32)
    colors = (np.clip(imgs * std + mean, 0, 1) * 255).astype(np.uint8)

    # Cameras: C2W (OpenCV) for view.py — see cameras_c2w_from_predictions.
    H, W = depth.shape[1:]
    c2w, intrinsics = cameras_c2w_from_predictions(predictions, num_views, H, W)

    for i in range(num_views):
        tag = f"{i:06d}"
        np.save(os.path.join(run_dir, "depth", f"{tag}.npy"), depth[i])
        iio.imwrite(os.path.join(run_dir, "color", f"{tag}.png"), colors[i])
        np.savez(
            os.path.join(run_dir, "camera", f"{tag}.npz"),
            pose=c2w[i].astype(np.float32),  # 4x4 C2W (OpenCV)
            intrinsics=intrinsics[i].astype(np.float32),
        )
        if sky is not None:
            np.save(os.path.join(run_dir, "sky", f"{tag}.npy"), sky[i])
    print(f"Saved {num_views} frames (depth/color/camera"
          f"{'/sky' if sky is not None else ''}) to {run_dir}")


def launch_viewer(cfg: DemoConfig, run_dir: str) -> None:
    """Run the viser viewer on the saved run dir (blocking)."""
    if not cfg.view_script or not os.path.isfile(cfg.view_script):
        raise FileNotFoundError(
            f"view script not found: {cfg.view_script!r} "
            "(pass --view_script pointing to this repo's view.py)"
        )
    cmd = [
        sys.executable, cfg.view_script,
        "--data_dir", run_dir,
        "--port", str(cfg.viewer_port),
        f"--up_direction={cfg.viewer_up_direction}",  # "=" form: a bare "-y" is parsed as a flag
    ]
    print(f">> Launching viewer: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = parse_args()
    print(cfg)

    output_dir = cfg.output_dir or os.path.join(
        "eval_results", "demo",
        os.path.splitext(os.path.basename(os.path.normpath(cfg.input_path)))[0],
    )
    run_dir = os.path.join(output_dir, cfg.model_name)

    tmp_frame_dir = None
    try:
        if os.path.isfile(cfg.input_path) and cfg.input_path.lower().endswith(IMAGE_EXTS):
            # Single image file: monocular (one-view) inference.
            filelist = [cfg.input_path]
        else:
            img_dir = cfg.input_path
            if os.path.isfile(cfg.input_path):  # video file: extract frames first
                tmp_frame_dir = tempfile.mkdtemp(prefix="mda_demo_frames_")
                print(f"Extracting frames from {cfg.input_path} at {cfg.fps} fps ...")
                extract_video_frames(cfg.input_path, tmp_frame_dir, cfg.fps)
                img_dir = tmp_frame_dir
            filelist = collect_images(img_dir, stride=cfg.image_stride)
        if not filelist:
            raise SystemExit(f"No images found in {cfg.input_path}")
        print(f"Found {len(filelist)} images (stride={cfg.image_stride}).")

        predictions = run_inference(cfg, filelist)
        save_viewer_outputs(cfg, predictions, run_dir)
    finally:
        if tmp_frame_dir is not None:
            shutil.rmtree(tmp_frame_dir, ignore_errors=True)

    print(f"Done! Results saved to {run_dir}")
    if cfg.viewer:
        launch_viewer(cfg, run_dir)
    else:
        print(
            "View with:\n"
            f"  python {cfg.view_script or '<path/to/view.py>'} "
            f"--data_dir {run_dir} --port {cfg.viewer_port}"
        )


if __name__ == "__main__":
    main()
