# -*- coding: utf-8 -*-
"""
MDA DA3 reconstruction script.

This script mirrors the VGGT custom script's command-line shape, progress
markers, and output filenames while using the MDA DA3 checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import site
import sys
import time
from pathlib import Path


def emit_progress(phase: str, progress: float, msg: str = "") -> None:
    """Emit a stable machine-readable marker for existing VGGT launchers."""
    print(f"▶ [PHASE:{phase}] ({progress:.2f}) {msg}", flush=True)


emit_progress("startup", 0.01, "Python起動中")


def remove_user_site_packages() -> None:
    """Prefer the active conda env over per-user Python packages."""
    try:
        user_sites = site.getusersitepackages()
    except AttributeError:
        user_sites = []
    if isinstance(user_sites, str):
        user_sites = [user_sites]

    normalized_user_sites = {
        os.path.normcase(os.path.abspath(path)) for path in user_sites
    }
    if not normalized_user_sites:
        return

    sys.path[:] = [
        path
        for path in sys.path
        if os.path.normcase(os.path.abspath(path)) not in normalized_user_sites
    ]


remove_user_site_packages()


def resolve_mda_root() -> Path:
    env_root = os.environ.get("MDA_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    here = Path(__file__).resolve()
    candidates.extend([here.parent.parent, here.parent])
    candidates.extend(here.parents)

    for candidate in candidates:
        root = candidate.resolve()
        if (root / "demo.py").is_file() and (root / "src").is_dir():
            return root

    raise RuntimeError(
        "MDA root was not found. Set MDA_ROOT to the MDA repository directory."
    )


MDA_ROOT = resolve_mda_root()
sys.path.insert(0, str(MDA_ROOT))
os.chdir(MDA_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri  # noqa: E402
from depth_anything_3.utils.export.glb import (  # noqa: E402
    _add_cameras_to_scene,
    _compute_alignment_transform_first_cam_glTF_center_by_points,
    _depths_to_world_points_with_colors,
    _estimate_scene_scale,
    _filter_and_downsample,
)
from src.testing.run_inference_video import (  # noqa: E402
    _concat_frame_dim,
    _preds_to_cpu,
    prepare_views,
)
from src.testing.utils.model_choice import choose_model  # noqa: E402
import trimesh  # noqa: E402


IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MDA DA3 3D再構成スクリプト（VGGT JSON互換）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-folder",
        "-i",
        default="my_images",
        help="入力画像が置かれているフォルダ（直下の *.png, *.jpg, *.jpeg を検索）",
    )
    parser.add_argument(
        "--conf-thres",
        "-c",
        type=float,
        default=70.0,
        help="信頼度フィルタ：下位X%%の低信頼度ポイントを除外する",
    )

    sky_group = parser.add_mutually_exclusive_group()
    sky_group.add_argument(
        "--mask-sky",
        dest="mask_sky",
        action="store_true",
        help="MDAのsky maskを使ってGLB点群から空領域を除外",
    )
    sky_group.add_argument(
        "--no-mask-sky",
        dest="mask_sky",
        action="store_false",
        help="空領域除外を無効化",
    )
    parser.set_defaults(mask_sky=False)

    parser.add_argument(
        "--mask-black-bg",
        action="store_true",
        help="黒背景ピクセル（RGB合計<16）をGLB点群から除外",
    )
    parser.add_argument(
        "--mask-white-bg",
        action="store_true",
        help="白背景ピクセル（R,G,B全てが>240）をGLB点群から除外",
    )
    parser.add_argument(
        "--no-show-cam",
        dest="show_cam",
        action="store_false",
        help="GLB内のカメラ可視化を無効化",
    )
    parser.set_defaults(show_cam=True)

    parser.add_argument("--size", type=int, default=518, help="MDA推論の長辺サイズ")
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=0,
        help="1回のforwardに入れる最大フレーム数。0なら全画像を同一forwardに入れる",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="使用する最大画像枚数。0なら全画像を使用し、>0なら始点・終点を含めて均等間引き",
    )
    parser.add_argument(
        "--oom-action",
        choices=("exit", "lower-size"),
        default="exit",
        help="CUDA OOM時の動作。exit=終了、lower-size=解像度を下げて1回だけ再実行",
    )
    parser.add_argument(
        "--retry-size",
        type=int,
        default=384,
        help="--oom-action lower-size の再実行で使う長辺サイズ",
    )
    parser.add_argument(
        "--model-name",
        default="mda_mog_sky_l2",
        help="src/testing/utils/model_choice.py のモデル名",
    )
    parser.add_argument(
        "--env-name",
        default="mda",
        help="想定conda環境名。実行環境切替はせず、表示と警告にのみ使用",
    )
    parser.add_argument(
        "--num-max-points",
        type=int,
        default=1_000_000,
        help="GLBに残す最大点数",
    )
    args = parser.parse_args()
    if args.max_chunk < 0:
        parser.error("--max-chunk must be 0 or a positive integer")
    if args.max_images < 0:
        parser.error("--max-images must be 0 or a positive integer")
    if args.size <= 0:
        parser.error("--size must be a positive integer")
    if args.retry_size <= 0:
        parser.error("--retry-size must be a positive integer")
    return args


def print_config(args: argparse.Namespace, total_images: int, selected_images: int) -> None:
    line = "=" * 68
    print()
    print(line)
    print("  MDA DA3 3D再構成  実行設定")
    print(line)
    print(f"  MDA root              : {MDA_ROOT}")
    print(f"  入力画像フォルダ      : {args.image_folder}")
    print(f"  検出した画像枚数      : {total_images} 枚")
    print(f"  使用する画像枚数      : {selected_images} 枚")
    print(f"  使用モデル            : {args.model_name}")
    print(f"  想定conda環境         : {args.env_name}")
    print(f"  推論サイズ            : {args.size}")
    print(f"  OOM時動作             : {args.oom_action}")
    if args.oom_action == "lower-size":
        print(f"  OOM再試行サイズ       : {args.retry_size}")
    if args.max_chunk == 0:
        print(f"  max_chunk             : 0 (全{selected_images}枚を同一forward)")
    else:
        print(f"  max_chunk             : {args.max_chunk}")
    print(f"  max_images            : {args.max_images}")
    print(f"  信頼度フィルタ        : conf_thres = {args.conf_thres:.1f}")
    print(f"  空マスク              : {'ON' if args.mask_sky else 'OFF'}")
    print(f"  黒背景マスク          : {'ON' if args.mask_black_bg else 'OFF'}")
    print(f"  白背景マスク          : {'ON' if args.mask_white_bg else 'OFF'}")
    print(f"  カメラ可視化          : {'ON' if args.show_cam else 'OFF'}")
    print(line)
    print()


def check_environment(args: argparse.Namespace) -> None:
    emit_progress("env_check", 0.02, "環境チェック中")
    print("▶ 環境チェック中...")
    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env and active_env != args.env_name:
        print(f"   注意: CONDA_DEFAULT_ENV={active_env} (想定: {args.env_name})")
    if torch.cuda.is_available():
        print(f"   CUDA                 : 使用可能 ({torch.cuda.get_device_name(0)})")
    else:
        print("   CUDA                 : 使用不可 (CPUで実行)")
    ckpt_path = MDA_ROOT / "checkpoints" / "MDA" / "DA3_MOG_Sky_LogL2.ckpt"
    if args.model_name == "mda_mog_sky_l2" and not ckpt_path.is_file():
        raise FileNotFoundError(
            f"DA3 MDA checkpoint not found: {ckpt_path}. "
            "Run scripts/setup_windows_mda_da3.ps1 first."
        )


def collect_image_paths(image_folder: str) -> list[str]:
    paths: list[str] = []
    for pattern in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(image_folder, pattern)))
    return sorted(paths)


def select_image_subset(image_paths: list[str], max_images: int) -> list[str]:
    if max_images <= 0 or max_images >= len(image_paths):
        return image_paths
    if max_images == 1:
        return [image_paths[0]]

    indices = np.linspace(0, len(image_paths) - 1, max_images, dtype=int).tolist()
    return [image_paths[i] for i in indices]


def load_original_sizes(image_paths: list[str]) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    for path in image_paths:
        with Image.open(path) as img:
            sizes[os.path.basename(path)] = (img.width, img.height)
    return sizes


def merge_predictions(chunks: list[dict]) -> dict:
    if len(chunks) == 1:
        return chunks[0]

    merged = dict(chunks[0])
    for key in ("depth", "depth_conf", "images", "pose_enc", "sky_mask"):
        if all(c.get(key) is not None for c in chunks):
            merged[key] = _concat_frame_dim([c[key] for c in chunks])

    if all("raw_preds" in c for c in chunks):
        raw_merged = dict(chunks[0]["raw_preds"])
        raw_keys = set()
        for c in chunks:
            raw_keys.update(c["raw_preds"].keys())
        for key in raw_keys:
            if all(key in c["raw_preds"] for c in chunks):
                raw_merged[key] = _concat_frame_dim([c["raw_preds"][key] for c in chunks])
        merged["raw_preds"] = raw_merged

    return merged


def run_mda_inference(args: argparse.Namespace, image_paths: list[str]) -> dict:
    emit_progress("model_load", 0.10, "MDA DA3モデルを読み込み中")
    print("▶ MDA DA3モデルを初期化・読み込み中...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = choose_model(args.model_name)
    model = loaded.model.to(device).eval()
    is_mog = "mog" in args.model_name.lower()

    emit_progress("preprocess", 0.20, f"画像を前処理中 ({len(image_paths)}枚)")
    print("▶ 画像を前処理中...")
    views = prepare_views(image_paths, args.size, loaded.patch_size, loaded.img_norm)

    emit_progress("infer", 0.30, "AI推論を実行中")
    print("▶ AI推論を実行中...")
    print("※ MDA DA3 Giant は大きなモデルです。数十秒から数分かかる場合があります。")

    start = time.time()
    chunk_predictions = []
    max_chunk = len(image_paths) if args.max_chunk == 0 else args.max_chunk
    if max_chunk < len(image_paths):
        print(
            "WARNING: max_chunk が使用画像枚数より小さいため、"
            "cross-frame attention が分割され、相対カメラ推定が弱くなる可能性があります。"
        )
    else:
        print("INFO: 全画像を同一forwardに入れるmulti-view推論で実行します。")

    for chunk_start in range(0, len(image_paths), max_chunk):
        chunk_end = min(chunk_start + max_chunk, len(image_paths))
        print(f">> Inference on frames [{chunk_start}:{chunk_end}] of {len(image_paths)}")
        with torch.no_grad():
            pred = model.inference(
                views[chunk_start:chunk_end],
                device,
                is_mog=is_mog,
                use_sky_mask=args.mask_sky,
                crop_center_112=False,
                cam_inp=False,
                gt_cam_output=False,
                output_normalize=False,
            )
        chunk_predictions.append(_preds_to_cpu(pred))
        del pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = merge_predictions(chunk_predictions)
    elapsed = time.time() - start
    print(f"OK: 推論完了 (AI計算の所要時間: {elapsed:.1f} 秒)")
    emit_progress("infer_done", 0.70, f"推論完了 ({elapsed:.1f}s)")
    return predictions


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def extract_depth_conf_images(predictions: dict, num_views: int):
    depth = predictions["depth"]
    if depth.ndim == 5:
        depth = depth.squeeze(0).squeeze(-1)
    else:
        depth = depth.squeeze(0)
    depth_np = tensor_to_numpy(depth[:num_views]).astype(np.float32)

    conf = predictions.get("depth_conf")
    if conf is None or len(conf) == 0:
        conf_np = np.ones_like(depth_np, dtype=np.float32)
    else:
        if conf.ndim == 5:
            conf = conf.squeeze(0).squeeze(-1)
        else:
            conf = conf.squeeze(0)
        conf_np = tensor_to_numpy(conf[:num_views]).astype(np.float32)

    imgs = predictions["images"].squeeze(0)[:num_views].permute(0, 2, 3, 1)
    imgs_np = tensor_to_numpy(imgs).astype(np.float32)
    colors = (np.clip(imgs_np * RGB_STD + RGB_MEAN, 0, 1) * 255).astype(np.uint8)
    return depth_np, conf_np, colors


def extract_w2c_intrinsics(predictions: dict, num_views: int, image_hw: tuple[int, int]):
    emit_progress("camera_calc", 0.75, "カメラパラメータ計算中")
    print("▶ カメラパラメータ（外部・内部行列）を計算中...")

    raw = predictions.get("raw_preds", {})
    ext = raw.get("extrinsics") if hasattr(raw, "get") else None
    intr = raw.get("intrinsics") if hasattr(raw, "get") else None

    if ext is not None and intr is not None:
        ext_np = tensor_to_numpy(ext)
        intr_np = tensor_to_numpy(intr)
        if ext_np.ndim == 4:
            ext_np = ext_np[0]
        if intr_np.ndim == 4:
            intr_np = intr_np[0]
        ext_np = ext_np[:num_views].astype(np.float64)
        intr_np = intr_np[:num_views].astype(np.float64)
        w2c = np.tile(np.eye(4, dtype=np.float64), (num_views, 1, 1))
        w2c[:, : ext_np.shape[1], : ext_np.shape[2]] = ext_np
        return w2c, intr_np

    pose_enc = predictions.get("pose_enc")
    if pose_enc is None:
        raise RuntimeError("No raw extrinsics/intrinsics or pose_enc found in predictions.")

    with torch.no_grad():
        c2w_34, intr_t = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_hw)
    c2w_34_np = tensor_to_numpy(c2w_34)[0, :num_views].astype(np.float64)
    intr_np = tensor_to_numpy(intr_t)[0, :num_views].astype(np.float64)
    c2w = np.tile(np.eye(4, dtype=np.float64), (num_views, 1, 1))
    c2w[:, :3, :] = c2w_34_np
    return np.linalg.inv(c2w), intr_np


def extract_sky_mask(predictions: dict, num_views: int) -> np.ndarray | None:
    sky = predictions.get("sky_mask")
    if sky is not None:
        sky_np = tensor_to_numpy(sky)
        return sky_np.squeeze(0).astype(bool)[:num_views]

    raw = predictions.get("raw_preds", {})
    if hasattr(raw, "get") and raw.get("mog_weight_full") is not None:
        weights = tensor_to_numpy(raw["mog_weight_full"])[0, :num_views]
        return weights.argmax(axis=-1) == (weights.shape[-1] - 1)
    return None


def write_camera_json(
    image_folder: str,
    image_paths: list[str],
    original_sizes: dict[str, tuple[int, int]],
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    processed_width: int,
    processed_height: int,
) -> str:
    all_cameras_data = {}
    for i, path in enumerate(image_paths):
        image_name = os.path.basename(path)
        orig_w, orig_h = original_sizes[image_name]
        all_cameras_data[image_name] = {
            "intrinsics": intrinsics[i].tolist(),
            "extrinsics": extrinsics_w2c[i, :3, :4].tolist(),
            "width": int(processed_width),
            "height": int(processed_height),
            "original_width": int(orig_w),
            "original_height": int(orig_h),
        }

    out_path = os.path.join(image_folder, "all_cameras_parameters.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cameras_data, f, indent=4)
    print(f"OK: 全カメラパラメータのJSON保存完了 : {out_path}")
    emit_progress("save_json", 0.85, "カメラJSON保存完了")
    return out_path


def percentile_conf_threshold(conf: np.ndarray, base_mask: np.ndarray, conf_thres: float) -> float:
    if conf_thres <= 0:
        return float("-inf")
    valid_values = conf[base_mask]
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size == 0:
        return float("inf")
    return float(np.percentile(valid_values, np.clip(conf_thres, 0.0, 100.0)))


def build_glb(
    args: argparse.Namespace,
    image_folder: str,
    depth: np.ndarray,
    conf: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    sky_mask: np.ndarray | None,
) -> str:
    base_mask = np.isfinite(depth) & (depth > 0)
    if args.mask_sky and sky_mask is not None:
        base_mask &= ~sky_mask
    if args.mask_black_bg:
        base_mask &= ~(colors.sum(axis=-1) < 16)
    if args.mask_white_bg:
        base_mask &= ~((colors[..., 0] > 240) & (colors[..., 1] > 240) & (colors[..., 2] > 240))

    conf_for_filter = conf.copy()
    conf_for_filter[~base_mask] = float("-inf")
    conf_thr = percentile_conf_threshold(conf_for_filter, base_mask, args.conf_thres)

    print()
    emit_progress("save_glb", 0.90, "GLBファイルを生成中")
    print("▶ GLBファイルを生成中... (頂点数が多いと少し時間がかかります)")
    print(
        f"   適用フィルタ: conf_thres={args.conf_thres}, mask_sky={args.mask_sky}, "
        f"mask_black_bg={args.mask_black_bg}, mask_white_bg={args.mask_white_bg}"
    )

    start = time.time()
    points, point_colors = _depths_to_world_points_with_colors(
        depth,
        intrinsics,
        extrinsics_w2c,
        colors,
        conf_for_filter,
        conf_thr,
    )
    align = _compute_alignment_transform_first_cam_glTF_center_by_points(
        extrinsics_w2c[0], points
    )
    if points.shape[0] > 0:
        points = trimesh.transform_points(points, align)
    points, point_colors = _filter_and_downsample(points, point_colors, args.num_max_points)

    scene = trimesh.Scene()
    scene.metadata = scene.metadata or {}
    scene.metadata["hf_alignment"] = align
    if points.shape[0] > 0:
        scene.add_geometry(trimesh.points.PointCloud(vertices=points, colors=point_colors))

    if args.show_cam:
        scale = _estimate_scene_scale(points, fallback=1.0) * 0.03
        h, w = depth.shape[1:]
        _add_cameras_to_scene(
            scene=scene,
            K=intrinsics,
            ext_w2c=extrinsics_w2c,
            image_sizes=[(h, w)] * depth.shape[0],
            scale=scale,
        )

    out_path = os.path.join(image_folder, "reconstructed_scene.glb")
    scene.export(out_path)
    elapsed = time.time() - start
    print(f"OK: GLB保存完了  : {out_path} (変換所要時間: {elapsed:.1f} 秒)")
    emit_progress("save_glb_done", 0.97, f"GLB保存完了 ({elapsed:.1f}s)")
    return out_path


def is_cuda_oom_error(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    text = str(error).lower()
    return "cuda out of memory" in text or "cuda error: out of memory" in text


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def run_pipeline(args: argparse.Namespace, image_folder: str, all_image_paths: list[str]) -> tuple[str, str]:
    if not all_image_paths:
        print(f"ERROR: {image_folder} フォルダに画像が見つかりません。")
        sys.exit(1)

    image_paths = select_image_subset(all_image_paths, args.max_images)
    if len(image_paths) != len(all_image_paths):
        print(
            f"INFO: --max-images {args.max_images} により "
            f"{len(all_image_paths)} 枚から {len(image_paths)} 枚を均等間引きして使用します。"
        )
        print("INFO: 使用画像:")
        for path in image_paths:
            print(f"   - {os.path.basename(path)}")

    print_config(args, len(all_image_paths), len(image_paths))
    check_environment(args)

    original_sizes = load_original_sizes(image_paths)
    predictions = run_mda_inference(args, image_paths)

    depth, conf, colors = extract_depth_conf_images(predictions, len(image_paths))
    processed_height, processed_width = depth.shape[1:]
    extrinsics_w2c, intrinsics = extract_w2c_intrinsics(
        predictions, len(image_paths), (processed_height, processed_width)
    )
    sky_mask = extract_sky_mask(predictions, len(image_paths)) if args.mask_sky else None

    emit_progress("world_unproj", 0.80, "深度マップから3D座標を計算中")
    print("▶ 深度マップから3Dワールド座標を計算中...")

    print()
    print("--- 保存処理を開始します ---")
    json_path = write_camera_json(
        image_folder=image_folder,
        image_paths=image_paths,
        original_sizes=original_sizes,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        processed_width=processed_width,
        processed_height=processed_height,
    )
    glb_path = build_glb(
        args=args,
        image_folder=image_folder,
        depth=depth,
        conf=conf,
        colors=colors,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        sky_mask=sky_mask,
    )

    clear_cuda_cache()

    emit_progress("done", 1.00, "すべて完了")
    print()
    print("=" * 68)
    print("  すべての処理が正常に完了しました")
    print("=" * 68)
    print(f"  カメラJSON          : {json_path}")
    print(f"  GLBシーン           : {glb_path}")
    print("=" * 68)
    return json_path, glb_path


def main() -> None:
    args = parse_args()
    image_folder = args.image_folder
    all_image_paths = collect_image_paths(image_folder)

    try:
        run_pipeline(args, image_folder, all_image_paths)
        return
    except Exception as error:
        if not is_cuda_oom_error(error):
            raise
        clear_cuda_cache()
        print()
        print("ERROR: CUDA out of memory が発生しました。")
        print(f"   初回サイズ: {args.size}")
        print(f"   OOM時動作 : {args.oom_action}")
        if args.oom_action != "lower-size":
            print("   再実行せず終了します。再試行する場合は --oom-action lower-size を指定してください。")
            raise
        if args.retry_size >= args.size:
            print(
                "   --retry-size が初回 --size 以上のため再実行できません。"
                " 初回より小さい値を指定してください。"
            )
            raise

    retry_args = argparse.Namespace(**vars(args))
    retry_args.size = args.retry_size
    print()
    print(
        f"INFO: --oom-action lower-size により、"
        f"size {args.size} -> {retry_args.size} で1回だけ再実行します。"
    )
    run_pipeline(retry_args, image_folder, all_image_paths)


if __name__ == "__main__":
    main()
