# -*- coding: utf-8 -*-
"""VGGT runner shared by the MDA workspace entry points.

The command-line options, progress markers, output filenames, camera JSON
schema, and GLB alignment match ``C:\aokiDev\vggt\python_vggt_customScript``.
The VGGT Python package is installed into the existing ``mda`` conda
environment by ``scripts/setup_windows_vggt.ps1``.
"""

from __future__ import annotations

import argparse
import glob
import os
import site
import sys
import time
from pathlib import Path


def emit_progress(phase: str, progress: float, message: str = "") -> None:
    print(f"▶ [PHASE:{phase}] ({progress:.2f}) {message}", flush=True)


emit_progress("startup", 0.01, "Python起動中")


def remove_user_site_packages() -> None:
    try:
        user_sites = site.getusersitepackages()
    except AttributeError:
        user_sites = []
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    user_sites = {os.path.normcase(os.path.abspath(path)) for path in user_sites}
    sys.path[:] = [
        path for path in sys.path
        if os.path.normcase(os.path.abspath(path)) not in user_sites
    ]


def resolve_mda_root() -> Path:
    candidates = []
    if os.environ.get("MDA_ROOT"):
        candidates.append(Path(os.environ["MDA_ROOT"]))
    here = Path(__file__).resolve()
    candidates.extend([here.parent.parent, *here.parents])
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "demo.py").is_file() and (root / "src").is_dir():
            return root
    raise RuntimeError("MDA root was not found. Set MDA_ROOT to the MDA repository directory.")


remove_user_site_packages()
MDA_ROOT = resolve_mda_root()
sys.path.insert(0, str(MDA_ROOT))
os.chdir(MDA_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from depth_anything_3.utils.export.glb import (  # noqa: E402
    _add_cameras_to_scene,
    _estimate_scene_scale,
)
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")


def parse_args(commercial: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VGGT 3D再構成スクリプト [商用モデル版]" if commercial
            else "VGGT 3D再構成スクリプト"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-folder", "-i", default="my_images")
    parser.add_argument(
        "--conf-thres", "-c", type=float, default=60.0,
        help="信頼度フィルタ。下位X%%を除外し、上位(100-X)%%を保持する。",
    )
    sky_group = parser.add_mutually_exclusive_group()
    sky_group.add_argument("--mask-sky", dest="mask_sky", action="store_true")
    sky_group.add_argument("--no-mask-sky", dest="mask_sky", action="store_false")
    parser.set_defaults(mask_sky=True)
    parser.add_argument("--mask-black-bg", action="store_true")
    parser.add_argument("--mask-white-bg", action="store_true")
    parser.add_argument("--no-show-cam", dest="show_cam", action="store_false")
    parser.set_defaults(show_cam=True)
    parser.add_argument(
        "--prediction-mode", default="Depthmap and Camera Branch",
        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="モデル重みだけをダウンロードして終了する。",
    )
    args = parser.parse_args()
    if not 0.0 <= args.conf_thres <= 100.0:
        parser.error("--conf-thres must be between 0 and 100")
    return args


def collect_image_paths(image_folder: str) -> list[str]:
    return sorted(
        path for pattern in IMAGE_EXTS
        for path in glob.glob(os.path.join(image_folder, pattern))
    )


def print_config(args: argparse.Namespace, num_images: int, model_id: str, commercial: bool) -> None:
    print("\n" + "=" * 68)
    print("  VGGT 3D再構成" + ("  [商用モデル版]" if commercial else ""))
    print("=" * 68)
    print(f"  使用モデル              : {model_id}")
    print(f"  入力画像フォルダ        : {args.image_folder}")
    print(f"  検出した画像枚数        : {num_images} 枚")
    print(f"  信頼度フィルタ          : 下位 {args.conf_thres:.1f}% を除外 / 上位 {100.0 - args.conf_thres:.1f}% を保持")
    print(f"  空マスク                : {'ON' if args.mask_sky else 'OFF'}")
    print(f"  黒背景マスク            : {'ON' if args.mask_black_bg else 'OFF'}")
    print(f"  白背景マスク            : {'ON' if args.mask_white_bg else 'OFF'}")
    print(f"  カメラ可視化            : {'ON' if args.show_cam else 'OFF'}")
    print(f"  予測モード              : {args.prediction_mode}")
    print("=" * 68 + "\n")


def check_environment(args: argparse.Namespace) -> None:
    emit_progress("env_check", 0.02, "環境チェック中")
    args._sky_disabled_reason = None
    if torch.cuda.is_available():
        print(f"   CUDA           : 使用可能 ({torch.cuda.get_device_name(0)})")
    else:
        print("   CUDA           : 使用不可（CPUで実行）")
    if args.mask_sky:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            args.mask_sky = False
            args._sky_disabled_reason = "missing_onnxruntime"
            print("   onnxruntime    : 未導入のため空マスクを無効化")


def prepare_sky_mask_dir(image_folder: str, image_paths: list[str]) -> Path:
    images_dir = Path(image_folder) / "images"
    images_dir.mkdir(exist_ok=True)
    for image_path in image_paths:
        source = Path(image_path).resolve()
        destination = images_dir / source.name
        if destination.exists():
            continue
        try:
            os.link(source, destination)
        except OSError:
            import shutil
            shutil.copy2(source, destination)
    return images_dir


def sky_checkpoint_path() -> Path:
    path = MDA_ROOT / "checkpoints" / "VGGT" / "skyseg.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download_sky_checkpoint() -> Path:
    path = sky_checkpoint_path()
    if path.is_file():
        return path
    print("▶ skyseg.onnx をダウンロード中...")
    response = requests.get(
        "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    temporary = path.with_suffix(".onnx.part")
    with open(temporary, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    temporary.replace(path)
    return path


def segment_sky(image_path: Path, session, output_path: Path, output_hw: tuple[int, int]) -> np.ndarray:
    if output_path.is_file():
        existing = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        if existing is not None:
            return cv2.resize(existing, output_hw)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for sky mask: {image_path}")
    resized = cv2.resize(image, (320, 320))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb / 255.0 - mean) / std
    tensor = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    mask = np.asarray(session.run([output_name], {input_name: tensor})).squeeze()
    minimum, maximum = float(mask.min()), float(mask.max())
    if maximum > minimum:
        mask = (mask - minimum) / (maximum - minimum)
    else:
        mask = np.zeros_like(mask)
    mask = (mask * 255.0).astype(np.uint8)
    # The original VGGT custom script stores 255 for non-sky and 0 for sky.
    mask = np.where(mask < 32, 255, 0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
    return cv2.resize(mask, output_hw)


def make_sky_mask(image_folder: str, image_paths: list[str], width: int, height: int) -> np.ndarray:
    import onnxruntime

    image_dir = prepare_sky_mask_dir(image_folder, image_paths)
    session = onnxruntime.InferenceSession(str(download_sky_checkpoint()))
    mask_dir = Path(image_folder) / "sky_masks"
    masks = [
        segment_sky(image_dir / Path(path).name, session, mask_dir / Path(path).name, (width, height))
        for path in image_paths
    ]
    return np.asarray(masks) > 0


def as_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic
    result = np.eye(4, dtype=extrinsic.dtype)
    result[:3, :4] = extrinsic
    return result


def vggt_alignment(first_w2c: np.ndarray) -> np.ndarray:
    c_gl = np.eye(4, dtype=np.float64)
    c_gl[1, 1] = -1.0
    c_gl[2, 2] = -1.0
    r_y180 = np.eye(4, dtype=np.float64)
    r_y180[0, 0] = -1.0
    r_y180[2, 2] = -1.0
    return np.linalg.inv(as_homogeneous(first_w2c).astype(np.float64)) @ c_gl @ r_y180


def build_glb(
    args: argparse.Namespace,
    image_folder: str,
    predictions: dict,
    width: int,
    height: int,
) -> str:
    if args.prediction_mode == "Pointmap Branch" and "world_points" in predictions:
        world_points = predictions["world_points"]
        confidence = predictions.get("world_points_conf", np.ones(world_points.shape[:-1]))
    else:
        world_points = predictions["world_points_from_depth"]
        confidence = predictions.get("depth_conf", np.ones(world_points.shape[:-1]))

    images = predictions["images"]
    extrinsics = predictions["extrinsic"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    colors = (np.clip(images, 0.0, 1.0).reshape(-1, 3) * 255.0).astype(np.uint8)
    confidence = np.asarray(confidence, dtype=np.float32)
    if args.mask_sky:
        confidence = confidence.copy()
        confidence *= make_sky_mask(image_folder, predictions["image_paths"], width, height)
    threshold = 0.0 if args.conf_thres == 0.0 else np.percentile(confidence, args.conf_thres)
    mask = (confidence.reshape(-1) >= threshold) & (confidence.reshape(-1) > 1e-5)
    if args.mask_black_bg:
        mask &= colors.sum(axis=1) >= 16
    if args.mask_white_bg:
        mask &= ~np.all(colors > 240, axis=1)
    points = np.asarray(world_points).reshape(-1, 3)[mask]
    colors = colors[mask]
    if points.size:
        points = trimesh.transform_points(points, vggt_alignment(extrinsics[0]))
    scene = trimesh.Scene()
    scene.metadata = scene.metadata or {}
    scene.metadata["hf_alignment"] = vggt_alignment(extrinsics[0])
    if points.size:
        scene.add_geometry(trimesh.points.PointCloud(vertices=points, colors=colors))
    if args.show_cam:
        _add_cameras_to_scene(
            scene=scene,
            K=predictions["intrinsic"],
            ext_w2c=extrinsics,
            image_sizes=[(height, width)] * len(extrinsics),
            scale=_estimate_scene_scale(points, fallback=1.0) * 0.03,
        )
    out_path = os.path.join(image_folder, "reconstructed_scene.glb")
    scene.export(out_path)
    return out_path


def write_camera_json(
    image_folder: str,
    image_paths: list[str],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
    model_id: str,
) -> str:
    from camera_parameters_json import (
        build_vggt_image_transforms,
        write_camera_parameters_json,
    )

    out_path = os.path.join(image_folder, "all_cameras_parameters.json")
    write_camera_parameters_json(
        out_path,
        image_paths,
        intrinsics,
        extrinsics,
        width,
        height,
        build_vggt_image_transforms(image_paths, width, height),
        provider="vggt",
        model_id=model_id,
        intrinsics_source="predicted_fov",
        principal_point_source="fixed_image_center",
    )
    return out_path


def load_model(model_id: str, model_display_name: str, commercial: bool):
    emit_progress("model_load", 0.10, f"{model_display_name} を読み込み中")
    print(f"▶ {model_display_name} を読み込み中...")
    try:
        model = VGGT.from_pretrained(model_id)
    except Exception as error:
        message = str(error).lower()
        if commercial and any(word in message for word in ("401", "403", "gated", "access", "authoriz", "permission", "unauthorized")):
            print("ERROR: 商用モデルへのアクセスが拒否されました。")
            print(f"  https://huggingface.co/{model_id} で規約に同意し、huggingface-cli login を実行してください。")
            raise SystemExit(2) from error
        raise
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.eval().to(device), device


def main(model_id: str, model_display_name: str, commercial: bool) -> None:
    args = parse_args(commercial)
    if args.download_only:
        load_model(model_id, model_display_name, commercial)
        emit_progress("done", 1.00, "モデル重みのダウンロード完了")
        return

    image_paths = collect_image_paths(args.image_folder)
    if not image_paths:
        print(f"ERROR: {args.image_folder} フォルダに画像が見つかりません。")
        raise SystemExit(1)
    print_config(args, len(image_paths), model_id, commercial)
    check_environment(args)
    model, device = load_model(model_id, model_display_name, commercial)
    emit_progress("preprocess", 0.20, f"画像を前処理中 ({len(image_paths)}枚)")
    images = load_and_preprocess_images(image_paths).to(device)
    height, width = images.shape[-2:]
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    emit_progress("infer", 0.30, "AI推論を実行中")
    started = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            prediction_tensors = model(images)
    inference_seconds = time.time() - started
    emit_progress("infer_done", 0.70, f"推論完了 ({inference_seconds:.1f}s)")

    emit_progress("camera_calc", 0.75, "カメラパラメータ計算中")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction_tensors["pose_enc"], images.shape[-2:])
    predictions = {
        key: value.detach().cpu().numpy().squeeze(0)
        for key, value in prediction_tensors.items()
        if isinstance(value, torch.Tensor)
    }
    predictions["extrinsic"] = extrinsic.detach().cpu().numpy().squeeze(0)
    predictions["intrinsic"] = intrinsic.detach().cpu().numpy().squeeze(0)
    image_array = images.detach().cpu().numpy()
    # load_and_preprocess_images() returns [S, C, H, W] in the current VGGT
    # package, while older variants return [1, S, C, H, W].  Preserve the
    # sequence dimension and only remove an actual singleton batch dimension.
    predictions["images"] = image_array[0] if image_array.shape[0] == 1 and image_array.ndim == 5 else image_array
    predictions["image_paths"] = image_paths
    emit_progress("world_unproj", 0.80, "深度マップから3D座標を計算中")
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )

    json_path = write_camera_json(args.image_folder, image_paths, predictions["intrinsic"], predictions["extrinsic"], width, height, model_id)
    emit_progress("save_json", 0.85, "カメラJSON保存完了")
    emit_progress("save_glb", 0.90, "GLBファイルを生成中")
    glb_path = build_glb(args, args.image_folder, predictions, width, height)
    emit_progress("save_glb_done", 0.97, "GLB保存完了")
    emit_progress("done", 1.00, "すべて完了")
    print(f"カメラJSON: {json_path}")
    print(f"GLBシーン : {glb_path}")
