"""Camera Parameters v2 writer shared by the MDA VGGT and DA3 runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose


def _image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return exif_transpose(image).size


def build_vggt_image_transforms(
    image_paths: Sequence[str | Path], processed_width: int, processed_height: int
) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    for path in image_paths:
        original_width, original_height = _image_size(path)
        resized_width = 518
        resized_height = round(original_height * (518 / original_width) / 14) * 14
        crop_top = max((resized_height - 518) // 2, 0)
        cropped_height = min(resized_height, 518)
        pad_left = max((processed_width - resized_width) // 2, 0)
        pad_top = max((processed_height - cropped_height) // 2, 0)
        transforms.append(
            np.array(
                [[resized_width / original_width, 0.0, float(pad_left)],
                 [0.0, resized_height / original_height, float(pad_top - crop_top)],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        )
    return transforms


def _mda_loaded_size(
    original_width: int, original_height: int, size: int, patch_size: int
) -> tuple[int, int]:
    scale = size / max(original_width, original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    cx, cy = resized_width // 2, resized_height // 2
    half_width = ((2 * cx) // patch_size) * (patch_size // 2)
    half_height = ((2 * cy) // patch_size) * (patch_size // 2)
    return 2 * half_width, 2 * half_height


def build_mda_image_transforms(
    image_paths: Sequence[str | Path],
    size: int,
    patch_size: int,
    processed_width: int,
    processed_height: int,
) -> list[np.ndarray]:
    """Reproduce prepare_views' resize, center crop and common-size resize."""

    if min(size, patch_size, processed_width, processed_height) <= 0:
        raise ValueError("前処理設定と解像度は正の値である必要があります")
    target_aspect = processed_height / processed_width
    transforms: list[np.ndarray] = []
    for path in image_paths:
        original_width, original_height = _image_size(path)
        loaded_width, loaded_height = _mda_loaded_size(
            original_width, original_height, size, patch_size
        )
        if loaded_height / loaded_width > target_aspect:
            crop_height = int(round(loaded_width * target_aspect))
            crop_width = loaded_width
        else:
            crop_height = loaded_height
            crop_width = int(round(loaded_height / target_aspect))
        left = (loaded_width - crop_width) // 2
        top = (loaded_height - crop_height) // 2
        output_scale_x = processed_width / crop_width
        output_scale_y = processed_height / crop_height
        transforms.append(
            np.array(
                [[output_scale_x * loaded_width / original_width,
                  0.0, -left * output_scale_x],
                 [0.0,
                  output_scale_y * loaded_height / original_height,
                  -top * output_scale_y],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        )
    return transforms


def write_camera_parameters_json(
    output_path: str | Path,
    image_paths: Sequence[str | Path],
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    processed_width: int,
    processed_height: int,
    image_transforms: Sequence[np.ndarray],
    *,
    provider: str,
    model_id: str,
    intrinsics_source: str,
    principal_point_source: str,
) -> Path:
    paths = [Path(path) for path in image_paths]
    k_matrices = np.asarray(intrinsics, dtype=np.float64)
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsics はカメラごとの3x4または4x4行列である必要があります")
    if k_matrices.shape != (len(paths), 3, 3) or len(extrinsics) != len(paths):
        raise ValueError("画像、intrinsics、extrinsics の件数または形状が一致しません")
    if len(image_transforms) != len(paths):
        raise ValueError("画像とimage_transformsの件数が一致しません")
    if not np.isfinite(k_matrices).all() or not np.isfinite(extrinsics).all():
        raise ValueError("カメラ行列に有限値ではない要素があります")

    e4 = np.zeros((len(extrinsics), 4, 4), dtype=np.float64)
    e4[:, :3, :4] = extrinsics[:, :3, :4]
    e4[:, 3, 3] = 1.0
    if np.any(np.abs(np.linalg.det(e4)) < 1e-8):
        raise ValueError("extrinsics に逆行列を計算できない姿勢があります")
    names = [path.name for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("画像ファイル名が重複しています")

    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    align_y_180 = np.diag([-1.0, 1.0, -1.0, 1.0])
    scene_alignment = np.linalg.inv(e4[0]) @ opencv_to_opengl @ align_y_180
    payload: dict[str, dict[str, object]] = {}
    for index, path in enumerate(paths):
        original_width, original_height = _image_size(path)
        transform = np.asarray(image_transforms[index], dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("image_transform は有限数の3x3行列である必要があります")
        camera_to_scene = scene_alignment @ np.linalg.inv(e4[index]) @ opencv_to_opengl
        payload[path.name] = {
            "schema_version": 2,
            "camera_model": "pinhole",
            "intrinsics": k_matrices[index].tolist(),
            "extrinsics": e4[index, :3, :4].tolist(),
            "extrinsics_convention": {
                "matrix": "world_to_camera",
                "camera_axes": "opencv",
                "layout": "row_major",
            },
            "width": int(processed_width),
            "height": int(processed_height),
            "original_width": int(original_width),
            "original_height": int(original_height),
            "distortion": {"model": "none", "coefficients": {}},
            "image_transform": {
                "original_to_processed": transform.tolist(),
                "pixel_center": "integer_center",
            },
            "scene_transform": {
                "camera_to_scene": camera_to_scene.tolist(),
                "camera_axes": "opengl",
                "scene": "glb",
            },
            "provenance": {
                "provider": provider,
                "model_id": model_id,
                "intrinsics_source": intrinsics_source,
                "principal_point_source": principal_point_source,
            },
        }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return destination
