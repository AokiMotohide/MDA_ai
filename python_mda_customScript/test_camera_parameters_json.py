from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from python_mda_customScript.camera_parameters_json import (
    build_mda_image_transforms,
    build_vggt_image_transforms,
    write_camera_parameters_json,
)


class CameraParametersJsonTest(unittest.TestCase):
    def test_vggt_transform_includes_batch_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            landscape = root / "landscape.png"
            square = root / "square.png"
            Image.new("RGB", (1920, 1080)).save(landscape)
            Image.new("RGB", (1000, 1000)).save(square)
            transforms = build_vggt_image_transforms([landscape, square], 518, 518)
            self.assertEqual(transforms[0][1, 2], 112.0)
            self.assertEqual(transforms[1][1, 2], 0.0)

    def test_mda_transform_records_common_aspect_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            landscape = root / "landscape.png"
            portrait = root / "portrait.png"
            Image.new("RGB", (1600, 900)).save(landscape)
            Image.new("RGB", (900, 1600)).save(portrait)
            transforms = build_mda_image_transforms(
                [landscape, portrait], 518, 14, 518, 280
            )
            self.assertAlmostEqual(transforms[0][0, 2], 0.0)
            self.assertLess(transforms[1][1, 2], 0.0)

    def test_writer_retains_legacy_fields_and_adds_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "camera.png"
            Image.new("RGB", (100, 80)).save(image_path)
            intrinsics = np.array(
                [[[50.0, 0.0, 45.0], [0.0, 51.0, 39.0], [0.0, 0.0, 1.0]]]
            )
            extrinsics = np.eye(4)[None]
            output_path = root / "all_cameras_parameters.json"
            write_camera_parameters_json(
                output_path,
                [image_path],
                intrinsics,
                extrinsics,
                100,
                80,
                [np.eye(3)],
                provider="mda",
                model_id="mda-test",
                intrinsics_source="raw_predictions",
                principal_point_source="estimated",
            )
            camera = json.loads(output_path.read_text(encoding="utf-8"))["camera.png"]
            self.assertEqual(camera["schema_version"], 2)
            self.assertEqual(np.asarray(camera["extrinsics"]).shape, (3, 4))
            self.assertEqual(camera["provenance"]["provider"], "mda")
            self.assertEqual(camera["provenance"]["principal_point_source"], "estimated")
            self.assertEqual(camera["scene_transform"]["scene"], "glb")


if __name__ == "__main__":
    unittest.main()
