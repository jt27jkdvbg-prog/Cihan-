
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wan_kaggle_reference_loader import (
    WanReferenceLoadError,
    load_reference_bundle,
)


class WanKaggleReferenceLoaderTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.asset_dir = self.root / "reference_assets"
        self.asset_dir.mkdir()

        self.image = self.asset_dir / "hero.jpg"
        self.image.write_bytes(
            b"\xff\xd8\xff" + b"verified-reference-image" * 400
        )

        self.digest = hashlib.sha256(
            self.image.read_bytes()
        ).hexdigest()

        self.bundle = {
            "schema_version": 1,
            "project_id": "loader-test",
            "scene_count": 1,
            "unique_reference_count": 1,
            "scene_reference_count": 1,
            "assets": [
                {
                    "character_id": "hero",
                    "character_name": "Hero",
                    "path": "reference_assets/hero.jpg",
                    "sha256": self.digest,
                    "format": "jpg",
                    "bytes": self.image.stat().st_size,
                }
            ],
            "scenes": [
                {
                    "scene_id": "scene-1",
                    "reference_images": [
                        {
                            "character_id": "hero",
                            "character_name": "Hero",
                            "path": "reference_assets/hero.jpg",
                            "sha256": self.digest,
                            "format": "jpg",
                        }
                    ],
                }
            ],
        }

        self.write_bundle()

    def tearDown(self):
        self.temp.cleanup()

    def write_bundle(self):
        (self.root / "reference_bundle.json").write_text(
            json.dumps(self.bundle),
            encoding="utf-8",
        )

    def test_loads_and_verifies_reference_images(self):
        result = load_reference_bundle(
            self.root,
            expected_project_id="loader-test",
        )

        self.assertEqual(
            result["runtime_validation"]["status"],
            "verified",
        )
        self.assertEqual(
            result["runtime_validation"]["images_verified"],
            1,
        )
        self.assertEqual(
            result["runtime_validation"]["scene_references_verified"],
            1,
        )

        local_path = result["scenes"][0]["reference_images"][0]["local_path"]
        self.assertTrue(Path(local_path).is_file())

    def test_rejects_wrong_project_id(self):
        with self.assertRaises(WanReferenceLoadError):
            load_reference_bundle(
                self.root,
                expected_project_id="another-project",
            )

    def test_rejects_modified_image(self):
        self.image.write_bytes(
            b"\xff\xd8\xff" + b"modified-image-data" * 400
        )

        with self.assertRaises(WanReferenceLoadError):
            load_reference_bundle(self.root)

    def test_rejects_missing_image(self):
        self.image.unlink()

        with self.assertRaises(WanReferenceLoadError):
            load_reference_bundle(self.root)

    def test_rejects_unsafe_asset_path(self):
        self.bundle["assets"][0]["path"] = "../hero.jpg"
        self.write_bundle()

        with self.assertRaises(WanReferenceLoadError):
            load_reference_bundle(self.root)

    def test_rejects_unregistered_scene_reference(self):
        self.bundle["scenes"][0]["reference_images"][0]["sha256"] = (
            "0" * 64
        )
        self.write_bundle()

        with self.assertRaises(WanReferenceLoadError):
            load_reference_bundle(self.root)

    def test_rejects_scene_character_mismatch(self):
        self.bundle["scenes"][0]["reference_images"][0]["character_id"] = (
            "unknown"
        )
        self.write_bundle()

        with self.assertRaises(WanReferenceLoadError):
            load_reference_bundle(self.root)

    def test_rejects_invalid_manifest(self):
        (self.root / "reference_bundle.json").write_text(
            "{invalid json",
            encoding="utf-8",
        )

        with self.assertRaises(WanReferenceLoadError):
            load_reference_bundle(self.root)


if __name__ == "__main__":
    unittest.main()
