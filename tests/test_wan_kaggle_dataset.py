
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wan_kaggle_dataset import (
    WanDatasetError,
    prepare_kaggle_dataset,
)


class WanKaggleDatasetTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.payload = self.root / "payload"
        self.assets = self.payload / "reference_assets"
        self.assets.mkdir(parents=True)

        self.image = self.assets / "hero.jpg"
        self.image.write_bytes(
            b"\xff\xd8\xff" + b"valid-test-image-data" * 400
        )

        self.digest = hashlib.sha256(
            self.image.read_bytes()
        ).hexdigest()

        self.bundle = {
            "schema_version": 1,
            "project_id": "dataset-test",
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
        (self.payload / "reference_bundle.json").write_text(
            json.dumps(self.bundle),
            encoding="utf-8",
        )

    def test_creates_private_kaggle_dataset_folder(self):
        output = self.root / "dataset-upload"

        metadata_path = prepare_kaggle_dataset(
            self.payload,
            output,
            "test-owner",
            "wan-test-123",
        )

        self.assertTrue(metadata_path.is_file())

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        self.assertEqual(
            metadata["id"],
            "test-owner/wan-test-123",
        )
        self.assertEqual(metadata["licenses"], [{"name": "other"}])

        copied_image = output / "reference_assets" / "hero.jpg"

        self.assertTrue(copied_image.is_file())
        self.assertEqual(
            hashlib.sha256(copied_image.read_bytes()).hexdigest(),
            self.digest,
        )

    def test_rejects_modified_asset(self):
        self.image.write_bytes(
            b"\xff\xd8\xff" + b"changed-image-data" * 400
        )

        with self.assertRaises(WanDatasetError):
            prepare_kaggle_dataset(
                self.payload,
                self.root / "output",
                "test-owner",
                "wan-test-123",
            )

    def test_rejects_missing_bundle(self):
        (self.payload / "reference_bundle.json").unlink()

        with self.assertRaises(WanDatasetError):
            prepare_kaggle_dataset(
                self.payload,
                self.root / "output",
                "test-owner",
                "wan-test-123",
            )

    def test_rejects_invalid_owner(self):
        with self.assertRaises(WanDatasetError):
            prepare_kaggle_dataset(
                self.payload,
                self.root / "output",
                "bad/owner",
                "wan-test-123",
            )

    def test_rejects_invalid_slug(self):
        with self.assertRaises(WanDatasetError):
            prepare_kaggle_dataset(
                self.payload,
                self.root / "output",
                "test-owner",
                "../unsafe",
            )

    def test_rejects_scene_reference_without_asset(self):
        self.bundle["scenes"][0]["reference_images"][0]["sha256"] = (
            "0" * 64
        )
        self.write_bundle()

        with self.assertRaises(WanDatasetError):
            prepare_kaggle_dataset(
                self.payload,
                self.root / "output",
                "test-owner",
                "wan-test-123",
            )

    def test_rejects_existing_output_directory(self):
        output = self.root / "existing"
        output.mkdir()

        with self.assertRaises(WanDatasetError):
            prepare_kaggle_dataset(
                self.payload,
                output,
                "test-owner",
                "wan-test-123",
            )


if __name__ == "__main__":
    unittest.main()
