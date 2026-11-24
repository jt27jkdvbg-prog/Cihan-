
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wan_kaggle_payload import (
    WanPayloadError,
    prepare_reference_payload,
)


class WanKagglePayloadTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

        self.image_dir = self.workspace / "artifacts" / "character_assets"
        self.image_dir.mkdir(parents=True)

        self.image = self.image_dir / "hero.jpg"
        self.image.write_bytes(
            b"\xff\xd8\xff" + b"fake-jpeg-test-data" * 400
        )

        self.image_hash = hashlib.sha256(
            self.image.read_bytes()
        ).hexdigest()

        self.plan_path = (
            self.workspace
            / "artifacts"
            / "wan"
            / "character_reference_plan.json"
        )
        self.plan_path.parent.mkdir(parents=True)

        self.plan = {
            "schema_version": 1,
            "status": "ready",
            "project_id": "payload-test",
            "project_title": "Payload Test",
            "engine": "wan-vace-r2v",
            "model_profile": "Wan2.1-VACE-1.3B",
            "conditioning_mode": "reference_to_video",
            "scenes": [
                {
                    "scene_id": "scene-1",
                    "order": 1,
                    "duration_seconds": 5,
                    "character_ids": ["hero"],
                    "reference_images": [
                        {
                            "character_id": "hero",
                            "character_name": "Hero",
                            "path": "artifacts/character_assets/hero.jpg",
                            "sha256": self.image_hash,
                            "format": "jpg",
                        }
                    ],
                    "visual_prompt": "A fictional character walks.",
                },
                {
                    "scene_id": "scene-2",
                    "order": 2,
                    "duration_seconds": 5,
                    "character_ids": ["hero"],
                    "reference_images": [
                        {
                            "character_id": "hero",
                            "character_name": "Hero",
                            "path": "artifacts/character_assets/hero.jpg",
                            "sha256": self.image_hash,
                            "format": "jpg",
                        }
                    ],
                    "visual_prompt": "The same character enters a room.",
                },
            ],
        }

        self.write_plan()

    def tearDown(self):
        self.temp.cleanup()

    def write_plan(self):
        self.plan_path.write_text(
            json.dumps(self.plan),
            encoding="utf-8",
        )

    def test_creates_self_contained_bundle(self):
        output = self.root / "kaggle-payload"

        manifest_path = prepare_reference_payload(
            self.plan_path,
            self.workspace,
            output,
        )

        self.assertTrue(manifest_path.is_file())

        bundle = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        self.assertEqual(bundle["project_id"], "payload-test")
        self.assertEqual(bundle["scene_count"], 2)
        self.assertEqual(bundle["unique_reference_count"], 1)
        self.assertEqual(bundle["scene_reference_count"], 2)

        copied = output / bundle["assets"][0]["path"]

        self.assertTrue(copied.is_file())
        self.assertEqual(
            hashlib.sha256(copied.read_bytes()).hexdigest(),
            self.image_hash,
        )

    def test_deduplicates_repeated_character_images(self):
        output = self.root / "payload"

        prepare_reference_payload(
            self.plan_path,
            self.workspace,
            output,
        )

        files = list((output / "reference_assets").iterdir())

        self.assertEqual(len(files), 1)

    def test_rejects_modified_source_image(self):
        self.image.write_bytes(
            b"\xff\xd8\xff" + b"modified-image-data" * 400
        )

        with self.assertRaises(WanPayloadError):
            prepare_reference_payload(
                self.plan_path,
                self.workspace,
                self.root / "payload",
            )

    def test_rejects_unsafe_reference_path(self):
        self.plan["scenes"][0]["reference_images"][0]["path"] = (
            "../../outside.jpg"
        )
        self.write_plan()

        with self.assertRaises(WanPayloadError):
            prepare_reference_payload(
                self.plan_path,
                self.workspace,
                self.root / "payload",
            )

    def test_rejects_missing_reference_image(self):
        self.image.unlink()

        with self.assertRaises(WanPayloadError):
            prepare_reference_payload(
                self.plan_path,
                self.workspace,
                self.root / "payload",
            )

    def test_rejects_invalid_plan_status(self):
        self.plan["status"] = "blocked"
        self.write_plan()

        with self.assertRaises(WanPayloadError):
            prepare_reference_payload(
                self.plan_path,
                self.workspace,
                self.root / "payload",
            )

    def test_rejects_existing_output_directory(self):
        output = self.root / "payload"
        output.mkdir()

        with self.assertRaises(WanPayloadError):
            prepare_reference_payload(
                self.plan_path,
                self.workspace,
                output,
            )


if __name__ == "__main__":
    unittest.main()
