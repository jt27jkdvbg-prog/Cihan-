
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from production_orchestrator import ProductionJob
from wan_orchestrator_bridge import (
    CharacterAssetRegistrationError,
    register_wan_output,
)


def make_project(with_character=False):
    characters = []
    character_ids = []

    if with_character:
        characters = [
            {
                "id": "main-hero",
                "name": "Main Hero",
                "visual_description": "A fictional adult character.",
                "appearance": "Short dark hair.",
                "clothing": "Dark blue jacket.",
            }
        ]
        character_ids = ["main-hero"]

    return {
        "project_id": "wan-bridge-test-001",
        "title": "Wan Bridge Test",
        "brief": "Testing the Wan integration bridge.",
        "target_duration_seconds": 10,
        "visual_style": "cinematic realistic",
        "profile": "landscape",
        "language": "tr",
        "characters": characters,
        "scenes": [
            {
                "id": "scene_1",
                "order": 1,
                "duration_seconds": 10,
                "visual_prompt": "A cinematic city at sunset.",
                "character_ids": character_ids,
            }
        ],
    }


class WanOrchestratorBridgeTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.workspace = self.root / "job"

        self.video = self.root / "film.mp4"
        self.video.write_bytes(b"fake-video-content")

        self.report = self.root / "performance_report.json"
        self.report.write_text(
            json.dumps({
                "status": "completed",
                "scenes": [],
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def register_job(self, project=None, assets_dir=None):
        return register_wan_output(
            project or make_project(),
            self.workspace,
            self.video,
            self.report,
            character_assets_dir=assets_dir,
        )

    def make_character_assets(self):
        root = self.root / "character-assets"
        character_dir = root / "main-hero"
        character_dir.mkdir(parents=True)

        image = character_dir / "anchor.jpg"
        image.write_bytes(
            b"\xff\xd8\xff" + (b"character-image-data" * 500)
        )

        manifest = {
            "schema_version": 1,
            "project_id": "wan-bridge-test-001",
            "project_title": "Wan Bridge Test",
            "asset_count": 1,
            "all_assets_validated": True,
            "assets": [
                {
                    "character_id": "main-hero",
                    "character_name": "Main Hero",
                    "asset_type": "canonical_character_anchor",
                    "path": "main-hero/anchor.jpg",
                    "engine": "fal",
                    "bytes": image.stat().st_size,
                    "sha256": hashlib.sha256(
                        image.read_bytes()
                    ).hexdigest(),
                    "validated": True,
                }
            ],
        }

        (root / "character_assets.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        return root, image

    def test_registers_real_wan_outputs(self):
        job = self.register_job()
        manifest = job.manifest

        self.assertEqual(
            manifest["stages"]["project_validation"]["status"],
            "succeeded",
        )

        self.assertEqual(
            manifest["stages"]["script"]["status"],
            "succeeded",
        )

        self.assertEqual(
            manifest["stages"]["video_generation"]["status"],
            "succeeded",
        )

        copied_video = (
            self.workspace / "artifacts" / "wan" / "final-film.mp4"
        )

        copied_report = (
            self.workspace
            / "artifacts"
            / "wan"
            / "performance_report.json"
        )

        self.assertEqual(
            copied_video.read_bytes(),
            self.video.read_bytes(),
        )

        self.assertEqual(
            json.loads(copied_report.read_text(encoding="utf-8")),
            {"status": "completed", "scenes": []},
        )

        self.assertEqual(job.verify_artifacts(), [])

    def test_character_stage_is_not_required_for_empty_project(self):
        job = self.register_job()

        self.assertEqual(
            job.manifest["stages"]["character_assets"]["status"],
            "succeeded",
        )

        marker = (
            self.workspace
            / "artifacts"
            / "inputs"
            / "character_assets_not_required.json"
        )

        self.assertTrue(marker.is_file())

        data = json.loads(marker.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "not_required")

    def test_character_assets_are_registered_when_valid(self):
        assets_dir, _ = self.make_character_assets()

        job = self.register_job(
            project=make_project(with_character=True),
            assets_dir=assets_dir,
        )

        stage = job.manifest["stages"]["character_assets"]

        self.assertEqual(stage["status"], "succeeded")
        self.assertEqual(len(stage["artifacts"]), 2)

        copied_manifest = (
            self.workspace
            / "artifacts"
            / "character_assets"
            / "character_assets.json"
        )

        self.assertTrue(copied_manifest.is_file())

        copied_data = json.loads(
            copied_manifest.read_text(encoding="utf-8")
        )

        self.assertTrue(
            copied_data["registered_in_production_job"]
        )

        copied_image = (
            self.workspace
            / copied_data["assets"][0]["path"]
        )

        self.assertTrue(copied_image.is_file())
        self.assertEqual(job.verify_artifacts(), [])

    def test_character_assets_remain_pending_without_manifest(self):
        job = self.register_job(
            project=make_project(with_character=True)
        )

        self.assertEqual(
            job.manifest["stages"]["character_assets"]["status"],
            "pending",
        )

    def test_invalid_character_hash_is_rejected(self):
        assets_dir, _ = self.make_character_assets()

        manifest_path = assets_dir / "character_assets.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["assets"][0]["sha256"] = "0" * 64

        manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        with self.assertRaises(CharacterAssetRegistrationError):
            self.register_job(
                project=make_project(with_character=True),
                assets_dir=assets_dir,
            )

    def test_missing_character_image_is_rejected(self):
        assets_dir, image = self.make_character_assets()
        image.unlink()

        with self.assertRaises(CharacterAssetRegistrationError):
            self.register_job(
                project=make_project(with_character=True),
                assets_dir=assets_dir,
            )

    def test_wrong_character_id_is_rejected(self):
        assets_dir, _ = self.make_character_assets()

        manifest_path = assets_dir / "character_assets.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["assets"][0]["character_id"] = "unexpected"

        manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        with self.assertRaises(CharacterAssetRegistrationError):
            self.register_job(
                project=make_project(with_character=True),
                assets_dir=assets_dir,
            )

    def test_unsafe_character_asset_path_is_rejected(self):
        assets_dir, _ = self.make_character_assets()

        manifest_path = assets_dir / "character_assets.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["assets"][0]["path"] = "../outside.jpg"

        manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        with self.assertRaises(CharacterAssetRegistrationError):
            self.register_job(
                project=make_project(with_character=True),
                assets_dir=assets_dir,
            )

    def test_audio_and_post_production_remain_pending(self):
        job = self.register_job()
        manifest = job.manifest

        for stage_name in (
            "audio_generation",
            "post_production",
            "quality_control",
            "publishing",
        ):
            self.assertEqual(
                manifest["stages"][stage_name]["status"],
                "pending",
            )

    def test_job_status_is_queued_when_no_stage_is_running(self):
        job = self.register_job()

        self.assertEqual(job.manifest["status"], "queued")

    def test_video_can_complete_without_character_assets(self):
        job = self.register_job(
            project=make_project(with_character=True)
        )

        self.assertEqual(
            job.manifest["stages"]["video_generation"]["status"],
            "succeeded",
        )

        self.assertEqual(
            job.manifest["stages"]["character_assets"]["status"],
            "pending",
        )

    def test_post_production_cannot_start_before_audio_and_assets(self):
        job = self.register_job(
            project=make_project(with_character=True)
        )

        with self.assertRaises(Exception):
            job.start_stage("post_production")

        self.assertEqual(
            job.manifest["stages"]["post_production"]["status"],
            "pending",
        )

    def test_missing_video_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            register_wan_output(
                make_project(),
                self.workspace,
                self.root / "missing.mp4",
                self.report,
            )

    def test_empty_video_is_rejected(self):
        empty_video = self.root / "empty.mp4"
        empty_video.write_bytes(b"")

        with self.assertRaises(ValueError):
            register_wan_output(
                make_project(),
                self.workspace,
                empty_video,
                self.report,
            )

    def test_invalid_performance_report_is_rejected(self):
        self.report.write_text("not valid json", encoding="utf-8")

        with self.assertRaises(ValueError):
            register_wan_output(
                make_project(),
                self.workspace,
                self.video,
                self.report,
            )

    def test_missing_report_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            register_wan_output(
                make_project(),
                self.workspace,
                self.video,
                self.root / "missing-report.json",
            )

    def test_job_can_be_reopened_and_artifacts_verified(self):
        self.register_job()
        reopened = ProductionJob.open(self.workspace)

        self.assertEqual(
            reopened.manifest["stages"]["video_generation"]["status"],
            "succeeded",
        )

        self.assertEqual(
            reopened.manifest["stages"]["character_assets"]["status"],
            "succeeded",
        )

        self.assertEqual(reopened.verify_artifacts(), [])


if __name__ == "__main__":
    unittest.main()

