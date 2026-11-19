
import json
import tempfile
import unittest
from pathlib import Path

from production_orchestrator import ProductionJob
from wan_orchestrator_bridge import register_wan_output


def make_project():
    return {
        "project_id": "wan-bridge-test-001",
        "title": "Wan Bridge Test",
        "brief": "Testing the Wan integration bridge.",
        "target_duration_seconds": 10,
        "visual_style": "cinematic realistic",
        "profile": "landscape",
        "language": "tr",
        "characters": [],
        "scenes": [
            {
                "id": "scene_1",
                "order": 1,
                "duration_seconds": 10,
                "visual_prompt": "A cinematic city at sunset.",
                "character_ids": [],
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

    def register_job(self):
        return register_wan_output(
            make_project(),
            self.workspace,
            self.video,
            self.report,
        )

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

        artifacts = manifest["stages"][
            "video_generation"
        ]["artifacts"]

        self.assertEqual(len(artifacts), 2)

        copied_video = (
            self.workspace
            / "artifacts"
            / "wan"
            / "final-film.mp4"
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
            json.loads(
                copied_report.read_text(encoding="utf-8")
            ),
            {
                "status": "completed",
                "scenes": [],
            },
        )

        self.assertEqual(job.verify_artifacts(), [])

    def test_character_assets_remain_pending(self):
        job = self.register_job()
        manifest = job.manifest

        self.assertEqual(
            manifest["stages"]["character_assets"]["status"],
            "pending",
        )

    def test_character_definitions_are_saved_as_input_only(self):
        job = self.register_job()

        definitions_path = (
            job.workspace
            / "artifacts"
            / "inputs"
            / "character_definitions.json"
        )

        self.assertTrue(definitions_path.is_file())

        definitions = json.loads(
            definitions_path.read_text(encoding="utf-8")
        )

        self.assertEqual(definitions, [])

        self.assertEqual(
            job.manifest["stages"]["character_assets"]["status"],
            "pending",
        )

    def test_audio_and_post_production_remain_pending(self):
        job = self.register_job()
        manifest = job.manifest

        self.assertEqual(
            manifest["stages"]["audio_generation"]["status"],
            "pending",
        )

        self.assertEqual(
            manifest["stages"]["post_production"]["status"],
            "pending",
        )

        self.assertEqual(
            manifest["stages"]["quality_control"]["status"],
            "pending",
        )

        self.assertEqual(
            manifest["stages"]["publishing"]["status"],
            "pending",
        )

    def test_job_status_is_queued_when_no_stage_is_running(self):
        job = self.register_job()

        self.assertEqual(
            job.manifest["status"],
            "queued",
        )

    def test_video_can_complete_without_character_assets(self):
        job = self.register_job()
        manifest = job.manifest

        self.assertEqual(
            manifest["stages"]["video_generation"]["status"],
            "succeeded",
        )

        self.assertEqual(
            manifest["stages"]["character_assets"]["status"],
            "pending",
        )

    def test_post_production_cannot_start_before_audio_and_assets(self):
        job = self.register_job()

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
        self.report.write_text(
            "not valid json",
            encoding="utf-8",
        )

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
            "pending",
        )

        self.assertEqual(
            reopened.manifest["status"],
            "queued",
        )

        self.assertEqual(reopened.verify_artifacts(), [])


if __name__ == "__main__":
    unittest.main()

