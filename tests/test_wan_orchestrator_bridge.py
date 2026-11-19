
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
            json.dumps({"status": "completed", "scenes": []}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_registers_real_wan_outputs(self):
        job = register_wan_output(
            make_project(),
            self.workspace,
            self.video,
            self.report,
        )

        manifest = job.manifest

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
            json.loads(copied_report.read_text(encoding="utf-8")),
            {"status": "completed", "scenes": []},
        )

        self.assertEqual(job.verify_artifacts(), [])

    def test_audio_and_post_production_remain_pending(self):
        job = register_wan_output(
            make_project(),
            self.workspace,
            self.video,
            self.report,
        )

        manifest = job.manifest

        self.assertEqual(
            manifest["stages"]["audio_generation"]["status"],
            "pending",
        )

        self.assertEqual(
            manifest["stages"]["post_production"]["status"],
            "pending",
        )

        self.assertEqual(manifest["status"], "running")

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
        register_wan_output(
            make_project(),
            self.workspace,
            self.video,
            self.report,
        )

        reopened = ProductionJob.open(self.workspace)

        self.assertEqual(
            reopened.manifest["stages"]["video_generation"]["status"],
            "succeeded",
        )

        self.assertEqual(reopened.verify_artifacts(), [])


if __name__ == "__main__":
    unittest.main()
