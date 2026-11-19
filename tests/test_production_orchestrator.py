
import json
import tempfile
import unittest
from pathlib import Path

from production_orchestrator import (
    HASH_CHUNK_SIZE,
    OrchestrationError,
    ProductionJob,
    STAGES,
    _job_lock,
)


def make_project():
    return {
        "project_id": "orchestrator-test-001",
        "title": "Orchestrator Test",
        "brief": "A test project for production orchestration.",
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


class ProductionOrchestratorTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name) / "job"
        self.project_data = make_project()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_job(self):
        return ProductionJob.create(
            self.workspace,
            self.project_data,
        )

    def complete_stage(self, job, stage):
        job.start_stage(stage)
        job.complete_stage(stage)

    def prepare_video_dependencies(self, job):
        for stage in (
            "project_validation",
            "script",
            "character_assets",
        ):
            self.complete_stage(job, stage)

    def test_job_creation_and_reopening(self):
        job = self.create_job()
        reopened = ProductionJob.open(self.workspace)

        self.assertEqual(
            reopened.manifest["job_id"],
            "orchestrator-test-001",
        )
        self.assertEqual(
            reopened.project.title,
            job.project.title,
        )

    def test_job_manifest_contains_all_stages(self):
        job = self.create_job()
        manifest = job.manifest

        self.assertEqual(set(manifest["stages"]), set(STAGES))

        for stage in STAGES:
            self.assertEqual(
                manifest["stages"][stage]["status"],
                "pending",
            )

    def test_manifest_uses_schema_version_two(self):
        job = self.create_job()

        self.assertEqual(
            job.manifest["orchestrator_schema_version"],
            2,
        )

    def test_existing_job_cannot_be_overwritten(self):
        self.create_job()

        with self.assertRaises(OrchestrationError):
            ProductionJob.create(
                self.workspace,
                self.project_data,
            )

    def test_invalid_project_is_rejected(self):
        invalid = make_project()
        invalid["title"] = ""

        with self.assertRaises(ValueError):
            ProductionJob.create(
                self.workspace,
                invalid,
            )

    def test_stage_cannot_start_before_dependencies(self):
        job = self.create_job()

        with self.assertRaises(OrchestrationError):
            job.start_stage("script")

    def test_unknown_stage_is_rejected(self):
        job = self.create_job()

        with self.assertRaises(OrchestrationError):
            job.start_stage("unknown_stage")

    def test_stage_lifecycle_completes_entire_job(self):
        job = self.create_job()

        for stage in (
            "project_validation",
            "script",
            "character_assets",
            "video_generation",
            "audio_generation",
            "post_production",
            "quality_control",
            "publishing",
        ):
            attempt = job.start_stage(stage)
            self.assertEqual(attempt, 1)
            job.complete_stage(stage)

        self.assertEqual(
            job.manifest["status"],
            "completed",
        )

        for stage in STAGES:
            self.assertEqual(
                job.manifest["stages"][stage]["status"],
                "succeeded",
            )

    def test_completed_stage_cannot_be_started_again(self):
        job = self.create_job()
        self.complete_stage(job, "project_validation")

        with self.assertRaises(OrchestrationError):
            job.start_stage("project_validation")

    def test_failed_stage_can_be_retried(self):
        job = self.create_job()

        job.start_stage("project_validation")
        job.fail_stage(
            "project_validation",
            "Temporary provider failure",
        )

        self.assertEqual(
            job.manifest["stages"]["project_validation"]["status"],
            "failed",
        )

        job.retry_failed_stage("project_validation")

        self.assertEqual(
            job.manifest["stages"]["project_validation"]["status"],
            "pending",
        )

        attempt = job.start_stage("project_validation")
        self.assertEqual(attempt, 2)

    def test_failed_stage_error_is_recorded(self):
        job = self.create_job()

        job.start_stage("project_validation")
        job.fail_stage(
            "project_validation",
            "Test failure",
        )

        self.assertEqual(
            job.manifest["stages"]["project_validation"]["error"],
            "Test failure",
        )

    def test_artifact_is_verified_and_recorded(self):
        job = self.create_job()

        artifact = self.workspace / "project-validation.json"
        artifact.write_text(
            '{"status": "ready"}',
            encoding="utf-8",
        )

        job.start_stage("project_validation")
        job.complete_stage(
            "project_validation",
            ["project-validation.json"],
        )

        recorded = job.manifest["stages"][
            "project_validation"
        ]["artifacts"]

        self.assertEqual(len(recorded), 1)
        self.assertEqual(
            recorded[0]["path"],
            "project-validation.json",
        )
        self.assertEqual(job.verify_artifacts(), [])

    def test_large_artifact_is_verified(self):
        job = self.create_job()

        artifact = self.workspace / "large-artifact.bin"
        artifact.write_bytes(
            b"x" * (HASH_CHUNK_SIZE * 3 + 123)
        )

        job.start_stage("project_validation")
        job.complete_stage(
            "project_validation",
            ["large-artifact.bin"],
        )

        self.assertEqual(job.verify_artifacts(), [])

        record = job.manifest["stages"][
            "project_validation"
        ]["artifacts"][0]

        self.assertEqual(
            record["size_bytes"],
            HASH_CHUNK_SIZE * 3 + 123,
        )

    def test_missing_artifact_prevents_completion(self):
        job = self.create_job()
        job.start_stage("project_validation")

        with self.assertRaises(OrchestrationError):
            job.complete_stage(
                "project_validation",
                ["missing.json"],
            )

        self.assertEqual(
            job.manifest["stages"]["project_validation"]["status"],
            "running",
        )

    def test_artifact_path_traversal_is_rejected(self):
        job = self.create_job()

        outside = Path(self.temp_dir.name) / "outside.txt"
        outside.write_text("outside", encoding="utf-8")

        job.start_stage("project_validation")

        with self.assertRaises(OrchestrationError):
            job.complete_stage(
                "project_validation",
                ["../outside.txt"],
            )

    def test_modified_artifact_fails_checksum_verification(self):
        job = self.create_job()

        artifact = self.workspace / "project-validation.json"
        artifact.write_text(
            '{"status": "ready"}',
            encoding="utf-8",
        )

        job.start_stage("project_validation")
        job.complete_stage(
            "project_validation",
            ["project-validation.json"],
        )

        artifact.write_text(
            '{"status": "other"}',
            encoding="utf-8",
        )

        problems = job.verify_artifacts()

        self.assertEqual(len(problems), 1)
        self.assertIn("checksum mismatch", problems[0])

    def test_open_refuses_modified_artifact(self):
        job = self.create_job()

        artifact = self.workspace / "project-validation.json"
        artifact.write_text(
            '{"status": "ready"}',
            encoding="utf-8",
        )

        job.start_stage("project_validation")
        job.complete_stage(
            "project_validation",
            ["project-validation.json"],
        )

        artifact.write_text(
            '{"status": "other"}',
            encoding="utf-8",
        )

        with self.assertRaises(OrchestrationError):
            ProductionJob.open(self.workspace)

    def test_modified_project_prevents_unsafe_resume(self):
        self.create_job()

        project = json.loads(
            (self.workspace / "project.json").read_text(
                encoding="utf-8"
            )
        )

        project["title"] = "Modified project"

        (self.workspace / "project.json").write_text(
            json.dumps(project),
            encoding="utf-8",
        )

        with self.assertRaises(OrchestrationError):
            ProductionJob.open(self.workspace)

    def test_active_lock_prevents_concurrent_modification(self):
        job = self.create_job()

        with _job_lock(job.lock_path):
            with self.assertRaises(OrchestrationError):
                job.start_stage("project_validation")

    def test_old_lock_file_does_not_block_new_process(self):
        job = self.create_job()

        job.lock_path.write_text(
            "old process metadata",
            encoding="utf-8",
        )

        attempt = job.start_stage("project_validation")
        self.assertEqual(attempt, 1)

    def test_video_requires_character_assets(self):
        job = self.create_job()

        self.complete_stage(job, "project_validation")
        self.complete_stage(job, "script")

        with self.assertRaises(OrchestrationError):
            job.start_stage("video_generation")

        job.complete_stage if False else None

        self.complete_stage(job, "character_assets")

        self.assertEqual(
            job.start_stage("video_generation"),
            1,
        )

    def test_audio_and_video_can_run_in_parallel(self):
        job = self.create_job()
        self.prepare_video_dependencies(job)

        job.start_stage("video_generation")
        job.start_stage("audio_generation")

        self.assertEqual(
            job.manifest["stages"]["video_generation"]["status"],
            "running",
        )
        self.assertEqual(
            job.manifest["stages"]["audio_generation"]["status"],
            "running",
        )

    def test_post_production_requires_both_outputs(self):
        job = self.create_job()
        self.prepare_video_dependencies(job)

        job.start_stage("video_generation")
        job.complete_stage("video_generation")

        with self.assertRaises(OrchestrationError):
            job.start_stage("post_production")

        job.start_stage("audio_generation")
        job.complete_stage("audio_generation")

        self.assertEqual(
            job.start_stage("post_production"),
            1,
        )

    def test_parallel_stage_status_is_preserved(self):
        job = self.create_job()
        self.prepare_video_dependencies(job)

        job.start_stage("video_generation")
        job.start_stage("audio_generation")

        job.fail_stage(
            "audio_generation",
            "Temporary audio provider failure",
        )

        self.assertEqual(job.manifest["status"], "failed")

        job.retry_failed_stage("audio_generation")

        self.assertEqual(job.manifest["status"], "running")
        self.assertEqual(
            job.manifest["stages"]["video_generation"]["status"],
            "running",
        )


if __name__ == "__main__":
    unittest.main()

