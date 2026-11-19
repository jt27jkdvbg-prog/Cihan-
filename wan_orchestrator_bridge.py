
"""Register real Wan outputs in the shared production orchestrator."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from pipeline_config import FilmProject
from production_orchestrator import ProductionJob


def _copy_nonempty(source: str | Path, destination: Path) -> None:
    source_path = Path(source)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Required Wan output was not found: {source_path}"
        )

    if source_path.stat().st_size <= 0:
        raise ValueError(
            f"Wan output is empty: {source_path}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)

    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(
            f"Output copy failed: {destination}"
        )


def register_wan_output(
    project_data: Mapping[str, Any],
    workspace: str | Path,
    video_path: str | Path,
    report_path: str | Path,
) -> ProductionJob:
    """
    Create a job, register validated project inputs, and record actual
    Wan video/report files as video_generation artifacts.

    Audio generation and post-production deliberately remain pending.
    """
    project = FilmProject.from_dict(dict(project_data))
    job = ProductionJob.create(workspace, project.to_dict())

    artifacts_dir = job.workspace / "artifacts"
    input_dir = artifacts_dir / "inputs"
    wan_dir = artifacts_dir / "wan"

    # Stage 1: validated project input.
    validation_path = input_dir / "project_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(
            {
                "valid": True,
                "project_id": project.project_id,
                "project_title": project.title,
                "scene_count": len(project.scenes),
                "character_count": len(project.characters),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    job.start_stage("project_validation")
    job.complete_stage(
        "project_validation",
        ["artifacts/inputs/project_validation.json"],
    )

    # Stage 2: the canonical project is the script/scene source of truth.
    script_path = input_dir / "script_input.json"
    script_path.write_text(
        json.dumps(
            project.to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    job.start_stage("script")
    job.complete_stage(
        "script",
        ["artifacts/inputs/script_input.json"],
    )

    # Stage 3: record the supplied character definitions.
    # This is a validated character brief, not generated image assets.
    characters_path = input_dir / "character_definitions.json"
    characters_path.write_text(
        json.dumps(
            [
                character.to_dict()
                for character in project.characters
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    job.start_stage("character_assets")
    job.complete_stage(
        "character_assets",
        ["artifacts/inputs/character_definitions.json"],
    )

    # Stage 4: copy and register the actual Wan outputs.
    final_video = wan_dir / "final-film.mp4"
    performance_report = wan_dir / "performance_report.json"

    _copy_nonempty(video_path, final_video)
    _copy_nonempty(report_path, performance_report)

    # A report must be valid JSON before it is accepted.
    try:
        report_data = json.loads(
            performance_report.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Wan performance report is not valid JSON."
        ) from exc

    if not isinstance(report_data, dict):
        raise ValueError(
            "Wan performance report must contain a JSON object."
        )

    job.start_stage("video_generation")
    job.complete_stage(
        "video_generation",
        [
            "artifacts/wan/final-film.mp4",
            "artifacts/wan/performance_report.json",
        ],
    )

    # Do not mark audio or post-production complete.
    return ProductionJob.open(job.workspace)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register Wan output in ProductionJob."
    )
    parser.add_argument("--project-json", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    project_data = json.loads(
        Path(args.project_json).read_text(encoding="utf-8")
    )

    job = register_wan_output(
        project_data=project_data,
        workspace=args.workspace,
        video_path=args.video,
        report_path=args.report,
    )

    manifest = job.manifest

    print("Wan output successfully registered.")
    print("Job ID:", manifest["job_id"])
    print("Job status:", manifest["status"])
    print(
        "Video stage:",
        manifest["stages"]["video_generation"]["status"],
    )
    print(
        "Audio stage:",
        manifest["stages"]["audio_generation"]["status"],
    )
    print(
        "Post-production stage:",
        manifest["stages"]["post_production"]["status"],
    )


if __name__ == "__main__":
    main()
