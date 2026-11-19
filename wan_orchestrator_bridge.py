
"""Register real Wan outputs in the shared production orchestrator."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from pipeline_config import FilmProject, ProjectConfigError
from production_orchestrator import ProductionJob
from project_adapter import legacy_to_pipeline_project


def _load_project(project_data: Mapping[str, Any]) -> FilmProject:
    """Accept both modern pipeline projects and legacy film projects."""
    if not isinstance(project_data, Mapping):
        raise ProjectConfigError("Project JSON must contain an object.")

    candidate = dict(project_data)

    if "film" in candidate and "production" in candidate:
        return legacy_to_pipeline_project(candidate)

    return FilmProject.from_dict(candidate)


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

    if source_path.stat().st_size != destination.stat().st_size:
        raise RuntimeError(
            f"Output copy size mismatch: {destination}"
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                value,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.flush()

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def register_wan_output(
    project_data: Mapping[str, Any],
    workspace: str | Path,
    video_path: str | Path,
    report_path: str | Path,
) -> ProductionJob:
    """
    Create a production job and register actual Wan video outputs.

    Character assets are deliberately NOT marked as completed here.
    The Wan text-to-video workflow does not generate persistent character
    reference images, so that stage must remain pending until a real
    character-asset generator is integrated.
    """
    project = _load_project(project_data)
    job = ProductionJob.create(workspace, project.to_dict())

    artifacts_dir = job.workspace / "artifacts"
    input_dir = artifacts_dir / "inputs"
    wan_dir = artifacts_dir / "wan"

    # Stage 1: validate and record the canonical project.
    validation_path = input_dir / "project_validation.json"

    _write_json(
        validation_path,
        {
            "valid": True,
            "project_id": project.project_id,
            "project_title": project.title,
            "scene_count": len(project.scenes),
            "character_count": len(project.characters),
            "schema": "pipeline_config.FilmProject",
        },
    )

    job.start_stage("project_validation")
    job.complete_stage(
        "project_validation",
        ["artifacts/inputs/project_validation.json"],
    )

    # Stage 2: preserve the canonical script and scene source.
    script_path = input_dir / "script_input.json"

    _write_json(script_path, project.to_dict())

    job.start_stage("script")
    job.complete_stage(
        "script",
        ["artifacts/inputs/script_input.json"],
    )

    # Character definitions are retained as input data only.
    # They are NOT proof that visual reference assets were generated.
    character_definitions_path = (
        input_dir / "character_definitions.json"
    )

    _write_json(
        character_definitions_path,
        [
            {
                "id": character.id,
                "name": character.name,
                "visual_description": character.visual_description,
                "appearance": character.appearance,
                "personality": character.personality,
            }
            for character in project.characters
        ],
    )

    # Stage 3: register actual Wan outputs.
    final_video = wan_dir / "final-film.mp4"
    performance_report = wan_dir / "performance_report.json"

    _copy_nonempty(video_path, final_video)
    _copy_nonempty(report_path, performance_report)

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

    # Audio, character assets, post-production, QC and publishing
    # remain pending until their real generators are integrated.
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

    for stage in (
        "character_assets",
        "video_generation",
        "audio_generation",
        "post_production",
        "quality_control",
        "publishing",
    ):
        print(
            f"{stage}:",
            manifest["stages"][stage]["status"],
        )


if __name__ == "__main__":
    main()

