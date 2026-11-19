
"""Register verified Wan and character assets in the production orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from pipeline_config import FilmProject, ProjectConfigError
from production_orchestrator import ProductionJob
from project_adapter import legacy_to_pipeline_project


class CharacterAssetRegistrationError(ValueError):
    """Raised when character assets cannot be safely registered."""


def _load_project(project_data: Mapping[str, Any]) -> FilmProject:
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
            f"Required output was not found: {source_path}"
        )

    if source_path.stat().st_size <= 0:
        raise ValueError(f"Output is empty: {source_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)

    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Output copy failed: {destination}")

    if source_path.stat().st_size != destination.stat().st_size:
        raise RuntimeError(f"Output size mismatch: {destination}")


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
            file.write("\n")
            file.flush()

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_component(value: str) -> str:
    slug = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "-",
        value.strip(),
    ).strip("-_")[:48] or "character"

    suffix = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:8]

    return f"{slug}-{suffix}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _image_extension(path: Path) -> str:
    with path.open("rb") as file:
        header = file.read(16)

    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"

    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"

    raise CharacterAssetRegistrationError(
        f"Unsupported character image format: {path.name}"
    )


def _register_character_assets(
    project: FilmProject,
    source_dir: str | Path,
    job: ProductionJob,
) -> list[str]:
    root = Path(source_dir).resolve()

    manifest_path = root / "character_assets.json"

    if not manifest_path.is_file():
        raise CharacterAssetRegistrationError(
            "character_assets.json is missing."
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CharacterAssetRegistrationError(
            "Character asset manifest is invalid JSON."
        ) from exc

    if not isinstance(manifest, dict):
        raise CharacterAssetRegistrationError(
            "Character asset manifest must be an object."
        )

    if manifest.get("schema_version") != 1:
        raise CharacterAssetRegistrationError(
            "Unsupported character asset manifest version."
        )

    if manifest.get("project_id") != project.project_id:
        raise CharacterAssetRegistrationError(
            "Character asset project ID does not match."
        )

    if manifest.get("all_assets_validated") is not True:
        raise CharacterAssetRegistrationError(
            "Character assets are not fully validated."
        )

    records = manifest.get("assets")

    if not isinstance(records, list):
        raise CharacterAssetRegistrationError(
            "Character asset records must be a list."
        )

    expected_ids = {character.id for character in project.characters}
    actual_ids = set()

    if len(records) != len(project.characters):
        raise CharacterAssetRegistrationError(
            "Character asset count does not match the project."
        )

    destination_root = (
        job.workspace / "artifacts" / "character_assets"
    )

    registered_records = []
    artifact_paths = []

    for record in records:
        if not isinstance(record, dict):
            raise CharacterAssetRegistrationError(
                "Invalid character asset record."
            )

        character_id = record.get("character_id")

        if not isinstance(character_id, str) or not character_id:
            raise CharacterAssetRegistrationError(
                "Character asset ID is missing."
            )

        if character_id in actual_ids:
            raise CharacterAssetRegistrationError(
                f"Duplicate character asset: {character_id}"
            )

        actual_ids.add(character_id)

        if record.get("validated") is not True:
            raise CharacterAssetRegistrationError(
                f"Asset is not validated: {character_id}"
            )

        relative = record.get("path")

        if not isinstance(relative, str) or not relative:
            raise CharacterAssetRegistrationError(
                f"Asset path is missing: {character_id}"
            )

        relative_path = PurePosixPath(relative)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CharacterAssetRegistrationError(
                f"Unsafe asset path: {relative}"
            )

        source = (root / Path(*relative_path.parts)).resolve()

        try:
            source.relative_to(root)
        except ValueError as exc:
            raise CharacterAssetRegistrationError(
                f"Asset escapes the source directory: {relative}"
            ) from exc

        if not source.is_file():
            raise CharacterAssetRegistrationError(
                f"Character image is missing: {relative}"
            )

        actual_size = source.stat().st_size
        expected_size = record.get("bytes")

        if not isinstance(expected_size, int) or actual_size != expected_size:
            raise CharacterAssetRegistrationError(
                f"Character image size mismatch: {character_id}"
            )

        expected_hash = record.get("sha256")

        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or _sha256(source) != expected_hash
        ):
            raise CharacterAssetRegistrationError(
                f"Character image SHA-256 mismatch: {character_id}"
            )

        extension = _image_extension(source)

        destination = (
            destination_root
            / _safe_component(character_id)
            / f"anchor{extension}"
        )

        _copy_nonempty(source, destination)

        relative_destination = destination.relative_to(
            job.workspace
        ).as_posix()

        copied_record = dict(record)
        copied_record["source_path"] = relative
        copied_record["path"] = relative_destination
        copied_record["sha256"] = _sha256(destination)

        registered_records.append(copied_record)
        artifact_paths.append(relative_destination)

    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)

        raise CharacterAssetRegistrationError(
            f"Character IDs mismatch. Missing={missing}, "
            f"unexpected={unexpected}"
        )

    registered_manifest = {
        **manifest,
        "assets": registered_records,
        "registered_in_production_job": True,
    }

    registered_manifest_path = destination_root / "character_assets.json"
    _write_json(registered_manifest_path, registered_manifest)

    artifact_paths.append(
        registered_manifest_path.relative_to(
            job.workspace
        ).as_posix()
    )

    return artifact_paths


def register_wan_output(
    project_data: Mapping[str, Any],
    workspace: str | Path,
    video_path: str | Path,
    report_path: str | Path,
    character_assets_dir: str | Path | None = None,
) -> ProductionJob:
    """Register validated Wan outputs and optional character reference assets."""

    project = _load_project(project_data)
    job = ProductionJob.create(workspace, project.to_dict())

    artifacts_dir = job.workspace / "artifacts"
    input_dir = artifacts_dir / "inputs"
    wan_dir = artifacts_dir / "wan"

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

    script_path = input_dir / "script_input.json"
    _write_json(script_path, project.to_dict())

    job.start_stage("script")
    job.complete_stage(
        "script",
        ["artifacts/inputs/script_input.json"],
    )

    definitions_path = input_dir / "character_definitions.json"

    _write_json(
        definitions_path,
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

    # Projects without characters do not need a character-image stage.
    if not project.characters:
        not_required_path = (
            input_dir / "character_assets_not_required.json"
        )

        _write_json(
            not_required_path,
            {
                "status": "not_required",
                "reason": "The project contains no characters.",
            },
        )

        job.start_stage("character_assets")
        job.complete_stage(
            "character_assets",
            [
                "artifacts/inputs/character_definitions.json",
                "artifacts/inputs/character_assets_not_required.json",
            ],
        )

    elif character_assets_dir is not None:
        job.start_stage("character_assets")

        character_artifacts = _register_character_assets(
            project,
            character_assets_dir,
            job,
        )

        job.complete_stage(
            "character_assets",
            [
                "artifacts/inputs/character_definitions.json",
                *character_artifacts,
            ],
        )

    # Without a real manifest, character_assets remains pending.
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

    return ProductionJob.open(job.workspace)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register Wan output in ProductionJob."
    )

    parser.add_argument("--project-json", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--character-assets-dir")

    args = parser.parse_args()

    project_data = json.loads(
        Path(args.project_json).read_text(encoding="utf-8")
    )

    job = register_wan_output(
        project_data=project_data,
        workspace=args.workspace,
        video_path=args.video,
        report_path=args.report,
        character_assets_dir=args.character_assets_dir,
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

