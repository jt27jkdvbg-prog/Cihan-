
"""Prepare and publish verified character references for the Wan Kaggle worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from character_asset_generator import generate_character_assets
from pipeline_config import FilmProject
from project_adapter import legacy_to_pipeline_project
from production_orchestrator import ProductionJob
from wan_kaggle_payload import prepare_reference_payload
from wan_kaggle_publisher import publish_reference_dataset
from wan_orchestrator_bridge import _register_character_assets
from wan_project_input import normalize_wan_project
from wan_reference_plan import write_vace_reference_plan


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_and_publish(
    project_json: str | Path,
    workspace: str | Path,
    payload_output: str | Path,
    dataset_output: str | Path,
    *,
    kaggle_owner: str,
    dataset_slug: str,
    kernel_metadata: str | Path,
) -> dict[str, Any]:
    project_file = Path(project_json).resolve()
    workspace_path = Path(workspace).resolve()
    payload_path = Path(payload_output).resolve()
    dataset_path = Path(dataset_output).resolve()
    metadata_path = Path(kernel_metadata).resolve()

    for path, label in (
        (project_file, "Project JSON"),
        (metadata_path, "Kaggle kernel metadata"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    for path, label in (
        (workspace_path, "Production workspace"),
        (payload_path, "Payload output"),
        (dataset_path, "Dataset output"),
    ):
        if path.exists():
            raise FileExistsError(
                f"{label} already exists; refusing to overwrite: {path}"
            )

    owner = kaggle_owner.strip().lower()
    slug = dataset_slug.strip().lower()

    if not owner or not slug:
        raise ValueError("Kaggle owner and dataset slug are required.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(metadata, dict) or metadata.get("is_private") is not True:
        raise ValueError("Kaggle kernel metadata must remain private.")

    if not isinstance(metadata.get("dataset_sources"), list):
        raise ValueError("Kaggle dataset_sources must be a list.")

    raw_project = json.loads(project_file.read_text(encoding="utf-8"))
    legacy_project = normalize_wan_project(raw_project)
    project = legacy_to_pipeline_project(legacy_project)

    # When a scene has no dialogue metadata identifying its cast,
    # include the project's characters as visual references for that scene.
    all_character_ids = [character.id for character in project.characters]

    for scene in project.scenes:
        if not scene.character_ids:
            scene.character_ids = list(all_character_ids)

    project_data = project.to_dict()

    workspace_path.mkdir(parents=True, exist_ok=False)
    job = ProductionJob.create(workspace_path, project_data)

    artifacts = workspace_path / "artifacts"
    inputs = artifacts / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    # Record the validation and script stages before generating assets.
    job.start_stage("project_validation")

    validation_path = inputs / "project_validation.json"
    _write_json(validation_path, {
        "valid": True,
        "project_id": project.project_id,
        "project_title": project.title,
        "scene_count": len(project.scenes),
        "character_count": len(project.characters),
        "source": "wan_kaggle_workflow_prepare",
    })

    job.complete_stage(
        "project_validation",
        ["artifacts/inputs/project_validation.json"],
    )

    job.start_stage("script")

    script_path = inputs / "script_input.json"
    _write_json(script_path, project_data)

    job.complete_stage(
        "script",
        ["artifacts/inputs/script_input.json"],
    )

    definitions_path = inputs / "character_definitions.json"
    _write_json(definitions_path, [
        {
            "id": character.id,
            "name": character.name,
            "visual_description": character.visual_description,
            "appearance": character.appearance,
            "personality": character.personality,
        }
        for character in project.characters
    ])

    job.start_stage("character_assets")

    source_assets = workspace_path / "generated_character_assets"

    try:
        manifest = generate_character_assets(
            project_data,
            source_assets,
            engines=["cloudflare"],
        )

        registered_artifacts = _register_character_assets(
            job.project,
            source_assets,
            job,
        )

        job.complete_stage(
            "character_assets",
            [
                "artifacts/inputs/character_definitions.json",
                *registered_artifacts,
            ],
        )

    except Exception as exc:
        try:
            job.fail_stage("character_assets", exc)
        except Exception:
            pass
        raise

    plan_path = write_vace_reference_plan(
        job.project.to_dict(),
        workspace_path,
    )

    bundle_path = prepare_reference_payload(
        plan_path,
        workspace_path,
        payload_path,
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    publication = publish_reference_dataset(
        payload_dir=payload_path,
        output_dir=dataset_path,
        owner=owner,
        slug=slug,
        kernel_metadata_path=metadata_path,
        title=f"Cihan Film {slug[-20:]}",
    )

    result = {
        "status": "published",
        "project_id": project.project_id,
        "project_title": project.title,
        "character_count": len(project.characters),
        "generated_reference_count": manifest["asset_count"],
        "scene_count": bundle["scene_count"],
        "scene_reference_count": bundle["scene_reference_count"],
        "dataset_id": publication["dataset_id"],
        "dataset_privacy": publication["privacy"],
        "kernel_metadata_updated": publication["kernel_metadata_updated"],
        "reference_plan": str(plan_path),
        "reference_bundle": str(bundle_path),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, validate and publish private Wan references."
    )
    parser.add_argument("--project-json", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--payload-output", required=True)
    parser.add_argument("--dataset-output", required=True)
    parser.add_argument("--kaggle-owner", required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--kernel-metadata", required=True)

    args = parser.parse_args()

    prepare_and_publish(
        args.project_json,
        args.workspace,
        args.payload_output,
        args.dataset_output,
        kaggle_owner=args.kaggle_owner,
        dataset_slug=args.dataset_slug,
        kernel_metadata=args.kernel_metadata,
    )


if __name__ == "__main__":
    main()
