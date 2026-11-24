"""Build verified per-scene character reference inputs for Wan VACE R2V."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from pipeline_config import FilmProject, ProjectConfigError
from project_adapter import legacy_to_pipeline_project


class WanReferencePlanError(ValueError):
    """Raised when a Wan character reference plan is invalid."""


def _load_project(data: Mapping[str, Any]) -> FilmProject:
    if not isinstance(data, Mapping):
        raise ProjectConfigError(
            "Project JSON must contain an object."
        )

    candidate = dict(data)

    if "film" in candidate and "production" in candidate:
        return legacy_to_pipeline_project(candidate)

    return FilmProject.from_dict(candidate)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_image(path: Path) -> str:
    if not path.is_file():
        raise WanReferencePlanError(
            f"Reference image is missing: {path.name}"
        )

    if path.stat().st_size < 5000:
        raise WanReferencePlanError(
            f"Reference image is too small: {path.name}"
        )

    with path.open("rb") as file:
        header = file.read(16)

    if header.startswith(b"\xff\xd8\xff"):
        return "jpg"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    if (
        header.startswith(b"RIFF")
        and header[8:12] == b"WEBP"
    ):
        return "webp"

    raise WanReferencePlanError(
        f"Unsupported reference image: {path.name}"
    )


def _safe_workspace_path(
    workspace: Path,
    relative_path: Any,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise WanReferencePlanError(
            "Reference image path is missing."
        )

    if "\\" in relative_path:
        raise WanReferencePlanError(
            "Reference paths must use forward slashes."
        )

    relative = PurePosixPath(relative_path)

    if relative.is_absolute() or ".." in relative.parts:
        raise WanReferencePlanError(
            f"Unsafe reference path: {relative_path}"
        )

    resolved = (
        workspace / Path(*relative.parts)
    ).resolve()

    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise WanReferencePlanError(
            f"Reference path escapes workspace: {relative_path}"
        ) from exc

    return resolved


def _load_registered_assets(
    project: FilmProject,
    workspace: Path,
    manifest_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    if not project.characters:
        return {}

    if manifest_path is None:
        path = (
            workspace
            / "artifacts"
            / "character_assets"
            / "character_assets.json"
        )
    else:
        supplied = Path(manifest_path)

        path = (
            supplied.resolve()
            if supplied.is_absolute()
            else (workspace / supplied).resolve()
        )

        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise WanReferencePlanError(
                "Manifest must be inside the production workspace."
            ) from exc

    if not path.is_file():
        raise WanReferencePlanError(
            "Registered character_assets.json was not found."
        )

    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WanReferencePlanError(
            "Character asset manifest is invalid JSON."
        ) from exc

    if not isinstance(manifest, dict):
        raise WanReferencePlanError(
            "Character asset manifest must be an object."
        )

    if manifest.get("schema_version") != 1:
        raise WanReferencePlanError(
            "Unsupported character asset manifest version."
        )

    if manifest.get("project_id") != project.project_id:
        raise WanReferencePlanError(
            "Character asset project ID does not match."
        )

    if manifest.get("registered_in_production_job") is not True:
        raise WanReferencePlanError(
            "Character assets are not registered in ProductionJob."
        )

    if manifest.get("all_assets_validated") is not True:
        raise WanReferencePlanError(
            "Character assets are not fully validated."
        )

    records = manifest.get("assets")

    if not isinstance(records, list):
        raise WanReferencePlanError(
            "Character asset records must be a list."
        )

    if manifest.get("asset_count") != len(project.characters):
        raise WanReferencePlanError(
            "Character asset count does not match the project."
        )

    if len(records) != len(project.characters):
        raise WanReferencePlanError(
            "Character reference record count is incorrect."
        )

    expected = {
        character.id: character
        for character in project.characters
    }

    assets: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            raise WanReferencePlanError(
                "Invalid character reference record."
            )

        character_id = record.get("character_id")

        if (
            not isinstance(character_id, str)
            or character_id not in expected
        ):
            raise WanReferencePlanError(
                f"Unknown character reference: {character_id}"
            )

        if character_id in assets:
            raise WanReferencePlanError(
                f"Duplicate character reference: {character_id}"
            )

        if record.get("validated") is not True:
            raise WanReferencePlanError(
                f"Character reference is not validated: {character_id}"
            )

        character = expected[character_id]

        if record.get("character_name") != character.name:
            raise WanReferencePlanError(
                f"Character name mismatch: {character_id}"
            )

        image_path = _safe_workspace_path(
            workspace,
            record.get("path"),
        )

        image_format = _validate_image(image_path)

        expected_bytes = record.get("bytes")

        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes != image_path.stat().st_size
        ):
            raise WanReferencePlanError(
                f"Character image size mismatch: {character_id}"
            )

        expected_hash = record.get("sha256")

        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or _sha256(image_path) != expected_hash
        ):
            raise WanReferencePlanError(
                f"Character image SHA-256 mismatch: {character_id}"
            )

        assets[character_id] = {
            "character_id": character_id,
            "character_name": character.name,
            "path": image_path.relative_to(
                workspace
            ).as_posix(),
            "sha256": expected_hash,
            "format": image_format,
        }

    missing = set(expected) - set(assets)

    if missing:
        raise WanReferencePlanError(
            "Missing character references: "
            + ", ".join(sorted(missing))
        )

    return assets


def build_vace_reference_plan(
    project_data: Mapping[str, Any],
    workspace: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a verified reference-image list for every project scene."""

    project = _load_project(project_data)
    root = Path(workspace).resolve()

    if not root.is_dir():
        raise WanReferencePlanError(
            "Production workspace does not exist."
        )

    assets = _load_registered_assets(
        project,
        root,
        manifest_path,
    )

    known_character_ids = {
        character.id
        for character in project.characters
    }

    scenes = []
    total_scene_references = 0

    for scene in sorted(
        project.scenes,
        key=lambda item: (item.order, item.id),
    ):
        character_ids = list(scene.character_ids)

        if len(character_ids) != len(set(character_ids)):
            raise WanReferencePlanError(
                f"Duplicate character ID in scene: {scene.id}"
            )

        unknown = set(character_ids) - known_character_ids

        if unknown:
            raise WanReferencePlanError(
                f"Scene {scene.id} references unknown characters: "
                + ", ".join(sorted(unknown))
            )

        references = [
            dict(assets[character_id])
            for character_id in character_ids
            if character_id in assets
        ]

        total_scene_references += len(references)

        scenes.append({
            "scene_id": scene.id,
            "order": scene.order,
            "duration_seconds": scene.duration_seconds,
            "character_ids": character_ids,
            "reference_images": references,
            "source_ref_images": [
                item["path"]
                for item in references
            ],
            "reference_conditioning_required": bool(references),
            "visual_prompt": scene.visual_prompt,
        })

    has_references = total_scene_references > 0

    return {
        "schema_version": 1,
        "project_id": project.project_id,
        "project_title": project.title,
        "status": "ready",
        "engine": "wan-vace-r2v" if has_references else "wan-t2v",
        "model_profile": (
            "Wan2.1-VACE-1.3B"
            if has_references
            else "Wan2.1-T2V-1.3B"
        ),
        "conditioning_mode": (
            "reference_to_video"
            if has_references
            else "text_to_video"
        ),
        "resolution": "832*480",
        "frame_num": 81,
        "character_count": len(project.characters),
        "scene_count": len(scenes),
        "unique_reference_count": len(assets),
        "scene_reference_count": total_scene_references,
        "scenes": scenes,
    }


def write_vace_reference_plan(
    project_data: Mapping[str, Any],
    workspace: str | Path,
    output_path: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Build and atomically save the verified Wan reference plan."""

    root = Path(workspace).resolve()

    plan = build_vace_reference_plan(
        project_data,
        root,
        manifest_path=manifest_path,
    )

    if output_path is None:
        destination = (
            root
            / "artifacts"
            / "wan"
            / "character_reference_plan.json"
        )
    else:
        supplied = Path(output_path)

        destination = (
            supplied.resolve()
            if supplied.is_absolute()
            else (root / supplied).resolve()
        )

        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise WanReferencePlanError(
                "Plan output must be inside the production workspace."
            ) from exc

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)

            json.dump(
                plan,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            destination,
        )

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build verified per-scene Wan VACE references."
    )

    parser.add_argument(
        "--project-json",
        required=True,
    )

    parser.add_argument(
        "--workspace",
        required=True,
    )

    parser.add_argument(
        "--output",
    )

    parser.add_argument(
        "--manifest",
    )

    args = parser.parse_args()

    project_data = json.loads(
        Path(args.project_json).read_text(
            encoding="utf-8"
        )
    )

    output = write_vace_reference_plan(
        project_data,
        args.workspace,
        args.output,
        manifest_path=args.manifest,
    )

    plan = json.loads(
        output.read_text(encoding="utf-8")
    )

    print("Wan reference plan created.")
    print("Project:", plan["project_title"])
    print("Engine:", plan["engine"])
    print("Scenes:", plan["scene_count"])
    print("Scene references:", plan["scene_reference_count"])
    print("Output:", output)


if __name__ == "__main__":
    main()
