
"""Package verified Wan character references for Kaggle transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


class WanPayloadError(ValueError):
    """Raised when a Wan reference payload cannot be prepared."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _safe_source(workspace: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise WanPayloadError("Reference path is missing.")

    if "\\" in relative_path:
        raise WanPayloadError("Reference paths must use forward slashes.")

    relative = PurePosixPath(relative_path)

    if relative.is_absolute() or ".." in relative.parts:
        raise WanPayloadError("Unsafe reference path.")

    source = (workspace / Path(*relative.parts)).resolve()

    try:
        source.relative_to(workspace)
    except ValueError as exc:
        raise WanPayloadError("Reference path escapes workspace.") from exc

    if not source.is_file():
        raise WanPayloadError(f"Reference image not found: {relative_path}")

    return source


def _image_format(path: Path) -> str:
    if path.stat().st_size < 5000:
        raise WanPayloadError(f"Reference image is too small: {path.name}")

    with path.open("rb") as file:
        header = file.read(16)

    if header.startswith(b"\xff\xd8\xff"):
        return "jpg"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"

    raise WanPayloadError(f"Unsupported image format: {path.name}")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)

    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def prepare_reference_payload(
    plan_path: str | Path,
    workspace: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create a self-contained, hash-verified Kaggle reference bundle."""

    root = Path(workspace).resolve()
    plan_file = Path(plan_path).resolve()
    destination = Path(output_dir).resolve()

    if not root.is_dir():
        raise WanPayloadError("Production workspace does not exist.")

    try:
        plan_file.relative_to(root)
    except ValueError as exc:
        raise WanPayloadError("Reference plan must be inside workspace.") from exc

    if not plan_file.is_file():
        raise WanPayloadError("Reference plan file does not exist.")

    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WanPayloadError("Reference plan is invalid JSON.") from exc

    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise WanPayloadError("Unsupported reference plan schema.")

    if plan.get("status") != "ready":
        raise WanPayloadError("Reference plan is not ready.")

    scenes = plan.get("scenes")

    if not isinstance(scenes, list) or not scenes:
        raise WanPayloadError("Reference plan has no scenes.")

    if destination.exists():
        raise WanPayloadError("Output directory already exists.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(
        prefix=destination.name + ".staging.",
        dir=destination.parent,
    ))

    try:
        asset_dir = staging / "reference_assets"
        asset_dir.mkdir(parents=True)

        copied_assets: dict[str, dict[str, Any]] = {}
        packaged_scenes = []

        for scene in scenes:
            if not isinstance(scene, dict):
                raise WanPayloadError("Invalid scene record.")

            references = scene.get("reference_images", [])

            if not isinstance(references, list):
                raise WanPayloadError("Invalid scene reference list.")

            packaged_references = []

            for reference in references:
                if not isinstance(reference, dict):
                    raise WanPayloadError("Invalid reference record.")

                character_id = reference.get("character_id")
                expected_hash = reference.get("sha256")

                if not isinstance(character_id, str) or not character_id:
                    raise WanPayloadError("Invalid character ID.")

                if (
                    not isinstance(expected_hash, str)
                    or len(expected_hash) != 64
                    or any(c not in "0123456789abcdef" for c in expected_hash)
                ):
                    raise WanPayloadError("Invalid reference SHA-256.")

                source = _safe_source(root, reference.get("path"))

                actual_hash = _sha256(source)

                if actual_hash != expected_hash:
                    raise WanPayloadError(
                        f"Source hash mismatch: {character_id}"
                    )

                image_format = _image_format(source)

                if reference.get("format") != image_format:
                    raise WanPayloadError(
                        f"Image format mismatch: {character_id}"
                    )

                asset_name = f"{actual_hash}.{image_format}"
                relative_target = f"reference_assets/{asset_name}"
                target = staging / relative_target

                if actual_hash not in copied_assets:
                    shutil.copyfile(source, target)

                    if _sha256(target) != actual_hash:
                        raise WanPayloadError(
                            f"Copied image verification failed: {character_id}"
                        )

                    copied_assets[actual_hash] = {
                        "character_id": character_id,
                        "character_name": reference.get("character_name", ""),
                        "path": relative_target,
                        "sha256": actual_hash,
                        "format": image_format,
                        "bytes": target.stat().st_size,
                    }

                packaged_references.append({
                    "character_id": character_id,
                    "character_name": reference.get("character_name", ""),
                    "path": relative_target,
                    "sha256": actual_hash,
                    "format": image_format,
                })

            packaged_scenes.append({
                "scene_id": scene.get("scene_id"),
                "order": scene.get("order"),
                "duration_seconds": scene.get("duration_seconds"),
                "character_ids": scene.get("character_ids", []),
                "reference_conditioning_required": bool(packaged_references),
                "reference_images": packaged_references,
                "visual_prompt": scene.get("visual_prompt", ""),
            })

        bundle = {
            "schema_version": 1,
            "project_id": plan.get("project_id"),
            "project_title": plan.get("project_title"),
            "source_plan_sha256": _sha256(plan_file),
            "conditioning_mode": plan.get("conditioning_mode"),
            "engine": plan.get("engine"),
            "model_profile": plan.get("model_profile"),
            "scene_count": len(packaged_scenes),
            "unique_reference_count": len(copied_assets),
            "scene_reference_count": sum(
                len(scene["reference_images"])
                for scene in packaged_scenes
            ),
            "assets": list(copied_assets.values()),
            "scenes": packaged_scenes,
        }

        _write_json(staging / "reference_bundle.json", bundle)

        os.replace(staging, destination)

    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return destination / "reference_bundle.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare verified Wan reference images for Kaggle."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = prepare_reference_payload(
        args.plan,
        args.workspace,
        args.output,
    )

    bundle = json.loads(result.read_text(encoding="utf-8"))

    print("Reference payload ready.")
    print("Unique images:", bundle["unique_reference_count"])
    print("Scene references:", bundle["scene_reference_count"])
    print("Manifest:", result)


if __name__ == "__main__":
    main()
