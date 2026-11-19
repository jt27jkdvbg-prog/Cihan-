
"""Load and verify scene-specific Wan references inside Kaggle."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


class WanReferenceLoadError(ValueError):
    """Raised when Kaggle reference assets cannot be loaded safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _safe_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WanReferenceLoadError("Reference path is missing.")

    if "\\" in value:
        raise WanReferenceLoadError("Reference path must use forward slashes.")

    relative = PurePosixPath(value)

    if relative.is_absolute() or ".." in relative.parts:
        raise WanReferenceLoadError("Unsafe reference path.")

    if not relative.parts or relative.parts[0] != "reference_assets":
        raise WanReferenceLoadError(
            "Reference image must be inside reference_assets."
        )

    path = (root / Path(*relative.parts)).resolve()

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WanReferenceLoadError(
            "Reference path escapes dataset directory."
        ) from exc

    if not path.is_file():
        raise WanReferenceLoadError(
            f"Reference image not found: {value}"
        )

    return path


def _validate_image(path: Path, image_format: Any) -> None:
    if image_format not in {"jpg", "png", "webp"}:
        raise WanReferenceLoadError("Unsupported reference image format.")

    if path.stat().st_size < 5000:
        raise WanReferenceLoadError("Reference image is too small.")

    with path.open("rb") as file:
        header = file.read(16)

    valid = (
        image_format == "jpg" and header.startswith(b"\xff\xd8\xff")
    ) or (
        image_format == "png" and header.startswith(b"\x89PNG\r\n\x1a\n")
    ) or (
        image_format == "webp"
        and header.startswith(b"RIFF")
        and header[8:12] == b"WEBP"
    )

    if not valid:
        raise WanReferenceLoadError("Reference image format mismatch.")


def load_reference_bundle(
    dataset_dir: str | Path,
    expected_project_id: str | None = None,
) -> dict[str, Any]:
    """Load a Kaggle dataset bundle and return verified local image paths."""

    root = Path(dataset_dir).resolve()

    if not root.is_dir():
        raise WanReferenceLoadError("Kaggle dataset directory is missing.")

    manifest_path = root / "reference_bundle.json"

    if not manifest_path.is_file():
        raise WanReferenceLoadError("reference_bundle.json is missing.")

    try:
        bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WanReferenceLoadError(
            "Reference bundle is invalid JSON."
        ) from exc

    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise WanReferenceLoadError("Unsupported reference bundle schema.")

    project_id = bundle.get("project_id")

    if not isinstance(project_id, str) or not project_id:
        raise WanReferenceLoadError("Project ID is missing.")

    if expected_project_id is not None and project_id != expected_project_id:
        raise WanReferenceLoadError("Reference bundle project ID mismatch.")

    assets = bundle.get("assets")
    scenes = bundle.get("scenes")

    if not isinstance(assets, list) or not isinstance(scenes, list):
        raise WanReferenceLoadError("Invalid assets or scenes list.")

    if not scenes:
        raise WanReferenceLoadError("Reference bundle has no scenes.")

    if bundle.get("scene_count") != len(scenes):
        raise WanReferenceLoadError("Scene count mismatch.")

    if bundle.get("unique_reference_count") != len(assets):
        raise WanReferenceLoadError("Unique asset count mismatch.")

    verified_assets: dict[str, dict[str, Any]] = {}

    for asset in assets:
        if not isinstance(asset, dict):
            raise WanReferenceLoadError("Invalid asset record.")

        digest = asset.get("sha256")

        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise WanReferenceLoadError("Invalid asset SHA-256.")

        if digest in verified_assets:
            raise WanReferenceLoadError("Duplicate asset SHA-256.")

        expected_bytes = asset.get("bytes")

        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 5000
        ):
            raise WanReferenceLoadError("Invalid asset byte count.")

        path = _safe_file(root, asset.get("path"))

        if path.stat().st_size != expected_bytes:
            raise WanReferenceLoadError("Reference image byte count mismatch.")

        if _sha256(path) != digest:
            raise WanReferenceLoadError("Reference image SHA-256 mismatch.")

        image_format = asset.get("format")
        _validate_image(path, image_format)

        verified_assets[digest] = {
            **asset,
            "local_path": str(path),
        }

    loaded_scenes = []
    total_scene_references = 0

    for scene in scenes:
        if not isinstance(scene, dict):
            raise WanReferenceLoadError("Invalid scene record.")

        references = scene.get("reference_images")

        if not isinstance(references, list):
            raise WanReferenceLoadError("Invalid scene reference list.")

        loaded_references = []

        for reference in references:
            if not isinstance(reference, dict):
                raise WanReferenceLoadError("Invalid scene reference.")

            digest = reference.get("sha256")
            asset = verified_assets.get(digest)

            if asset is None:
                raise WanReferenceLoadError(
                    "Scene references an unregistered image."
                )

            if reference.get("path") != asset.get("path"):
                raise WanReferenceLoadError("Scene reference path mismatch.")

            if reference.get("format") != asset.get("format"):
                raise WanReferenceLoadError("Scene reference format mismatch.")

            if reference.get("character_id") != asset.get("character_id"):
                raise WanReferenceLoadError("Scene character ID mismatch.")

            loaded_references.append({
                **reference,
                "local_path": asset["local_path"],
            })

            total_scene_references += 1

        loaded_scenes.append({
            **scene,
            "reference_images": loaded_references,
        })

    if bundle.get("scene_reference_count") != total_scene_references:
        raise WanReferenceLoadError("Scene reference count mismatch.")

    return {
        **bundle,
        "assets": list(verified_assets.values()),
        "scenes": loaded_scenes,
        "runtime_validation": {
            "status": "verified",
            "images_verified": len(verified_assets),
            "scene_references_verified": total_scene_references,
            "project_id": project_id,
        },
    }
