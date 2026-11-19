
"""Build a private Kaggle dataset folder from a verified Wan payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


class WanDatasetError(ValueError):
    """Raised when a Kaggle dataset bundle cannot be prepared."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _safe_relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise WanDatasetError("A relative asset path is required.")

    if "\\" in value:
        raise WanDatasetError("Asset paths must use forward slashes.")

    relative = PurePosixPath(value)

    if relative.is_absolute() or ".." in relative.parts:
        raise WanDatasetError("Unsafe asset path.")

    if not relative.parts or "." in relative.parts:
        raise WanDatasetError("Invalid asset path.")

    return relative


def _resolve_file(root: Path, value: Any) -> Path:
    relative = _safe_relative(value)
    resolved = (root / Path(*relative.parts)).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WanDatasetError("Asset path escapes payload directory.") from exc

    if not resolved.is_file():
        raise WanDatasetError(f"Payload file is missing: {value}")

    return resolved


def _validate_image(path: Path, image_format: Any) -> None:
    if image_format not in {"jpg", "png", "webp"}:
        raise WanDatasetError(f"Unsupported image format: {image_format}")

    if path.stat().st_size < 5000:
        raise WanDatasetError(f"Image is too small: {path.name}")

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
        raise WanDatasetError(f"Image format mismatch: {path.name}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _validate_identity(owner: str, slug: str) -> None:
    pattern = re.compile(r"^[a-z0-9_-]{3,50}$")

    if not isinstance(owner, str) or not pattern.fullmatch(owner):
        raise WanDatasetError("Invalid Kaggle owner name.")

    if not isinstance(slug, str) or not pattern.fullmatch(slug):
        raise WanDatasetError("Invalid Kaggle dataset slug.")


def prepare_kaggle_dataset(
    payload_dir: str | Path,
    output_dir: str | Path,
    owner: str,
    slug: str,
    title: str | None = None,
) -> Path:
    """Create a private Kaggle upload folder from a verified payload."""

    _validate_identity(owner, slug)

    source_root = Path(payload_dir).resolve()
    destination = Path(output_dir).resolve()

    if not source_root.is_dir():
        raise WanDatasetError("Payload directory does not exist.")

    if destination.exists():
        raise WanDatasetError("Output directory already exists.")

    manifest_path = source_root / "reference_bundle.json"

    if not manifest_path.is_file():
        raise WanDatasetError("reference_bundle.json is missing.")

    try:
        bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WanDatasetError("Reference bundle is invalid JSON.") from exc

    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise WanDatasetError("Unsupported reference bundle schema.")

    assets = bundle.get("assets")
    scenes = bundle.get("scenes")

    if not isinstance(assets, list) or not isinstance(scenes, list):
        raise WanDatasetError("Bundle assets or scenes are invalid.")

    if bundle.get("scene_count") != len(scenes):
        raise WanDatasetError("Scene count does not match the bundle.")

    if bundle.get("unique_reference_count") != len(assets):
        raise WanDatasetError("Asset count does not match the bundle.")

    if not scenes:
        raise WanDatasetError("The reference bundle contains no scenes.")

    validated_assets: dict[str, tuple[Path, dict[str, Any]]] = {}

    for asset in assets:
        if not isinstance(asset, dict):
            raise WanDatasetError("Invalid asset record.")

        digest = asset.get("sha256")
        image_format = asset.get("format")
        expected_bytes = asset.get("bytes")
        relative = asset.get("path")

        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise WanDatasetError("Invalid asset SHA-256.")

        if digest in validated_assets:
            raise WanDatasetError("Duplicate asset hash in bundle.")

        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 5000
        ):
            raise WanDatasetError("Invalid asset byte count.")

        safe_path = _safe_relative(relative)

        if safe_path.parts[0] != "reference_assets":
            raise WanDatasetError("Assets must be inside reference_assets.")

        source = _resolve_file(source_root, relative)

        if source.stat().st_size != expected_bytes:
            raise WanDatasetError("Asset byte count mismatch.")

        if _sha256(source) != digest:
            raise WanDatasetError("Asset SHA-256 mismatch.")

        _validate_image(source, image_format)

        validated_assets[digest] = (source, asset)

    scene_reference_count = 0

    for scene in scenes:
        if not isinstance(scene, dict):
            raise WanDatasetError("Invalid scene record.")

        references = scene.get("reference_images")

        if not isinstance(references, list):
            raise WanDatasetError("Invalid scene reference list.")

        for reference in references:
            if not isinstance(reference, dict):
                raise WanDatasetError("Invalid scene reference.")

            digest = reference.get("sha256")
            asset_entry = validated_assets.get(digest)

            if asset_entry is None:
                raise WanDatasetError(
                    "Scene references an asset absent from the bundle."
                )

            asset_record = asset_entry[1]

            if reference.get("path") != asset_record.get("path"):
                raise WanDatasetError("Scene asset path mismatch.")

            if reference.get("format") != asset_record.get("format"):
                raise WanDatasetError("Scene asset format mismatch.")

            scene_reference_count += 1

    if bundle.get("scene_reference_count") != scene_reference_count:
        raise WanDatasetError("Scene reference count mismatch.")

    dataset_title = title or f"Cihan Wan {slug}"

    if not isinstance(dataset_title, str):
        raise WanDatasetError("Dataset title must be text.")

    dataset_title = dataset_title.strip()

    if not 6 <= len(dataset_title) <= 50:
        raise WanDatasetError("Dataset title must be 6 to 50 characters.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(
        prefix=destination.name + ".staging.",
        dir=destination.parent,
    ))

    try:
        shutil.copy2(manifest_path, staging / "reference_bundle.json")

        for source, asset in validated_assets.values():
            relative = _safe_relative(asset["path"])
            target = staging / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

            if _sha256(target) != asset["sha256"]:
                raise WanDatasetError("Copied asset SHA-256 mismatch.")

        metadata = {
            "title": dataset_title,
            "id": f"{owner}/{slug}",
            "licenses": [{"name": "other"}],
            "description": (
                "Private per-run character reference images and "
                "verified scene manifest for the Cihan Wan film pipeline. "
                "Reference assets are intended for this production only."
            ),
        }

        _write_json(staging / "dataset-metadata.json", metadata)

        os.replace(staging, destination)

    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return destination / "dataset-metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a private Kaggle dataset from a Wan payload."
    )
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title")

    args = parser.parse_args()

    metadata_path = prepare_kaggle_dataset(
        payload_dir=args.payload,
        output_dir=args.output,
        owner=args.owner,
        slug=args.slug,
        title=args.title,
    )

    print("Kaggle dataset folder ready.")
    print("Metadata:", metadata_path)
    print("Dataset ID:", f"{args.owner}/{args.slug}")
    print("Privacy: private by default; do not use --public.")


if __name__ == "__main__":
    main()
