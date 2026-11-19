
"""Generate and validate persistent visual reference assets for film characters.

The generator reuses the image providers already implemented in generate.py.
It creates one canonical anchor image per character and writes a manifest
plus an updated project JSON containing the generated reference paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import generate as image_pipeline
from pipeline_config import FilmProject, ProjectConfigError
from project_adapter import legacy_to_pipeline_project


class CharacterAssetError(RuntimeError):
    """Raised when character reference assets cannot be generated safely."""


def _load_project(data: Mapping[str, Any]) -> FilmProject:
    if not isinstance(data, Mapping):
        raise ProjectConfigError("Project JSON must contain an object.")

    candidate = dict(data)

    if "film" in candidate and "production" in candidate:
        return legacy_to_pipeline_project(candidate)

    return FilmProject.from_dict(candidate)


def _safe_component(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_")
    slug = slug[:48] or "character"
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{suffix}"


def _character_prompt(project: FilmProject, character: Any) -> str:
    details = [
        f"Character name: {character.name}.",
        f"Visual identity: {character.visual_description}.",
    ]

    if character.appearance:
        details.append(f"Appearance: {character.appearance}.")

    if character.clothing:
        details.append(f"Clothing: {character.clothing}.")

    if character.age is not None:
        details.append(f"Character age: {character.age}.")

    if character.continuity_prompt:
        details.append(
            f"Continuity requirements: {character.continuity_prompt}."
        )

    if project.character_bible:
        details.append(
            f"Shared character bible: {project.character_bible}."
        )

    if project.visual_style:
        details.append(f"Visual style: {project.visual_style}.")

    details.append(
        "Create one original fictional character reference image. "
        "Show one person only, full body, standing in a neutral relaxed pose, "
        "with the face clearly visible and unobstructed. "
        "Keep the character centered and fully inside the frame. "
        "Use a simple neutral studio background and even professional "
        "lighting. Preserve the described facial features, hairstyle, "
        "body proportions and clothing. Realistic anatomy, natural hands, "
        "clear silhouette, high detail. No extra people, no collage, "
        "no split screen, no labels, no captions, no logos, no watermark."
    )

    return " ".join(details)


def _detect_image_extension(path: Path) -> str:
    try:
        with path.open("rb") as file:
            header = file.read(16)
    except OSError as exc:
        raise CharacterAssetError(
            f"Generated image cannot be read: {path}"
        ) from exc

    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"

    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"

    raise CharacterAssetError(
        f"Generated file is not a supported JPEG, PNG or WebP image: {path}"
    )


def _validate_image(path: Path, minimum_bytes: int = 5000) -> str:
    if not path.is_file():
        raise CharacterAssetError(
            f"Character image was not created: {path}"
        )

    size = path.stat().st_size

    if size < minimum_bytes:
        raise CharacterAssetError(
            f"Character image is too small ({size} bytes): {path}"
        )

    return _detect_image_extension(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(
                value,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def generate_character_assets(
    project_data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    engines: Sequence[str] | None = None,
    minimum_image_bytes: int = 5000,
) -> dict[str, Any]:
    """Generate one validated anchor image for every project character."""
    project = _load_project(project_data)

    if not project.characters:
        raise CharacterAssetError(
            "The project has no characters; character assets cannot be "
            "marked as completed."
        )

    selected_engines = (
        list(engines)
        if engines is not None
        else image_pipeline.available_image_engines()
    )

    if not selected_engines:
        raise CharacterAssetError(
            "No image generation engine is available."
        )

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    asset_records: list[dict[str, Any]] = []
    updated_project = project.to_dict()

    for index, character in enumerate(project.characters, start=1):
        character_dir = root / _safe_component(character.id)
        character_dir.mkdir(parents=True, exist_ok=True)

        temporary_image = character_dir / "anchor.generated"

        try:
            prompt = _character_prompt(project, character)

            engine_used = image_pipeline.generate_image(
                prompt,
                temporary_image,
                selected_engines,
            )

            extension = _validate_image(
                temporary_image,
                minimum_bytes=minimum_image_bytes,
            )

            final_image = character_dir / f"anchor{extension}"
            os.replace(temporary_image, final_image)

            relative_path = final_image.relative_to(root).as_posix()

            asset_records.append({
                "character_id": character.id,
                "character_name": character.name,
                "asset_type": "canonical_character_anchor",
                "path": relative_path,
                "engine": engine_used,
                "bytes": final_image.stat().st_size,
                "sha256": _sha256(final_image),
                "validated": True,
            })

            updated_character = updated_project["characters"][index - 1]
            updated_character["reference_image"] = relative_path
            updated_character["reference_images"] = [relative_path]

        except Exception as exc:
            temporary_image.unlink(missing_ok=True)
            raise CharacterAssetError(
                f"Reference generation failed for character "
                f"'{character.name}' ({character.id}): {exc}"
            ) from exc

    manifest = {
        "schema_version": 1,
        "project_id": project.project_id,
        "project_title": project.title,
        "asset_count": len(asset_records),
        "all_assets_validated": (
            len(asset_records) == len(project.characters)
            and all(item["validated"] for item in asset_records)
        ),
        "assets": asset_records,
    }

    if not manifest["all_assets_validated"]:
        raise CharacterAssetError(
            "Character asset validation did not complete."
        )

    _atomic_write_json(root / "character_assets.json", manifest)
    _atomic_write_json(
        root / "project_with_character_assets.json",
        updated_project,
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate persistent character reference images."
    )
    parser.add_argument("--project-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    project_data = json.loads(
        Path(args.project_json).read_text(encoding="utf-8")
    )

    manifest = generate_character_assets(
        project_data,
        args.output_dir,
    )

    print("Character reference generation completed.")
    print("Project:", manifest["project_title"])
    print("Validated character assets:", manifest["asset_count"])
    print(
        "Manifest:",
        Path(args.output_dir) / "character_assets.json",
    )


if __name__ == "__main__":
    main()
