
"""Generate and validate persistent visual reference assets for film characters.

The generator reuses the image providers already implemented in generate.py.
It creates one canonical anchor image per character and writes a manifest
plus an updated project JSON containing the generated reference paths.

Free-first policy:
- Never enables or silently selects paid providers.
- Retries only transient provider/network failures.
- Never retries or attempts to bypass content-safety rejections.
- Never marks incomplete or invalid images as successful assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import generate as image_pipeline
from pipeline_config import FilmProject, ProjectConfigError
from project_adapter import legacy_to_pipeline_project


class CharacterAssetError(RuntimeError):
    """Raised when character reference assets cannot be generated safely."""


MAX_TRANSIENT_RETRIES = 1
RETRY_WAIT_SECONDS = 5


def _load_project(data: Mapping[str, Any]) -> FilmProject:
    if not isinstance(data, Mapping):
        raise ProjectConfigError("Project JSON must contain an object.")

    candidate = dict(data)

    if "film" in candidate and "production" in candidate:
        return legacy_to_pipeline_project(candidate)

    return FilmProject.from_dict(candidate)


def _safe_component(value: str) -> str:
    slug = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "-",
        value.strip(),
    ).strip("-_")

    slug = slug[:48] or "character"
    suffix = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:8]

    return f"{slug}-{suffix}"


def _character_prompt(project: FilmProject, character: Any) -> str:
    """Build a visual-only, neutral character reference prompt.

    Story-wide continuity text and character-bible prose are deliberately
    excluded. They are not necessary to describe a reference image and may
    contain unrelated terms that cause false-positive moderation blocks.
    Character continuity is maintained by reusing the validated anchor.
    """
    details = [
        f"Character name: {character.name}.",
        f"Visual identity: {character.visual_description}.",
    ]

    if character.appearance:
        details.append(
            f"Appearance: {character.appearance}."
        )

    if character.clothing:
        details.append(
            f"Clothing: {character.clothing}."
        )

    if project.visual_style:
        details.append(
            f"Visual style: {project.visual_style}."
        )

    details.append(
        "Create a neutral, non-sexualized reference illustration "
        "of one fictional character. The character is fully clothed "
        "in the described everyday outfit. Show the complete figure "
        "standing in a relaxed neutral pose, with the face clearly "
        "visible. Center the subject and keep the entire figure "
        "inside the frame. Use a plain neutral studio background "
        "and soft even lighting. Preserve only the supplied visible "
        "facial features, hairstyle and clothing. Natural anatomy "
        "and hands, clean silhouette, high-quality character design. "
        "No suggestive pose, no nudity, no extra people, no collage, "
        "no split screen, no text, no labels, no logos, no watermark."
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
        "Generated file is not a supported JPEG, PNG or WebP image: "
        f"{path}"
    )


def _validate_image(
    path: Path,
    minimum_bytes: int = 5000,
) -> str:
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
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
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


def _is_transient_provider_error(error: Exception) -> bool:
    """Return True only for errors that may succeed on a later attempt.

    Content moderation, authentication, invalid requests and other HTTP 4xx
    errors are deliberately not retried.
    """
    message = str(error).lower()

    permanent_markers = (
        "8007",
        "nsfw",
        "moderation",
        "content safety",
        "safety filter",
        "content filter",
        "http 400",
        "http 401",
        "http 403",
        "http 404",
        "http 422",
    )

    if any(marker in message for marker in permanent_markers):
        return False

    transient_markers = (
        "http 408",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "timed out",
        "timeout",
        "connection error",
        "connection reset",
        "temporarily unavailable",
        "temporary failure",
        "rate limit",
        "server error",
    )

    return any(
        marker in message
        for marker in transient_markers
    )


def _generate_image_safely(
    prompt: str,
    output_path: Path,
    engines: Sequence[str],
) -> tuple[str, int]:
    """Generate an image with one retry for transient provider failures."""
    attempts = MAX_TRANSIENT_RETRIES + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        output_path.unlink(missing_ok=True)

        try:
            engine_used = image_pipeline.generate_image(
                prompt,
                output_path,
                list(engines),
            )

            if not output_path.is_file():
                raise CharacterAssetError(
                    "Image provider returned without creating a file."
                )

            return engine_used, attempt

        except Exception as exc:
            last_error = exc
            output_path.unlink(missing_ok=True)

            if (
                attempt >= attempts
                or not _is_transient_provider_error(exc)
            ):
                break

            print(
                "Temporary image provider failure. "
                f"Retrying once in {RETRY_WAIT_SECONDS} seconds: {exc}",
                flush=True,
            )

            time.sleep(RETRY_WAIT_SECONDS)

    raise CharacterAssetError(
        "Image generation failed after "
        f"{attempt} attempt(s): {last_error}"
    ) from last_error


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

    allowed_engines = {"cloudflare", "fal"}
    unknown_engines = set(selected_engines) - allowed_engines

    if unknown_engines:
        raise CharacterAssetError(
            "Unsupported image engine(s): "
            + ", ".join(sorted(unknown_engines))
        )

    # Do not silently add FAL or any other paid provider.
    # The caller must explicitly configure the provider it wants to use.
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    asset_records: list[dict[str, Any]] = []
    updated_project = project.to_dict()

    for index, character in enumerate(
        project.characters,
        start=1,
    ):
        character_dir = root / _safe_component(character.id)
        character_dir.mkdir(parents=True, exist_ok=True)

        temporary_image = character_dir / "anchor.generated"

        try:
            prompt = _character_prompt(
                project,
                character,
            )

            engine_used, attempts_used = _generate_image_safely(
                prompt,
                temporary_image,
                selected_engines,
            )

            extension = _validate_image(
                temporary_image,
                minimum_bytes=minimum_image_bytes,
            )

            final_image = character_dir / f"anchor{extension}"

            os.replace(
                temporary_image,
                final_image,
            )

            relative_path = final_image.relative_to(
                root
            ).as_posix()

            asset_records.append({
                "character_id": character.id,
                "character_name": character.name,
                "asset_type": "canonical_character_anchor",
                "path": relative_path,
                "engine": engine_used,
                "attempts": attempts_used,
                "bytes": final_image.stat().st_size,
                "sha256": _sha256(final_image),
                "validated": True,
            })

            updated_character = updated_project[
                "characters"
            ][index - 1]

            updated_character["reference_image"] = relative_path
            updated_character["reference_images"] = [
                relative_path
            ]

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
            and all(
                item["validated"]
                for item in asset_records
            )
        ),
        "assets": asset_records,
    }

    if not manifest["all_assets_validated"]:
        raise CharacterAssetError(
            "Character asset validation did not complete."
        )

    _atomic_write_json(
        root / "character_assets.json",
        manifest,
    )

    _atomic_write_json(
        root / "project_with_character_assets.json",
        updated_project,
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate persistent character reference images."
    )

    parser.add_argument(
        "--project-json",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    project_data = json.loads(
        Path(args.project_json).read_text(
            encoding="utf-8"
        )
    )

    manifest = generate_character_assets(
        project_data,
        args.output_dir,
    )

    print(
        "Character reference generation completed.",
        flush=True,
    )

    print(
        "Project:",
        manifest["project_title"],
        flush=True,
    )

    print(
        "Validated character assets:",
        manifest["asset_count"],
        flush=True,
    )

    print(
        "Manifest:",
        Path(args.output_dir) / "character_assets.json",
        flush=True,
    )


if __name__ == "__main__":
    main()



