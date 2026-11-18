
"""Compatibility adapters between the legacy film schema and pipeline_config.

The adapter preserves legacy Wan production settings explicitly. It does not
silently pretend that the legacy 832x480/16fps output is a 1920x1080 profile.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from pipeline_config import (
    FilmProject,
    ProjectConfigError,
    SCHEMA_VERSION,
)


LEGACY_PRODUCTION = {
    "aspect_ratio": "26:15",
    "resolution": "832x480",
    "width": 832,
    "height": 480,
    "fps": 16,
}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectConfigError(f"{name} must be an object.")
    return value


def legacy_to_pipeline_project(
    legacy_project: Mapping[str, Any],
    project_id: str | None = None,
) -> FilmProject:
    """Convert a validated legacy film project into FilmProject."""

    from film_project import validate_film_project

    _require_mapping(legacy_project, "legacy_project")

    report = validate_film_project(dict(legacy_project))
    if not report["valid"]:
        raise ProjectConfigError(
            "Invalid legacy project: " + "; ".join(report["errors"])
        )

    film = legacy_project["film"]
    characters = legacy_project["characters"]
    scenes = legacy_project["scenes"]

    target_duration = film["target_duration_seconds"]
    estimated_scene_duration = target_duration / len(scenes)

    new_characters = []

    for item in characters:
        description = item.get("description", "")

        new_characters.append({
            "id": item["id"],
            "name": item["name"],
            "visual_description": description or item["name"],
            "voice_description": item.get("voice", ""),
            "appearance": description,
            "personality": item.get("personality", ""),
            "voice_profile": deepcopy(
                item.get("voice_profile", {})
            ),
            "metadata": {
                "legacy_voice_gender": item.get("voice_gender", ""),
            },
        })

    new_scenes = []

    for index, item in enumerate(scenes, 1):
        turns = deepcopy(item.get("dialogue_turns", []))

        dialogue_text = "\n".join(
            turn.get("text", "")
            for turn in turns
            if isinstance(turn, Mapping)
        )

        character_ids = []
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue

            speaker = turn.get("speaker")
            if speaker and speaker not in character_ids:
                character_ids.append(speaker)

            for listener in turn.get("listener_ids", []):
                if listener and listener not in character_ids:
                    character_ids.append(listener)

        image_prompt = item.get("image_prompt", "")
        motion_prompt = item.get("motion_prompt", "")

        combined_prompt = "\n".join(
            part for part in (image_prompt, motion_prompt) if part
        )

        new_scenes.append({
            "id": item["scene_id"],
            "order": index,
            "duration_seconds": estimated_scene_duration,
            "visual_prompt": combined_prompt or item.get("visual_subject", "Scene"),
            "dialogue": dialogue_text,
            "character_ids": character_ids,
            "location": item.get("location", ""),
            "camera": {
                "description": item.get("camera", ""),
            },
            "sound_design": {
                "description": item.get("sound", ""),
            },
            "transition_out": item.get("transition", "cut"),
            "visual_subject": item.get("visual_subject", ""),
            "action": item.get("action", ""),
            "shot_plan": deepcopy(item.get("shot_plan", [])),
            "dialogue_turns": turns,
            "props": (
                [item["props"]]
                if item.get("props")
                else []
            ),
            "metadata": {
                "legacy_narration": item.get("narration", ""),
                "legacy_image_prompt": image_prompt,
                "legacy_motion_prompt": motion_prompt,
                "legacy_sound": item.get("sound", ""),
                "legacy_camera": item.get("camera", ""),
                "legacy_transition": item.get("transition", ""),
                "duration_was_estimated": True,
            },
        })

    project = FilmProject.from_dict({
        "project_id": (
            project_id
            or film.get("title", "legacy-film").lower().replace(" ", "-")
        ),
        "title": film["title"],
        "brief": film.get("description", "") or film.get("topic", ""),
        "target_duration_seconds": target_duration,
        "visual_style": film.get("visual_style", "cinematic realistic"),
        "profile": "landscape",
        "language": film.get("language", "tr"),
        "characters": new_characters,
        "scenes": new_scenes,
        "character_bible": film.get("character_bible", ""),
        "metadata": {
            "legacy_schema_version": legacy_project.get("schema_version"),
            "legacy_tags": deepcopy(film.get("tags", [])),
            "legacy_topic": film.get("topic", ""),
        },
        "production_settings": {
            **LEGACY_PRODUCTION,
            "source": "legacy_film_project",
            "note": (
                "These exact settings override the generic landscape "
                "profile when using the legacy Wan worker."
            ),
        },
    })

    return project


def pipeline_project_to_legacy(
    project: FilmProject,
) -> dict[str, Any]:
    """Convert FilmProject to the strict legacy format without data loss."""

    from film_project import (
        SCHEMA_VERSION as LEGACY_SCHEMA_VERSION,
        validate_film_project,
    )

    if not isinstance(project, FilmProject):
        raise ProjectConfigError(
            "project must be a FilmProject instance."
        )

    if not 2 <= len(project.characters) <= 8:
        raise ProjectConfigError(
            "Legacy format requires between 2 and 8 characters."
        )

    if not 8 <= len(project.scenes) <= 15:
        raise ProjectConfigError(
            "Legacy format requires between 8 and 15 scenes."
        )

    settings = project.production_settings

    for key, expected in LEGACY_PRODUCTION.items():
        if settings.get(key) != expected:
            raise ProjectConfigError(
                f"Legacy conversion requires production setting "
                f"{key}={expected!r}; got {settings.get(key)!r}."
            )

    new_characters = []

    for character in project.characters:
        new_characters.append({
            "id": character.id,
            "name": character.name,
            "description": (
                character.visual_description
                or character.appearance
            ),
            "personality": character.personality,
            "voice": character.voice_description,
            "voice_gender": character.metadata.get(
                "legacy_voice_gender", "male"
            ),
            "voice_profile": deepcopy(character.voice_profile),
        })

    new_scenes = []

    for scene in project.scenes:
        metadata = scene.metadata

        image_prompt = (
            metadata.get("legacy_image_prompt")
            or scene.visual_prompt
        )
        motion_prompt = (
            metadata.get("legacy_motion_prompt")
            or scene.action
            or scene.visual_prompt
        )
        sound = (
            metadata.get("legacy_sound")
            or scene.sound_design.get("description", "")
        )
        camera = (
            metadata.get("legacy_camera")
            or scene.camera.get("description", "")
        )

        turns = deepcopy(scene.dialogue_turns)

        new_scenes.append({
            "scene_id": scene.id,
            "narration": metadata.get("legacy_narration", ""),
            "visual_subject": scene.visual_subject,
            "action": scene.action,
            "location": scene.location,
            "props": ", ".join(scene.props),
            "camera": camera,
            "image_prompt": image_prompt,
            "motion_prompt": motion_prompt,
            "sound": sound,
            "dialogue_turns": turns,
            "shot_plan": deepcopy(scene.shot_plan),
            "transition": (
                metadata.get("legacy_transition")
                or scene.transition_out
            ),
        })

    legacy_project = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "film": {
            "title": project.title,
            "topic": project.metadata.get("legacy_topic", ""),
            "description": project.brief,
            "language": project.language,
            "target_duration_seconds": int(
                round(project.target_duration_seconds)
            ),
            "visual_style": project.visual_style,
            "character_bible": project.character_bible,
            "tags": deepcopy(
                project.metadata.get("legacy_tags", [])
            ),
        },
        "production": {
            "aspect_ratio": LEGACY_PRODUCTION["aspect_ratio"],
            "resolution": LEGACY_PRODUCTION["resolution"],
            "fps": LEGACY_PRODUCTION["fps"],
        },
        "characters": new_characters,
        "scenes": new_scenes,
    }

    report = validate_film_project(legacy_project)

    if not report["valid"]:
        raise ProjectConfigError(
            "Converted legacy project is invalid: "
            + "; ".join(report["errors"])
        )

    return legacy_project


__all__ = [
    "LEGACY_PRODUCTION",
    "legacy_to_pipeline_project",
    "pipeline_project_to_legacy",
]
