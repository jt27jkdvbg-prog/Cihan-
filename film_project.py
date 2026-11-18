
"""Canonical, dependency-free schema for the AI short-film pipeline.

This module normalizes generated scripts into one stable project structure,
validates the project, and adapts it to the existing generate.py format.

Current visual worker target:
- Landscape frame: 832x480 (exact ratio 26:15)
- Frame rate: 16 fps
"""

from copy import deepcopy


SCHEMA_VERSION = "1.0"

MIN_SCENES = 8
MAX_SCENES = 15
MIN_CHARACTERS = 2
MAX_CHARACTERS = 8

# Must match the current Wan2.1 worker in wan-vace-kaggle.yml.
PRODUCTION_DEFAULTS = {
    "aspect_ratio": "26:15",
    "resolution": "832x480",
    "fps": 16,
}


def _text(value, default=""):
    """Return a trimmed string without turning None into 'None'."""
    if value is None:
        return default
    return str(value).strip()


def _list(value, field_name):
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return value


def _unique_id(value, fallback):
    value = _text(value)
    return value or fallback


def build_film_project(
    script,
    topic="",
    target_seconds=60,
    language="Turkish",
    visual_style="cinematic realistic",
):
    """Normalize a generated script into the canonical film-project format."""
    if not isinstance(script, dict):
        raise ValueError("Script must be a JSON object.")

    source = deepcopy(script)
    raw_characters = _list(source.get("characters"), "characters")
    raw_scenes = _list(source.get("scenes"), "scenes")

    if not MIN_CHARACTERS <= len(raw_characters) <= MAX_CHARACTERS:
        raise ValueError(
            f"Character count must be between {MIN_CHARACTERS} "
            f"and {MAX_CHARACTERS}; got {len(raw_characters)}."
        )

    if not MIN_SCENES <= len(raw_scenes) <= MAX_SCENES:
        raise ValueError(
            f"Scene count must be between {MIN_SCENES} "
            f"and {MAX_SCENES}; got {len(raw_scenes)}."
        )

    normalized_characters = []
    character_ids = set()
    character_names = {}

    for index, item in enumerate(raw_characters, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Character {index} must be an object.")

        character_id = _unique_id(item.get("id"), f"character_{index:02d}")
        name = _text(item.get("name"))

        if not name:
            raise ValueError(f"Character {index} needs a name.")
        if character_id in character_ids:
            raise ValueError(f"Duplicate character ID: {character_id}")

        name_key = name.casefold()
        if name_key in character_names:
            raise ValueError(f"Duplicate character name: {name}")

        character_ids.add(character_id)
        character_names[name_key] = character_id

        voice_profile = item.get("voice_profile")
        if not isinstance(voice_profile, dict):
            voice_profile = {}

        normalized_characters.append({
            "id": character_id,
            "name": name,
            "description": _text(item.get("description")),
            "personality": _text(item.get("personality")),
            "voice": _text(item.get("voice")),
            "voice_gender": _text(item.get("voice_gender")) or "male",
            "voice_profile": {
                "edge": _text(voice_profile.get("edge")),
                "gemini": _text(voice_profile.get("gemini")),
            },
        })

    normalized_scenes = []
    scene_ids = set()

    for index, item in enumerate(raw_scenes, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Scene {index} must be an object.")

        scene_id = _unique_id(item.get("scene_id"), f"scene_{index:02d}")
        if scene_id in scene_ids:
            raise ValueError(f"Duplicate scene ID: {scene_id}")
        scene_ids.add(scene_id)

        raw_turns = item.get("dialogue_turns") or []
        if not isinstance(raw_turns, list):
            raise ValueError(
                f"Scene {index}: dialogue_turns must be a list."
            )

        turns = []
        for turn_index, turn in enumerate(raw_turns, 1):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"Scene {index}, dialogue {turn_index} must be an object."
                )

            speaker = _text(turn.get("speaker"))
            speaker_id = (
                speaker if speaker in character_ids
                else character_names.get(speaker.casefold())
            )
            if not speaker_id:
                raise ValueError(
                    f"Scene {index}, dialogue {turn_index}: "
                    f"unknown speaker '{speaker}'."
                )

            dialogue_text = _text(turn.get("text"))
            if not dialogue_text:
                raise ValueError(
                    f"Scene {index}, dialogue {turn_index}: "
                    "dialogue text is empty."
                )

            raw_listeners = turn.get("listener_ids") or []
            if not isinstance(raw_listeners, list):
                raw_listeners = []

            listeners = []
            for listener in raw_listeners:
                listener_text = _text(listener)
                listener_id = (
                    listener_text if listener_text in character_ids
                    else character_names.get(listener_text.casefold())
                )
                if listener_id and listener_id != speaker_id:
                    if listener_id not in listeners:
                        listeners.append(listener_id)

            turns.append({
                "speaker": speaker_id,
                "text": dialogue_text,
                "emotion": _text(turn.get("emotion")) or "neutral",
                "delivery": _text(turn.get("delivery")) or "natural",
                "listener_ids": listeners,
            })

        shot_plan = item.get("shot_plan")
        if not isinstance(shot_plan, list):
            shot_plan = []

        normalized_scenes.append({
            "scene_id": scene_id,
            "narration": _text(item.get("narration")),
            "visual_subject": _text(item.get("visual_subject")),
            "action": _text(item.get("action")),
            "location": _text(item.get("location")),
            "props": _text(item.get("props")),
            "camera": _text(item.get("camera")),
            "image_prompt": _text(item.get("image_prompt")),
            "motion_prompt": _text(item.get("motion_prompt")),
            "sound": _text(item.get("sound")),
            "dialogue_turns": turns,
            "shot_plan": deepcopy(shot_plan),
            "transition": _text(item.get("transition")),
        })

    try:
        duration = int(target_seconds)
    except (TypeError, ValueError, OverflowError):
        duration = 60

    duration = max(15, min(duration, 180))

    project = {
        "schema_version": SCHEMA_VERSION,
        "film": {
            "title": (
                _text(source.get("title"))
                or _text(topic)
                or "Untitled Film"
            ),
            "topic": _text(topic),
            "description": _text(source.get("description")),
            "language": _text(language) or "Turkish",
            "target_duration_seconds": duration,
            "visual_style": (
                _text(source.get("style"))
                or _text(visual_style)
                or "cinematic realistic"
            ),
            "character_bible": _text(source.get("character_bible")),
            "tags": (
                deepcopy(source.get("tags"))
                if isinstance(source.get("tags"), list)
                else []
            ),
        },
        "production": deepcopy(PRODUCTION_DEFAULTS),
        "characters": normalized_characters,
        "scenes": normalized_scenes,
    }

    report = validate_film_project(project)
    if not report["valid"]:
        raise ValueError(
            "Film project validation failed: "
            + "; ".join(report["errors"])
        )

    return project


def validate_film_project(project):
    """Validate the canonical project without changing it."""
    errors = []
    warnings = []

    if not isinstance(project, dict):
        return {
            "valid": False,
            "errors": ["Project must be an object."],
            "warnings": [],
            "scene_count": 0,
            "character_count": 0,
            "dialogue_turn_count": 0,
        }

    if project.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema version: {project.get('schema_version')}"
        )

    film = project.get("film")
    if not isinstance(film, dict):
        errors.append("Missing film metadata.")
        film = {}

    if not _text(film.get("title")):
        errors.append("Film title is missing.")

    duration = film.get("target_duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, int):
        errors.append("Target duration must be an integer.")
    elif not 15 <= duration <= 180:
        errors.append("Target duration must be between 15 and 180 seconds.")

    characters = project.get("characters")
    if not isinstance(characters, list):
        errors.append("Characters must be a list.")
        characters = []

    if not MIN_CHARACTERS <= len(characters) <= MAX_CHARACTERS:
        errors.append(
            f"Character count must be between {MIN_CHARACTERS} "
            f"and {MAX_CHARACTERS}."
        )

    character_ids = set()
    character_names = set()

    for index, character in enumerate(characters, 1):
        if not isinstance(character, dict):
            errors.append(f"Character {index} is invalid.")
            continue

        character_id = _text(character.get("id"))
        name = _text(character.get("name"))
        if not character_id or not name:
            errors.append(f"Character {index} needs an ID and name.")
            continue

        if character_id in character_ids:
            errors.append(f"Duplicate character ID: {character_id}")
        if name.casefold() in character_names:
            errors.append(f"Duplicate character name: {name}")

        character_ids.add(character_id)
        character_names.add(name.casefold())

    scenes = project.get("scenes")
    if not isinstance(scenes, list):
        errors.append("Scenes must be a list.")
        scenes = []

    if not MIN_SCENES <= len(scenes) <= MAX_SCENES:
        errors.append(
            f"Scene count must be between {MIN_SCENES} and {MAX_SCENES}."
        )

    scene_ids = set()
    dialogue_count = 0

    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            errors.append(f"Scene {index} is invalid.")
            continue

        scene_id = _text(scene.get("scene_id"))
        if not scene_id:
            errors.append(f"Scene {index} has no scene ID.")
        elif scene_id in scene_ids:
            errors.append(f"Duplicate scene ID: {scene_id}")
        scene_ids.add(scene_id)

        for field in ("image_prompt", "motion_prompt", "sound"):
            if not _text(scene.get(field)):
                errors.append(f"Scene {index} is missing {field}.")

        turns = scene.get("dialogue_turns")
        if not isinstance(turns, list):
            errors.append(f"Scene {index}: dialogue_turns must be a list.")
            continue

        if not _text(scene.get("narration")) and not turns:
            errors.append(f"Scene {index} needs narration or dialogue.")

        for turn_index, turn in enumerate(turns, 1):
            if not isinstance(turn, dict):
                errors.append(
                    f"Scene {index}, dialogue {turn_index} is invalid."
                )
                continue

            speaker = _text(turn.get("speaker"))
            if speaker not in character_ids:
                errors.append(
                    f"Scene {index}, dialogue {turn_index}: "
                    f"unknown speaker '{speaker}'."
                )

            if not _text(turn.get("text")):
                errors.append(
                    f"Scene {index}, dialogue {turn_index} is empty."
                )

            listeners = turn.get("listener_ids", [])
            if not isinstance(listeners, list):
                errors.append(
                    f"Scene {index}, dialogue {turn_index}: "
                    "listener_ids must be a list."
                )
            else:
                for listener in listeners:
                    if listener not in character_ids:
                        errors.append(
                            f"Scene {index}, dialogue {turn_index}: "
                            f"unknown listener '{listener}'."
                        )

            dialogue_count += 1

    if dialogue_count == 0:
        errors.append("The film has no character dialogue.")

    production = project.get("production")
    if not isinstance(production, dict):
        errors.append("Production settings are missing.")
    else:
        for key, expected in PRODUCTION_DEFAULTS.items():
            actual = production.get(key)
            if actual != expected:
                errors.append(
                    f"Invalid production setting {key}: "
                    f"expected {expected}, got {actual}."
                )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "scene_count": len(scenes),
        "character_count": len(characters),
        "dialogue_turn_count": dialogue_count,
    }


def to_legacy_script(project):
    """Adapt a valid canonical project for the existing generate.py code."""
    report = validate_film_project(project)
    if not report["valid"]:
        raise ValueError(
            "Cannot convert invalid project: "
            + "; ".join(report["errors"])
        )

    film = project["film"]

    return {
        "title": film["title"],
        "description": film.get("description", ""),
        "tags": deepcopy(film.get("tags", [])),
        "style": film.get("visual_style", "cinematic realistic"),
        "character_bible": film.get("character_bible", ""),
        "characters": deepcopy(project["characters"]),
        "scenes": deepcopy(project["scenes"]),
        "fallback_mode": False,
    }

