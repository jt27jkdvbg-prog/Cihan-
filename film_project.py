
"""Canonical, dependency-free schema for the AI short-film pipeline.

Normalizes generated scripts into one stable project structure,
validates the project, and adapts it to the existing generate.py format.

Visual worker target:
- Landscape frame: 832x480 (26:15)
- Frame rate: 16 fps

Audio:
- Free-first TTS provider order
- Optional paid xAI/Grok TTS
- Paid music and sound-effect generation disabled by default
- Character-specific voice profiles
- Music, ambience, sound effects and subtitle preferences

Cost policy:
- Free-only mode by default
- Paid services require explicit approval
- Unknown prices block execution
- Project budget defaults to zero

No API keys or external dependencies belong in this module.
"""

from copy import deepcopy


SCHEMA_VERSION = "1.0"

MIN_SCENES = 8
MAX_SCENES = 15
MIN_CHARACTERS = 2
MAX_CHARACTERS = 8

PRODUCTION_DEFAULTS = {
    "aspect_ratio": "26:15",
    "resolution": "832x480",
    "fps": 16,
}

# Free-first order. xAI is an optional paid fallback and must not
# be used unless the production cost policy explicitly permits it.
DEFAULT_AUDIO_SETTINGS = {
    "enabled": True,
    "tts_provider_order": ["gemini", "edge", "gtts", "xai"],
    "narration_enabled": True,
    "subtitles": {
        "enabled": True,
        "format": "srt",
        "burn_in": False,
    },
    "music": {
        "enabled": False,
        "provider": "licensed_library",
        "model": "",
        "prompt": "",
        "volume": 0.18,
        "ducking": True,
    },
    "ambience": {
        "enabled": False,
        "provider": "licensed_library",
        "prompt": "",
        "volume": 0.22,
    },
    "sound_effects": {
        "enabled": False,
        "provider": "licensed_library",
        "model": "",
        "volume": 0.45,
    },
    "mix": {
        "sample_rate": 48000,
        "channels": 2,
        "loudness_lufs": -16.0,
        "true_peak_db": -1.5,
        "fade_in_seconds": 1.0,
        "fade_out_seconds": 2.0,
    },
}

DEFAULT_COST_POLICY = {
    "mode": "free_only",
    "paid_services_allowed": False,
    "require_approval": True,
    "show_cost_before_run": True,
    "block_unknown_prices": True,
    "currency": "USD",
    "max_project_budget": 0.0,
    "record_estimated_costs": True,
    "record_actual_costs": True,
}

# This is a classification, not a live price quote.
# Exact prices and quotas must be checked by the production cost checker.
SERVICE_COST_CATALOG = {
    "gemini_tts": {
        "category": "free_quota",
        "description": "Free quota may be available; usage limits apply.",
        "requires_price_check": True,
    },
    "edge_tts": {
        "category": "free",
        "description": "Free-to-use TTS service; availability may change.",
        "requires_price_check": False,
    },
    "gtts": {
        "category": "free",
        "description": "Free basic text-to-speech library; availability may change.",
        "requires_price_check": False,
    },
    "xai_tts": {
        "category": "paid",
        "description": "Paid usage-based TTS service.",
        "requires_price_check": True,
    },
    "fal_music": {
        "category": "paid",
        "description": "Paid usage-based music generation.",
        "requires_price_check": True,
    },
    "fal_sound_effects": {
        "category": "paid",
        "description": "Paid usage-based sound-effect generation.",
        "requires_price_check": True,
    },
    "licensed_library": {
        "category": "free_or_licensed",
        "description": "Use only assets whose license permits the intended use.",
        "requires_price_check": True,
    },
}

TTS_PROVIDERS = {"xai", "gemini", "edge", "gtts"}


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


def _clamp_float(value, default, minimum, maximum):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        result = float(default)

    return max(minimum, min(result, maximum))


def _normalize_provider_order(value):
    """Normalize TTS fallback order and remove unsupported duplicates."""
    if not isinstance(value, list):
        value = DEFAULT_AUDIO_SETTINGS["tts_provider_order"]

    result = []

    for provider in value:
        provider = _text(provider).lower()

        if provider in TTS_PROVIDERS and provider not in result:
            result.append(provider)

    if not result:
        result = list(DEFAULT_AUDIO_SETTINGS["tts_provider_order"])

    return result


def _normalize_cost_policy(raw_policy=None):
    """Create a safe cost policy. Paid usage is disabled by default."""
    raw = raw_policy if isinstance(raw_policy, dict) else {}
    result = deepcopy(DEFAULT_COST_POLICY)

    mode = _text(raw.get("mode"), result["mode"]).lower()
    if mode not in {"free_only", "approval_required", "paid_allowed"}:
        mode = "free_only"

    result["mode"] = mode

    # A project cannot silently enable paid services.
    requested_paid = bool(
        raw.get("paid_services_allowed", result["paid_services_allowed"])
    )

    if mode == "free_only":
        result["paid_services_allowed"] = False
    else:
        result["paid_services_allowed"] = requested_paid

    # Approval and price checks remain mandatory for paid services.
    result["require_approval"] = True
    result["show_cost_before_run"] = True
    result["block_unknown_prices"] = True

    currency = _text(raw.get("currency"), "USD").upper()
    if len(currency) == 3 and currency.isalpha():
        result["currency"] = currency

    result["max_project_budget"] = _clamp_float(
        raw.get("max_project_budget"),
        result["max_project_budget"],
        0.0,
        1000000.0,
    )

    result["record_estimated_costs"] = bool(
        raw.get(
            "record_estimated_costs",
            result["record_estimated_costs"],
        )
    )
    result["record_actual_costs"] = bool(
        raw.get(
            "record_actual_costs",
            result["record_actual_costs"],
        )
    )

    if mode == "free_only":
        result["max_project_budget"] = 0.0

    return result


def _normalize_audio_settings(raw_audio=None):
    """Create a complete, free-first audio-settings object."""
    raw = raw_audio if isinstance(raw_audio, dict) else {}
    result = deepcopy(DEFAULT_AUDIO_SETTINGS)

    result["enabled"] = bool(raw.get("enabled", result["enabled"]))
    result["narration_enabled"] = bool(
        raw.get("narration_enabled", result["narration_enabled"])
    )
    result["tts_provider_order"] = _normalize_provider_order(
        raw.get("tts_provider_order")
    )

    raw_subtitles = raw.get("subtitles")
    if isinstance(raw_subtitles, dict):
        result["subtitles"]["enabled"] = bool(
            raw_subtitles.get(
                "enabled",
                result["subtitles"]["enabled"],
            )
        )
        result["subtitles"]["burn_in"] = bool(
            raw_subtitles.get(
                "burn_in",
                result["subtitles"]["burn_in"],
            )
        )

        subtitle_format = _text(
            raw_subtitles.get("format"),
            result["subtitles"]["format"],
        ).lower()

        if subtitle_format in {"srt", "vtt", "ass"}:
            result["subtitles"]["format"] = subtitle_format

    raw_music = raw.get("music")
    if isinstance(raw_music, dict):
        result["music"]["enabled"] = bool(
            raw_music.get("enabled", result["music"]["enabled"])
        )
        result["music"]["provider"] = _text(
            raw_music.get("provider"),
            result["music"]["provider"],
        ).lower()
        result["music"]["model"] = _text(
            raw_music.get("model"),
            result["music"]["model"],
        )
        result["music"]["prompt"] = _text(
            raw_music.get("prompt"),
            result["music"]["prompt"],
        )
        result["music"]["volume"] = _clamp_float(
            raw_music.get("volume"),
            result["music"]["volume"],
            0.0,
            1.0,
        )
        result["music"]["ducking"] = bool(
            raw_music.get("ducking", result["music"]["ducking"])
        )

    raw_ambience = raw.get("ambience")
    if isinstance(raw_ambience, dict):
        result["ambience"]["enabled"] = bool(
            raw_ambience.get(
                "enabled",
                result["ambience"]["enabled"],
            )
        )
        result["ambience"]["provider"] = _text(
            raw_ambience.get("provider"),
            result["ambience"]["provider"],
        ).lower()
        result["ambience"]["prompt"] = _text(
            raw_ambience.get("prompt"),
            result["ambience"]["prompt"],
        )
        result["ambience"]["volume"] = _clamp_float(
            raw_ambience.get("volume"),
            result["ambience"]["volume"],
            0.0,
            1.0,
        )

    raw_effects = raw.get("sound_effects")
    if isinstance(raw_effects, dict):
        result["sound_effects"]["enabled"] = bool(
            raw_effects.get(
                "enabled",
                result["sound_effects"]["enabled"],
            )
        )
        result["sound_effects"]["provider"] = _text(
            raw_effects.get("provider"),
            result["sound_effects"]["provider"],
        ).lower()
        result["sound_effects"]["model"] = _text(
            raw_effects.get("model"),
            result["sound_effects"]["model"],
        )
        result["sound_effects"]["volume"] = _clamp_float(
            raw_effects.get("volume"),
            result["sound_effects"]["volume"],
            0.0,
            1.0,
        )

    raw_mix = raw.get("mix")
    if isinstance(raw_mix, dict):
        mix = result["mix"]

        try:
            sample_rate = int(raw_mix.get("sample_rate", mix["sample_rate"]))
        except (TypeError, ValueError, OverflowError):
            sample_rate = mix["sample_rate"]

        if sample_rate in {22050, 44100, 48000, 96000}:
            mix["sample_rate"] = sample_rate

        try:
            channels = int(raw_mix.get("channels", mix["channels"]))
        except (TypeError, ValueError, OverflowError):
            channels = mix["channels"]

        if channels in {1, 2}:
            mix["channels"] = channels

        mix["loudness_lufs"] = _clamp_float(
            raw_mix.get("loudness_lufs"),
            mix["loudness_lufs"],
            -30.0,
            -5.0,
        )
        mix["true_peak_db"] = _clamp_float(
            raw_mix.get("true_peak_db"),
            mix["true_peak_db"],
            -9.0,
            0.0,
        )
        mix["fade_in_seconds"] = _clamp_float(
            raw_mix.get("fade_in_seconds"),
            mix["fade_in_seconds"],
            0.0,
            15.0,
        )
        mix["fade_out_seconds"] = _clamp_float(
            raw_mix.get("fade_out_seconds"),
            mix["fade_out_seconds"],
            0.0,
            15.0,
        )

    return result


def _normalize_voice_profile(raw_profile):
    """Keep provider voice IDs stable and compatible with speech_director."""
    raw = raw_profile if isinstance(raw_profile, dict) else {}

    return {
        "xai": _text(raw.get("xai")),
        "gemini": _text(raw.get("gemini")),
        "edge": _text(raw.get("edge")),
    }


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

        character_id = _unique_id(
            item.get("id"),
            f"character_{index:02d}",
        )
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

        normalized_characters.append({
            "id": character_id,
            "name": name,
            "description": _text(item.get("description")),
            "personality": _text(item.get("personality")),
            "voice": _text(item.get("voice")),
            "voice_gender": _text(item.get("voice_gender")) or "male",
            "voice_profile": _normalize_voice_profile(
                item.get("voice_profile")
            ),
        })

    normalized_scenes = []
    scene_ids = set()

    for index, item in enumerate(raw_scenes, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Scene {index} must be an object.")

        scene_id = _unique_id(
            item.get("scene_id"),
            f"scene_{index:02d}",
        )

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
                    f"Scene {index}, dialogue {turn_index} "
                    "must be an object."
                )

            speaker = _text(turn.get("speaker"))

            speaker_id = (
                speaker
                if speaker in character_ids
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
                    listener_text
                    if listener_text in character_ids
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

        raw_effects = item.get("sound_effects")
        if not isinstance(raw_effects, list):
            raw_effects = []

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
            "sound_effects": [
                _text(effect)
                for effect in raw_effects
                if _text(effect)
            ],
            "ambience": _text(item.get("ambience")),
            "music": _text(item.get("music")),
            "dialogue_turns": turns,
            "shot_plan": deepcopy(shot_plan),
            "transition": _text(item.get("transition")),
        })

    try:
        duration = int(target_seconds)
    except (TypeError, ValueError, OverflowError):
        duration = 60

    duration = max(15, min(duration, 180))

    raw_audio = source.get("audio")

    if not isinstance(raw_audio, dict):
        raw_audio = source.get("audio_settings")

    raw_cost_policy = source.get("cost_policy")

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
        "audio": _normalize_audio_settings(raw_audio),
        "cost_policy": _normalize_cost_policy(raw_cost_policy),
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

        voice_profile = character.get("voice_profile", {})

        if voice_profile is not None and not isinstance(voice_profile, dict):
            errors.append(
                f"Character {index}: voice_profile must be an object."
            )

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
            errors.append(
                f"Scene {index}: dialogue_turns must be a list."
            )
            continue

        if not _text(scene.get("narration")) and not turns:
            errors.append(
                f"Scene {index} needs narration or dialogue."
            )

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

        effects = scene.get("sound_effects", [])

        if not isinstance(effects, list):
            errors.append(
                f"Scene {index}: sound_effects must be a list."
            )

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

    # Audio is optional for older project JSON files.
    audio = project.get("audio")

    if audio is not None:
        if not isinstance(audio, dict):
            errors.append("Audio settings must be an object.")
        else:
            provider_order = audio.get("tts_provider_order")

            if not isinstance(provider_order, list) or not provider_order:
                errors.append(
                    "Audio tts_provider_order must be a non-empty list."
                )
            else:
                seen_providers = set()

                for provider in provider_order:
                    if provider not in TTS_PROVIDERS:
                        errors.append(
                            f"Unsupported TTS provider: {provider}"
                        )
                    elif provider in seen_providers:
                        errors.append(
                            f"Duplicate TTS provider: {provider}"
                        )

                    seen_providers.add(provider)

            for section in (
                "subtitles",
                "music",
                "ambience",
                "sound_effects",
                "mix",
            ):
                if not isinstance(audio.get(section), dict):
                    errors.append(
                        f"Audio setting '{section}' must be an object."
                    )

            for section in ("music", "ambience", "sound_effects"):
                settings = audio.get(section)

                if isinstance(settings, dict):
                    volume = settings.get("volume")

                    if (
                        isinstance(volume, bool)
                        or not isinstance(volume, (int, float))
                        or not 0.0 <= volume <= 1.0
                    ):
                        errors.append(
                            f"Audio {section} volume must be between 0 and 1."
                        )

    # Cost policy is optional for older project JSON files.
    cost_policy = project.get("cost_policy")

    if cost_policy is not None:
        if not isinstance(cost_policy, dict):
            errors.append("Cost policy must be an object.")
        else:
            mode = cost_policy.get("mode")

            if mode not in {
                "free_only",
                "approval_required",
                "paid_allowed",
            }:
                errors.append("Invalid cost policy mode.")

            budget = cost_policy.get("max_project_budget")

            if (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or budget < 0
            ):
                errors.append(
                    "Cost policy max_project_budget must be non-negative."
                )

            if mode == "free_only":
                if cost_policy.get("paid_services_allowed") is not False:
                    errors.append(
                        "Free-only mode cannot allow paid services."
                    )
                if budget != 0:
                    errors.append(
                        "Free-only mode must have a zero project budget."
                    )

            if cost_policy.get("require_approval") is not True:
                errors.append(
                    "Paid services must require explicit approval."
                )

            if cost_policy.get("block_unknown_prices") is not True:
                errors.append(
                    "Services with unknown prices must be blocked."
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

    result = {
        "title": film["title"],
        "description": film.get("description", ""),
        "tags": deepcopy(film.get("tags", [])),
        "style": film.get("visual_style", "cinematic realistic"),
        "character_bible": film.get("character_bible", ""),
        "characters": deepcopy(project["characters"]),
        "scenes": deepcopy(project["scenes"]),
        "fallback_mode": False,
    }

    if isinstance(project.get("audio"), dict):
        result["audio"] = deepcopy(project["audio"])

    if isinstance(project.get("cost_policy"), dict):
        result["cost_policy"] = deepcopy(project["cost_policy"])

    return result

