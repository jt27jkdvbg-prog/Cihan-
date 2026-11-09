
"""Structured cast and dialogue utilities for multi-character films."""

MIN_CAST = 2
MAX_CAST = 8


def normalize_cast(cast):
    """Normalize a list of character dictionaries and enforce a 2–8 cast size."""
    if not isinstance(cast, list):
        raise ValueError("Cast must be a list.")

    normalized = []
    seen = set()

    for item in cast:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            continue

        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "id": str(item.get("id") or key.replace(" ", "_")),
            "name": name,
            "description": str(item.get("description", "")).strip(),
            "voice": str(item.get("voice", "")).strip(),
            "personality": str(item.get("personality", "")).strip(),
        })

    if not MIN_CAST <= len(normalized) <= MAX_CAST:
        raise ValueError(
            f"Cast must contain between {MIN_CAST} and {MAX_CAST} unique characters."
        )

    return normalized


def normalize_dialogue_turns(scene, cast):
    """Convert supported dialogue formats into ordered speaker turns."""
    if not isinstance(scene, dict):
        raise ValueError("Scene must be a dictionary.")

    cast_by_name = {c["name"].casefold(): c for c in cast}
    cast_by_id = {c["id"].casefold(): c for c in cast}

    raw = scene.get("dialogue_turns")

    # Backward compatibility: old scripts may store dialogue as plain text.
    if raw is None:
        legacy = scene.get("dialogue", "")
        if isinstance(legacy, str) and legacy.strip():
            raise ValueError(
                "Legacy dialogue has no speaker. Convert it to dialogue_turns "
                "with an explicit speaker for every line."
            )
        raw = []

    if not isinstance(raw, list):
        raise ValueError("dialogue_turns must be a list.")

    turns = []
    for index, turn in enumerate(raw, 1):
        if not isinstance(turn, dict):
            raise ValueError(f"Dialogue turn {index} must be an object.")

        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()

        if not speaker or not text:
            raise ValueError(
                f"Dialogue turn {index} needs both speaker and text."
            )

        character = (
            cast_by_name.get(speaker.casefold())
            or cast_by_id.get(speaker.casefold())
        )
        if character is None:
            raise ValueError(
                f"Dialogue turn {index} uses unknown speaker: {speaker}"
            )

        turns.append({
            "speaker_id": character["id"],
            "speaker": character["name"],
            "text": text,
            "delivery": str(turn.get("delivery", "natural")).strip(),
            "emotion": str(turn.get("emotion", "neutral")).strip(),
            "listener_ids": [
                c["id"]
                for c in cast
                if c["id"] in turn.get("listener_ids", [])
                and c["id"] != character["id"]
            ],
        })

    return turns


def validate_cast_and_dialogue(scenes, cast):
    """Return a structured validation report without modifying input data."""
    errors = []
    warnings = []

    try:
        normalized_cast = normalize_cast(cast)
    except ValueError as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}

    if not isinstance(scenes, list) or not scenes:
        return {
            "valid": False,
            "errors": ["Scenes must be a non-empty list."],
            "warnings": [],
        }

    dialogue_count = 0

    for scene_number, scene in enumerate(scenes, 1):
        try:
            turns = normalize_dialogue_turns(scene, normalized_cast)
            dialogue_count += len(turns)
        except ValueError as exc:
            errors.append(f"Scene {scene_number}: {exc}")
            continue

        if len(turns) == 1:
            warnings.append(
                f"Scene {scene_number}: only one dialogue turn; "
                "consider adding a response or reaction."
            )

        speakers = {turn["speaker_id"] for turn in turns}
        if len(speakers) > 1 and len(turns) < 2:
            warnings.append(
                f"Scene {scene_number}: dialogue needs clearer turn-taking."
            )

    if dialogue_count == 0:
        warnings.append("No character dialogue found in the script.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "cast_count": len(normalized_cast),
        "dialogue_turn_count": dialogue_count,
    }
