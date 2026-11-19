
"""Character voice profiles and dialogue scheduling utilities."""

GEMINI_VOICES = [
    "Charon",
    "Kore",
    "Fenrir",
    "Aoede",
    "Puck",
    "Leda",
    "Orus",
    "Zephyr",
]

DEFAULT_TTS_ORDER = ["xai", "gemini", "edge", "gtts"]


def _clean(value, default=""):
    if value is None:
        return default
    return str(value).strip()


def build_voice_profiles(characters):
    """Create stable voice settings for every character."""
    if not isinstance(characters, list):
        raise ValueError("Characters must be a list.")

    if not 2 <= len(characters) <= 8:
        raise ValueError("Cast must contain 2 to 8 characters.")

    profiles = {}
    used_gemini_voices = set()

    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            raise ValueError("Every character must be an object.")

        character_id = _clean(character.get("id"))
        name = _clean(character.get("name"))

        if not character_id or not name:
            raise ValueError("Every character needs an ID and name.")

        if character_id in profiles:
            raise ValueError(f"Duplicate character ID: {character_id}")

        voice = character.get("voice_profile", {})
        if not isinstance(voice, dict):
            voice = {}

        gender = _clean(
            character.get("voice_gender"), "male"
        ).lower()

        edge_default = (
            "tr-TR-EmelNeural"
            if gender == "female"
            else "tr-TR-AhmetNeural"
        )

        gemini_voice = _clean(
            voice.get("gemini"),
            GEMINI_VOICES[index % len(GEMINI_VOICES)],
        )

        if gemini_voice in used_gemini_voices:
            raise ValueError(
                f"Duplicate Gemini voice assigned to {name}."
            )

        used_gemini_voices.add(gemini_voice)

        profiles[character_id] = {
            "character_id": character_id,
            "name": name,

            # Provider-specific stable voice identifiers.
            "xai": _clean(voice.get("xai")),
            "gemini": gemini_voice,
            "edge": _clean(voice.get("edge"), edge_default),

            # Optional narration/voice direction metadata.
            "personality": _clean(
                character.get("personality")
            ),
            "voice_description": _clean(
                character.get("voice")
            ),
            "voice_gender": gender,

            # Keep provider preferences attached to the character.
            "provider_order": list(DEFAULT_TTS_ORDER),
        }

    return profiles


def build_speech_schedule(scenes, characters):
    """Flatten scene dialogue into an ordered speech schedule."""
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Scenes must be a non-empty list.")

    profiles = build_voice_profiles(characters)
    schedule = []

    for scene_index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError(f"Invalid scene: {scene_index + 1}")

        turns = scene.get("dialogue_turns", [])

        if not isinstance(turns, list):
            raise ValueError(
                f"Scene {scene_index + 1}: dialogue_turns must be a list."
            )

        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError("Invalid dialogue turn.")

            speaker = _clean(turn.get("speaker"))
            dialogue = _clean(turn.get("text"))

            if speaker not in profiles:
                raise ValueError(
                    f"Unknown speaker in scene {scene_index + 1}: {speaker}"
                )

            if not dialogue:
                raise ValueError("Dialogue text cannot be empty.")

            raw_listeners = turn.get("listener_ids", [])
            if not isinstance(raw_listeners, list):
                raise ValueError("listener_ids must be a list.")

            listeners = []
            for listener in raw_listeners:
                listener_id = _clean(listener)
                if (
                    listener_id in profiles
                    and listener_id != speaker
                    and listener_id not in listeners
                ):
                    listeners.append(listener_id)

            schedule.append({
                "scene_index": scene_index,
                "turn_index": turn_index,
                "speaker_id": speaker,
                "speaker_name": profiles[speaker]["name"],
                "text": dialogue,
                "voice": profiles[speaker],
                "emotion": _clean(
                    turn.get("emotion"), "neutral"
                ),
                "delivery": _clean(
                    turn.get("delivery"), "natural"
                ),
                "listener_ids": listeners,
            })

    if not schedule:
        raise ValueError("The film contains no character dialogue.")

    return schedule
