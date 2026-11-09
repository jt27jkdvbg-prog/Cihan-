"""Character voice and dialogue scheduling utilities."""

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


def build_voice_profiles(characters):
    """Create stable voice settings for every character."""
    if not isinstance(characters, list):
        raise ValueError("Characters must be a list.")

    profiles = {}
    used_gemini_voices = set()

    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            continue

        character_id = str(character.get("id", "")).strip()
        name = str(character.get("name", "")).strip()

        if not character_id or not name:
            raise ValueError("Every character needs an ID and name.")

        if character_id in profiles:
            raise ValueError(f"Duplicate character ID: {character_id}")

        voice = character.get("voice_profile", {})
        if not isinstance(voice, dict):
            voice = {}

        gemini_voice = str(
            voice.get("gemini") or GEMINI_VOICES[index % len(GEMINI_VOICES)]
        ).strip()

        if gemini_voice in used_gemini_voices:
            raise ValueError(
                f"Duplicate Gemini voice assigned to {name}."
            )

        used_gemini_voices.add(gemini_voice)

        gender = str(character.get("voice_gender", "male")).lower()
        edge_default = (
            "tr-TR-EmelNeural"
            if gender == "female"
            else "tr-TR-AhmetNeural"
        )

        profiles[character_id] = {
            "character_id": character_id,
            "name": name,
            "gemini": gemini_voice,
            "edge": str(voice.get("edge") or edge_default).strip(),
            "personality": str(
                character.get("personality", "")
            ).strip(),
            "voice_description": str(
                character.get("voice", "")
            ).strip(),
        }

    if not 2 <= len(profiles) <= 8:
        raise ValueError("Cast must contain 2 to 8 characters.")

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

            speaker = str(turn.get("speaker", "")).strip()
            text = str(turn.get("text", "")).strip()

            if speaker not in profiles:
                raise ValueError(
                    f"Unknown speaker in scene {scene_index + 1}: {speaker}"
                )

            if not text:
                raise ValueError("Dialogue text cannot be empty.")

            profile = profiles[speaker]

            schedule.append({
                "scene_index": scene_index,
                "turn_index": turn_index,
                "speaker_id": speaker,
                "speaker_name": profile["name"],
                "text": text,
                "voice": profile,
                "emotion": str(
                    turn.get("emotion", "neutral")
                ).strip(),
                "delivery": str(
                    turn.get("delivery", "natural")
                ).strip(),
                "listener_ids": [
                    item for item in turn.get("listener_ids", [])
                    if item in profiles and item != speaker
                ],
            })

    if not schedule:
        raise ValueError("The film contains no character dialogue.")

    return schedule
