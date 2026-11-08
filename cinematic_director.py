
"""Cinematic short-film prompt and script validation helpers."""


def build_script_prompt(topic, notes="", target_seconds=60, language="Turkish"):
    try:
        seconds = max(15, int(target_seconds))
    except (TypeError, ValueError):
        seconds = 60

    scene_count = max(8, min(12, round(seconds / 6)))

    return f"""
You are an expert film director, screenwriter and cinematographer.

Create a cinematic {language} short film about: {topic}
Additional context: {notes or "Use relevant and reliable information."}
Target duration: approximately {seconds} seconds.
Create exactly {scene_count} scenes.

Return ONLY valid JSON:
{{
  "title": "Film title",
  "style": "Visual style and color palette",
  "character_bible": "Detailed descriptions of recurring characters",
  "scenes": [
    {{
      "narration": "Turkish narration",
      "dialogue": "Optional spoken dialogue",
      "image_prompt": "Detailed cinematic visual description",
      "motion_prompt": "Visible physical action and camera movement",
      "sound": "Atmosphere and sound effects",
      "transition": "Transition to next scene"
    }}
  ]
}}

PRODUCTION RULES:
- Vertical 9:16 cinematic composition.
- Strong opening hook and satisfying ending.
- Clear beginning, development and conclusion.
- Consistent faces, clothing and character identities.
- Every scene must contain visible physical action.
- Use tracking, dolly, pan, tilt and varied camera shots.
- Natural Turkish narration and dialogue.
- Never prohibit speech or lip movement.
- Add scene-specific sound effects and ambience.
- Match visual style to the subject.
- Avoid static poster-like compositions.
- Keep important subjects inside the vertical frame.
- Do not invent facts about real people or history.
""".strip()


def enrich_image_prompt(style, character_bible, scene):
    parts = [
        f"Visual style: {style or 'cinematic realism'}.",
        f"Character continuity: {character_bible}." if character_bible else "",
        scene.get("image_prompt", ""),
        "Cinematic lighting, realistic details, vertical 9:16 framing.",
        "Preserve recurring characters' faces, clothing and identity."
    ]
    return " ".join(part for part in parts if part)


def validate_script(data, target_seconds=60):
    errors = []
    warnings = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Invalid JSON object."], "warnings": []}

    scenes = data.get("scenes")

    if not isinstance(scenes, list):
        return {"valid": False, "errors": ["Scenes must be a list."], "warnings": []}

    if not data.get("title"):
        errors.append("Missing title.")

    if not 8 <= len(scenes) <= 12:
        warnings.append("Expected between 8 and 12 scenes.")

    for number, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            errors.append(f"Scene {number} is invalid.")
            continue

        for field in ("narration", "image_prompt", "motion_prompt", "sound"):
            if not str(scene.get(field, "")).strip():
                errors.append(f"Scene {number}: missing {field}.")

    narration = " ".join(
        str(scene.get("narration", ""))
        for scene in scenes if isinstance(scene, dict)
    )

    word_count = len(narration.split())

    try:
        seconds = max(1, int(target_seconds))
    except (TypeError, ValueError):
        seconds = 60

    if word_count > seconds * 2.8:
        warnings.append("Narration may be too long.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "scene_count": len(scenes),
        "narration_word_count": word_count
    }
