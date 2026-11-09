
"""Cinematic script generation, character dialogue and validation."""

STYLE_PRESETS = {
    "3d": (
        "Premium stylized 3D animated feature-film aesthetic, expressive "
        "original characters, detailed environments, cinematic lighting "
        "and fluid character animation."
    ),
    "anime": (
        "Original Japanese anime feature-film aesthetic, expressive "
        "character animation, detailed backgrounds and cinematic lighting."
    ),
    "realistic": (
        "High-end live-action cinematic realism, natural human movement, "
        "detailed environments and realistic lighting."
    ),
    "2d": (
        "Premium hand-drawn 2D animation, expressive poses and fluid motion."
    ),
    "game": (
        "AAA video game cinematic, detailed 3D characters and dramatic lighting."
    ),
    "storybook": (
        "Animated storybook aesthetic, painterly environments and rich colors."
    ),
    "comic": (
        "Cinematic graphic novel animation, dynamic compositions and shadows."
    ),
}


def resolve_visual_style(style="3d", custom_style=""):
    aliases = {
        "3d animasyon": "3d",
        "3d animation": "3d",
        "gerçekçi sinema": "realistic",
        "2d animasyon": "2d",
        "2d animation": "2d",
        "oyun sinematiği": "game",
        "game cinematic": "game",
        "masalsı çizim": "storybook",
        "çizgi roman": "comic",
    }

    key = str(style or "3d").strip().lower()
    key = aliases.get(key, key)

    if key == "custom":
        return str(custom_style or "").strip() or STYLE_PRESETS["3d"]

    return STYLE_PRESETS.get(
        key,
        str(style).strip() or STYLE_PRESETS["3d"],
    )


def build_script_prompt(
    topic,
    notes="",
    target_seconds=60,
    language="Turkish",
    visual_style="3d",
    custom_style="",
):
    try:
        seconds = max(15, int(target_seconds))
    except (TypeError, ValueError):
        seconds = 60

    scene_count = max(8, min(12, round(seconds / 6)))
    style_description = resolve_visual_style(
        visual_style, custom_style
    )

    return f"""
You are an award-winning film director, screenwriter,
cinematographer and character performance director.

Create an original cinematic short film in {language}.

STORY TOPIC:
{topic}

ADDITIONAL CONTEXT:
{notes or "Use an original, coherent story."}

TARGET DURATION:
Approximately {seconds} seconds.

SCENE COUNT:
Exactly {scene_count} scenes.

VISUAL STYLE:
{style_description}

CHARACTER REQUIREMENTS:
- Create between 2 and 8 recurring characters.
- Choose the number of characters according to the story.
- Give every character a unique stable ID.
- Define age, appearance, hair, clothing, personality and voice style.
- Preserve each character's identity in every scene.
- Never randomly change faces, hairstyles or clothing.
- Characters must have distinct motivations and personalities.
- Do not make every character speak in the same way.

DIALOGUE REQUIREMENTS:
- This is a character-driven film, not a narrated slideshow.
- Dialogue must be written as separate ordered turns.
- Every dialogue turn must identify its speaker by character ID or name.
- Characters must respond to what other characters actually say.
- Include natural reactions, questions, disagreements and pauses.
- Use more than two speakers when the story benefits from it.
- Avoid repetitive exposition and artificial conversations.
- Narration is optional and must not replace character interaction.
- Do not invent dialogue for characters who are not present.

SHOT REQUIREMENTS:
- Plan visual shots for each scene.
- Use wide shots to establish locations and group positions.
- Use close-ups for important speaking characters.
- Use reaction shots for listening characters.
- Use over-the-shoulder shots during conversations.
- Use tracking and dolly shots when characters move.
- Specify which characters are visible in each shot.
- Do not describe a static image as if it were real motion.

Return ONLY valid JSON with this structure:

{{
  "title": "Film title",
  "style": "{style_description}",
  "character_bible": "Detailed continuity description",
  "characters": [
    {{
      "id": "character_1",
      "name": "Character name",
      "description": "Fixed physical appearance and clothing",
      "personality": "Personality and motivation",
      "voice": "Distinct voice characteristics"
    }}
  ],
  "scenes": [
    {{
      "scene_id": "scene_01",
      "narration": "",
      "visual_subject": "Visible characters and subjects",
      "action": "Specific physical actions",
      "location": "Detailed location",
      "props": "Important objects",
      "camera": "Camera framing and movement",
      "image_prompt": "Detailed first-frame visual prompt",
      "motion_prompt": "Specific continuous physical movement",
      "sound": "Ambience and physical sound effects",
      "dialogue_turns": [
        {{
          "speaker": "character_1",
          "text": "Dialogue in {language}",
          "emotion": "concerned",
          "delivery": "natural",
          "listener_ids": ["character_2"]
        }}
      ],
      "shot_plan": [
        {{
          "shot_type": "wide",
          "visible_characters": ["character_1", "character_2"],
          "focus_character": "character_1",
          "camera": "Slow cinematic tracking shot",
          "action": "Characters look toward each other"
        }}
      ],
      "transition": "Transition description"
    }}
  ]
}}

PRODUCTION RULES:
- Use vertical 9:16 composition.
- Create a strong opening and satisfying conclusion.
- Every scene must advance the story.
- Every scene needs a visible subject and physical action.
- Preserve character appearance and screen geography.
- Use natural dialogue in {language}.
- Use varied cinematic compositions.
- Avoid repeated or generic filler scenes.
- Do not use copyrighted existing characters.
- Do not invent facts about real people or history.
- Return valid JSON only, without Markdown.
""".strip()


def enrich_image_prompt(style, character_bible, scene):
    style_description = resolve_visual_style(style)

    parts = [
        f"Mandatory visual style: {style_description}.",
        (
            f"Character continuity: {character_bible}."
            if character_bible else ""
        ),
        f"Visible subject: {scene.get('visual_subject', '')}.",
        f"Physical action: {scene.get('action', '')}.",
        f"Location: {scene.get('location', '')}.",
        f"Props: {scene.get('props', '')}.",
        f"Camera: {scene.get('camera', '')}.",
        str(scene.get("image_prompt", "")),
        "Vertical 9:16 cinematic composition.",
        "Preserve recurring characters' faces and clothing.",
        "No text, subtitles, logos or watermarks.",
    ]

    return " ".join(str(part) for part in parts if part)


def validate_script(data, target_seconds=60):
    errors = []
    warnings = []

    if not isinstance(data, dict):
        return {
            "valid": False,
            "errors": ["Invalid JSON object."],
            "warnings": [],
        }

    scenes = data.get("scenes")

    if not isinstance(scenes, list):
        return {
            "valid": False,
            "errors": ["Scenes must be a list."],
            "warnings": [],
        }

    if not data.get("title"):
        errors.append("Missing title.")

    if not 8 <= len(scenes) <= 12:
        warnings.append("Expected between 8 and 12 scenes.")

    characters = data.get("characters")
    character_ids = set()
    new_format = characters is not None

    if new_format:
        if not isinstance(characters, list) or not 2 <= len(characters) <= 8:
            errors.append("Cast must contain between 2 and 8 characters.")
        else:
            for character in characters:
                if not isinstance(character, dict):
                    errors.append("Invalid character entry.")
                    continue

                character_id = str(character.get("id", "")).strip()
                name = str(character.get("name", "")).strip()

                if not character_id or not name:
                    errors.append("Every character needs an ID and name.")
                    continue

                if character_id in character_ids:
                    errors.append(
                        f"Duplicate character ID: {character_id}"
                    )

                character_ids.add(character_id)

    image_prompts = []
    narration_parts = []
    total_dialogue_turns = 0

    for number, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            errors.append(f"Scene {number} is invalid.")
            continue

        narration = str(scene.get("narration", "")).strip()
        narration_parts.append(narration)

        for field in ("image_prompt", "motion_prompt", "sound"):
            if not str(scene.get(field, "")).strip():
                errors.append(f"Scene {number}: missing {field}.")

        turns = scene.get("dialogue_turns", [])

        if new_format:
            if not isinstance(turns, list):
                errors.append(
                    f"Scene {number}: dialogue_turns must be a list."
                )
                turns = []

            if not narration and not turns:
                errors.append(
                    f"Scene {number}: needs narration or dialogue."
                )

            for turn_number, turn in enumerate(turns, 1):
                if not isinstance(turn, dict):
                    errors.append(
                        f"Scene {number}: invalid dialogue turn."
                    )
                    continue

                speaker = str(turn.get("speaker", "")).strip()
                dialogue_text = str(turn.get("text", "")).strip()

                if speaker not in character_ids:
                    errors.append(
                        f"Scene {number}, turn {turn_number}: "
                        f"unknown speaker {speaker}."
                    )

                if not dialogue_text:
                    errors.append(
                        f"Scene {number}, turn {turn_number}: empty text."
                    )

                total_dialogue_turns += 1

        image_prompt = str(scene.get("image_prompt", "")).strip()

        if len(image_prompt) < 40:
            warnings.append(
                f"Scene {number}: visual description may be too generic."
            )

        if image_prompt:
            image_prompts.append(image_prompt.lower())

    if new_format and total_dialogue_turns == 0:
        errors.append("Character-driven script contains no dialogue.")

    if len(image_prompts) != len(set(image_prompts)):
        warnings.append("Some scenes have duplicate image prompts.")

    narration = " ".join(narration_parts)
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
        "narration_word_count": word_count,
        "character_count": len(character_ids),
        "dialogue_turn_count": total_dialogue_turns,
    }

