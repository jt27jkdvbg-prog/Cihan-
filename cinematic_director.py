
"""Visual style presets and cinematic script validation."""

STYLE_PRESETS = {
    "3d": (
        "Premium stylized 3D animated feature-film aesthetic, "
        "expressive original characters, detailed 3D environments, "
        "cinematic global illumination, rich materials and smooth animation."
    ),
    "anime": (
        "Original Japanese anime feature-film aesthetic, expressive "
        "character animation, detailed backgrounds, dramatic composition, "
        "beautiful cinematic lighting and fluid motion."
    ),
    "realistic": (
        "High-end live-action cinematic realism, natural human movement, "
        "detailed environments, realistic lighting, film-grade color."
    ),
    "2d": (
        "Premium hand-drawn 2D animation, expressive character poses, "
        "beautiful layered backgrounds, fluid frame animation."
    ),
    "game": (
        "AAA video game cinematic, high-detail 3D characters and worlds, "
        "dramatic lighting, premium real-time-rendered cutscene quality."
    ),
    "storybook": (
        "Beautiful animated storybook aesthetic, painterly environments, "
        "rich colors, magical atmosphere and expressive characters."
    ),
    "comic": (
        "Cinematic graphic novel animation, dynamic compositions, "
        "illustrated textures, dramatic shadows and fluid motion."
    ),
}


def resolve_visual_style(style="3d", custom_style=""):
    aliases = {
        "3d animasyon": "3d",
        "3d animation": "3d",
        "anime": "anime",
        "gerçekçi sinema": "realistic",
        "realistic": "realistic",
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
        return custom_style.strip() or STYLE_PRESETS["3d"]

    return STYLE_PRESETS.get(key, str(style).strip() or STYLE_PRESETS["3d"])


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
    style_description = resolve_visual_style(visual_style, custom_style)

    return f"""
You are an expert film director, screenwriter and cinematographer.

Create a cinematic {language} short film about: {topic}
Additional context: {notes or "Use relevant and reliable information."}
Target duration: approximately {seconds} seconds.
Create exactly {scene_count} scenes.

MANDATORY VISUAL STYLE:
{style_description}

The selected visual style must remain consistent throughout the entire film.
Do not switch to photographs, unrelated art styles or generic stock imagery.

Return ONLY valid JSON:
{{
  "title": "Film title",
  "style": "{style_description}",
  "character_bible": "Detailed fixed descriptions of recurring characters",
  "scenes": [
    {{
      "narration": "Turkish narration",
      "dialogue": "Optional spoken dialogue",
      "visual_subject": "Exact visible subject",
      "action": "What the subject physically does",
      "location": "Specific environment",
      "props": "Important objects",
      "camera": "Camera framing and movement",
      "image_prompt": "Detailed first-frame visual description",
      "motion_prompt": "Specific character and environment movement",
      "sound": "Scene-specific ambience and sound effects",
      "transition": "Transition to next scene"
    }}
  ]
}}

PRODUCTION RULES:
- Vertical 9:16 cinematic composition.
- Strong opening hook and satisfying ending.
- Clear beginning, development and conclusion.
- Every image must directly illustrate its scene narration.
- Never use unrelated decorative images or generic filler scenes.
- Every scene must contain a visible subject and physical action.
- Do not repeat the same composition in every scene.
- Preserve recurring characters' faces, hair, clothing and proportions.
- Describe recurring characters consistently using character_bible.
- Use varied tracking, dolly, pan, tilt and cinematic camera shots.
- Natural Turkish narration and dialogue.
- Add scene-specific sound effects and ambience.
- Keep important subjects inside the vertical frame.
- Do not invent facts about real people or history.
- Use original character designs; do not copy existing franchises.

Return only valid JSON. Do not include Markdown.
""".strip()


def enrich_image_prompt(style, character_bible, scene):
    style_description = resolve_visual_style(style)

    parts = [
        f"Mandatory visual style: {style_description}.",
        f"Character continuity: {character_bible}." if character_bible else "",
        f"Visible subject: {scene.get('visual_subject', '')}.",
        f"Physical action: {scene.get('action', '')}.",
        f"Location: {scene.get('location', '')}.",
        f"Props: {scene.get('props', '')}.",
        f"Camera: {scene.get('camera', '')}.",
        scene.get("image_prompt", ""),
        "Vertical 9:16 composition.",
        "Preserve recurring characters' faces, clothing and identity.",
        "No text, letters, subtitles, logos or watermarks.",
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

    image_prompts = []

    for number, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            errors.append(f"Scene {number} is invalid.")
            continue

        for field in ("narration", "image_prompt", "motion_prompt", "sound"):
            value = str(scene.get(field, "")).strip()
            if not value:
                errors.append(f"Scene {number}: missing {field}.")

        image_prompt = str(scene.get("image_prompt", "")).strip()

        if len(image_prompt) < 40:
            warnings.append(
                f"Scene {number}: visual description may be too generic."
            )

        if image_prompt:
            image_prompts.append(image_prompt.lower())

    if len(image_prompts) != len(set(image_prompts)):
        warnings.append("Some scenes have duplicate image prompts.")

    narration = " ".join(
        str(scene.get("narration", ""))
        for scene in scenes
        if isinstance(scene, dict)
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
        "narration_word_count": word_count,
    }

