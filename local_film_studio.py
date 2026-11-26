#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the film studio with local Qwen and incremental scene generation."""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from typing import Any


def _load_model():
    model_path = os.environ.get("LOCAL_QWEN_MODEL_PATH", "").strip()
    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError(
            "LOCAL_QWEN_MODEL_PATH geçerli bir GGUF dosyasını göstermiyor."
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python kurulu değil. Yerel model çalışma zamanı yüklenemedi."
        ) from exc

    return Llama(
        model_path=model_path,
        n_ctx=int(os.environ.get("LOCAL_QWEN_CONTEXT", "8192")),
        n_threads=max(2, min(os.cpu_count() or 4, 6)),
        n_batch=256,
        n_gpu_layers=0,
        verbose=False,
    )


@lru_cache(maxsize=1)
def _model():
    return _load_model()


def local_qwen_generate(prompt: str) -> str:
    """Generate one bounded JSON object using the local GGUF model."""
    response = _model().create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional screenwriter and film director. "
                    "Return exactly one complete valid JSON object. No Markdown. "
                    "Do not add keys or content not requested. /no_think"
                ),
            },
            {
                "role": "user",
                "content": (
                    prompt
                    + "\n\nReturn JSON only. Ensure all arrays are JSON arrays. /no_think"
                ),
            },
        ],
        temperature=0.35,
        top_p=0.85,
        max_tokens=max(
            512,
            int(os.environ.get("LOCAL_QWEN_MAX_TOKENS", "3600")),
        ),
        response_format={"type": "json_object"},
    )

    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Yerel Qwen modelinden yanıt alınamadı.")

    content = (choices[0].get("message") or {}).get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Yerel Qwen boş yanıt döndürdü.")

    return content.strip()


def _decode_json_container(value: Any, field: str) -> Any:
    if isinstance(value, str):
        candidate = value.strip()

        if candidate.startswith(("[", "{")):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    return value


def _as_object_list(value: Any, field: str) -> list:
    value = _decode_json_container(value, field)

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        for key in (field, "items", "data", "list", "results"):
            nested = value.get(key)

            if isinstance(nested, list):
                return nested

        values = list(value.values())

        if values and all(isinstance(item, dict) for item in values):
            return values

    raise ValueError(
        f"{field} alanı listeye dönüştürülemedi; "
        f"gelen tür: {type(value).__name__}."
    )


def _normalize_script_shape(script: Any) -> dict:
    if not isinstance(script, dict):
        raise ValueError("Modelin ürettiği ana JSON değeri nesne olmalı.")

    source = dict(script)

    for wrapper_key in ("film_project", "project", "film", "output"):
        nested = source.get(wrapper_key)

        if isinstance(nested, dict) and (
            "characters" in nested or "scenes" in nested
        ):
            source.update(nested)
            break

    for field in ("characters", "scenes"):
        if field not in source:
            raise ValueError(
                f"Model çıktısında zorunlu '{field}' alanı yok."
            )

        source[field] = _as_object_list(source[field], field)

    return source


def _json_call(
    generator,
    prompt: str,
    token_limit: int,
    label: str,
    attempts: int = 2,
) -> dict:
    last_error = None
    old_limit = os.environ.get("LOCAL_QWEN_MAX_TOKENS")

    try:
        os.environ["LOCAL_QWEN_MAX_TOKENS"] = str(token_limit)

        for attempt in range(1, attempts + 1):
            try:
                print(
                    f"[LOCAL-QWEN] {label}: üretim "
                    f"{attempt}/{attempts} "
                    f"(max_tokens={token_limit})",
                    flush=True,
                )

                raw, provider = generator.generate_text(prompt)
                parsed = generator.extract_json(raw)

                if not isinstance(parsed, dict):
                    raise ValueError("Model yanıtı JSON nesnesi değil.")

                print(
                    f"[LOCAL-QWEN] {label}: JSON alındı ({provider}).",
                    flush=True,
                )

                return parsed

            except Exception as exc:
                last_error = exc

                print(
                    f"[LOCAL-QWEN] {label}: deneme {attempt} başarısız: "
                    f"{type(exc).__name__}: {str(exc)[:350]}",
                    flush=True,
                )

        raise RuntimeError(f"{label} üretilemedi: {last_error}")

    finally:
        if old_limit is None:
            os.environ.pop("LOCAL_QWEN_MAX_TOKENS", None)
        else:
            os.environ["LOCAL_QWEN_MAX_TOKENS"] = old_limit


def _metadata_prompt(
    topic: str,
    style: str,
    language: str,
    seconds: int,
    scene_count: int,
) -> str:
    return f"""Create the planning foundation for an original cinematic short film.

TOPIC: {topic}
LANGUAGE: {language}
VISUAL STYLE: {style}
TARGET DURATION: {seconds} seconds
TOTAL SCENES TO BE GENERATED LATER: {scene_count}

Return one JSON object with exactly these top-level keys:

{{
  "title": "short film title",
  "description": "short coherent synopsis with beginning, development and ending",
  "style": "visual style",
  "character_bible": "continuity rules for faces, hair, clothing, age appearance and proportions",
  "tags": ["tag"],
  "characters": [
    {{
      "id": "character_01",
      "name": "name",
      "description": "distinct fixed visual appearance and clothing",
      "personality": "motivation and personality",
      "voice": "distinct voice description",
      "voice_gender": "male or female",
      "voice_profile": {{
        "xai": "",
        "gemini": "",
        "edge": ""
      }}
    }},
    {{
      "id": "character_02",
      "name": "name",
      "description": "distinct fixed visual appearance and clothing",
      "personality": "motivation and personality",
      "voice": "distinct voice description",
      "voice_gender": "male or female",
      "voice_profile": {{
        "xai": "",
        "gemini": "",
        "edge": ""
      }}
    }}
  ]
}}

Rules:
- Create 2 to 5 memorable recurring characters, never more than 8.
- Give every character a unique stable ID.
- Plan a clear story arc that can be told in exactly {scene_count} scenes.
- DO NOT generate any scenes now.
- Character designs must be easy to reproduce consistently.
- Use {language} for all text.
- Keep action cinematic and non-graphic.
- Do not copy existing films or copyrighted characters.
- Return valid JSON only."""


def _scene_prompt(
    topic: str,
    style: str,
    language: str,
    seconds: int,
    scene_count: int,
    characters: list,
    synopsis: str,
    first: int,
    last: int,
) -> str:
    character_data = [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "description": c.get("description"),
            "personality": c.get("personality"),
        }
        for c in characters
    ]

    return f"""Write ONLY scenes {first} through {last} of an original short film.

TOPIC: {topic}
SYNOPSIS / STORY ARC: {synopsis}
VISUAL STYLE: {style}
LANGUAGE: {language}
TOTAL FILM LENGTH: {seconds} seconds
TOTAL SCENES IN FILM: {scene_count}
SCENE NUMBERS REQUIRED IN THIS BATCH: {list(range(first, last + 1))}

RECURRING CHARACTER CANON
Use these exact IDs and preserve their appearance:

{json.dumps(character_data, ensure_ascii=False)}

Return one JSON object with a single top-level key "scenes".
Its value must be an array containing EXACTLY {last-first+1} scene objects.

Each scene object must use this structure:

{{
  "scene_id": "scene_{first:02d}",
  "narration": "brief narration or empty string",
  "visual_subject": "who or what is visible",
  "action": "clear visible action",
  "location": "specific location",
  "props": "important visible objects",
  "camera": "framing, lens feel and camera movement",
  "image_prompt": "detailed first-frame cinematic image prompt",
  "motion_prompt": "clear continuous motion for the shot",
  "sound": "sound design description",
  "sound_effects": [],
  "ambience": "environmental sound",
  "music": "music mood",
  "dialogue_turns": [
    {{
      "speaker": "character_01",
      "text": "natural dialogue",
      "emotion": "emotion",
      "delivery": "delivery",
      "facial_expression": "expression",
      "mouth_action": "speaking mouth movement",
      "gesture": "specific gesture",
      "body_language": "body language",
      "listener_reaction": "reaction",
      "listener_ids": ["character_02"]
    }}
  ],
  "shot_plan": [],
  "transition": "transition to next scene"
}}

MANDATORY RULES:

1. Produce exactly the requested number of scenes.
2. Use sequential IDs scene_{first:02d} onward.
3. Every scene needs non-empty image_prompt, motion_prompt and sound.
4. Every scene needs narration or at least one dialogue turn.
5. At least one scene in the complete film must contain character dialogue.
6. Include dialogue in this batch if it is scene 1.
7. Dialogue speaker and listener IDs MUST exactly match the character IDs above.
8. Do not invent characters.
9. Keep each scene purposeful and connected to the synopsis.
10. Vary shot sizes and camera movement.
11. Maintain screen geography and visual continuity.
12. Write all narration and dialogue in {language}.
13. Avoid graphic violence.
14. Use concise but production-ready descriptions so the JSON fits the output limit.
15. Return JSON only. No Markdown, no commentary, no extra keys."""


def _validate_scene_batch(
    scenes: list,
    first: int,
    last: int,
    character_ids: set,
) -> list:
    if len(scenes) != last - first + 1:
        raise ValueError(
            f"Bu parçada {last-first+1} sahne gerekiyordu; "
            f"{len(scenes)} geldi."
        )

    cleaned = []

    for offset, item in enumerate(scenes):
        if not isinstance(item, dict):
            raise ValueError(f"{first+offset}. sahne nesne değil.")

        expected_id = f"scene_{first+offset:02d}"
        item = dict(item)
        item["scene_id"] = expected_id

        for field in ("image_prompt", "motion_prompt", "sound"):
            if not str(item.get(field) or "").strip():
                raise ValueError(f"{expected_id}: {field} boş.")

        turns = item.get("dialogue_turns") or []

        if not isinstance(turns, list):
            raise ValueError(
                f"{expected_id}: dialogue_turns liste değil."
            )

        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError(
                    f"{expected_id}: geçersiz diyalog nesnesi."
                )

            speaker = str(turn.get("speaker") or "").strip()

            if speaker not in character_ids:
                raise ValueError(
                    f"{expected_id}: bilinmeyen konuşmacı ID: {speaker}"
                )

            if not str(turn.get("text") or "").strip():
                raise ValueError(f"{expected_id}: diyalog metni boş.")

            listeners = turn.get("listener_ids") or []

            if (
                not isinstance(listeners, list)
                or any(x not in character_ids for x in listeners)
            ):
                raise ValueError(
                    f"{expected_id}: listener_ids geçersiz."
                )

        if (
            not str(item.get("narration") or "").strip()
            and not turns
        ):
            raise ValueError(
                f"{expected_id}: anlatım veya diyalog gerekli."
            )

        cleaned.append(item)

    if first == 1 and not any(
        scene.get("dialogue_turns") for scene in cleaned
    ):
        raise ValueError(
            "İlk sahne grubunda en az bir karakter diyaloğu bulunmalı."
        )

    return cleaned


def _force_free_settings(script: dict) -> dict:
    script["cost_policy"] = {
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

    audio = script.get("audio")

    if not isinstance(audio, dict):
        audio = {}
        script["audio"] = audio

    audio["enabled"] = True
    audio["narration_enabled"] = True
    audio["tts_provider_order"] = ["gemini", "edge", "gtts"]

    audio.setdefault(
        "subtitles",
        {
            "enabled": True,
            "format": "srt",
            "burn_in": False,
        },
    )

    for section in ("music", "ambience", "sound_effects"):
        settings = audio.get(section)

        if not isinstance(settings, dict):
            settings = {}
            audio[section] = settings

        settings["enabled"] = False
        settings["provider"] = "licensed_library"

    return script


def _generate_project_in_batches(
    topic: str,
    style: str = "cinematic realistic",
) -> dict:
    import topic_project_generator as generator

    topic = str(topic or "").strip()

    if not topic:
        raise ValueError("Film konusu boş olamaz.")

    language = (
        os.environ.get("VIDEO_LANGUAGE", "Turkish").strip()
        or "Turkish"
    )

    try:
        seconds = int(os.environ.get("TARGET_SECONDS", "60"))
    except (TypeError, ValueError):
        seconds = 60

    seconds = max(15, min(seconds, 180))
    style = str(style or "cinematic realistic").strip() or "cinematic realistic"

    # Wan VACE: 81 kare / 16 FPS = yaklaşık 5,06 saniye/sahne.
    # İstenen süreyi karşılamak için gereken sahne sayısını yukarı yuvarla.
    scene_count = max(8, min(15, (seconds * 16 + 80) // 81))

    print(
        f"[LOCAL-QWEN] Aşamalı üretim başladı: "
        f"{scene_count} sahne hedefleniyor.",
        flush=True,
    )

    # AŞAMA 1: Film bilgileri ve karakterler.
    metadata = _json_call(
        generator,
        _metadata_prompt(
            topic,
            style,
            language,
            seconds,
            scene_count,
        ),
        token_limit=2400,
        label="Film bilgileri ve karakterler",
        attempts=2,
    )

    characters = _as_object_list(
        metadata.get("characters"),
        "characters",
    )

    if not 2 <= len(characters) <= 8:
        raise ValueError(
            f"Karakter sayısı 2-8 arasında olmalı; gelen: {len(characters)}"
        )

    character_ids = set()

    for index, character in enumerate(characters, 1):
        if (
            not isinstance(character, dict)
            or not str(character.get("name") or "").strip()
        ):
            raise ValueError(
                f"Karakter {index} geçersiz veya isimsiz."
            )

        character.setdefault("id", f"character_{index:02d}")
        character["id"] = str(character["id"]).strip()

        if (
            not character["id"]
            or character["id"] in character_ids
        ):
            raise ValueError(
                f"Karakter ID boş veya tekrar ediyor: {character['id']}"
            )

        character_ids.add(character["id"])

        character.setdefault(
            "voice_profile",
            {
                "xai": "",
                "gemini": "",
                "edge": "",
            },
        )

    # AŞAMA 2: Sahneleri ikişer ikişer üret.
    scenes = []
    batch_size = 2

    for first in range(1, scene_count + 1, batch_size):
        last = min(scene_count, first + batch_size - 1)

        prompt = _scene_prompt(
            topic,
            style,
            language,
            seconds,
            scene_count,
            characters,
            str(metadata.get("description") or ""),
            first,
            last,
        )

        batch = None
        last_error = None

        for attempt in range(1, 3):
            try:
                print(
                    f"[LOCAL-QWEN] Sahne grubu "
                    f"{first}-{last}/{scene_count} başlıyor.",
                    flush=True,
                )

                parsed = _json_call(
                    generator,
                    prompt,
                    token_limit=3600,
                    label=f"Sahneler {first}-{last}",
                    attempts=1,
                )

                batch = _validate_scene_batch(
                    _as_object_list(
                        parsed.get("scenes"),
                        "scenes",
                    ),
                    first,
                    last,
                    character_ids,
                )

                break

            except Exception as exc:
                last_error = exc

                print(
                    f"[LOCAL-QWEN] Sahne grubu {first}-{last} "
                    f"kontrolü başarısız ({attempt}/2): "
                    f"{type(exc).__name__}: {str(exc)[:300]}",
                    flush=True,
                )

                if attempt == 1:
                    prompt += (
                        "\n\nCORRECTION: The previous batch failed validation. "
                        "Return exactly the requested scene objects. "
                        "Keep all IDs valid. Every scene must have narration "
                        "or dialogue, image_prompt, motion_prompt and sound."
                    )

        if batch is None:
            raise RuntimeError(
                f"Sahne grubu {first}-{last} üretilemedi: {last_error}"
            )

        scenes.extend(batch)

        print(
            f"[LOCAL-QWEN] İlerleme: "
            f"{len(scenes)}/{scene_count} sahne tamamlandı.",
            flush=True,
        )

    # AŞAMA 3: Bütün parçaları birleştir ve mevcut doğrulayıcıya gönder.
    script = dict(metadata)
    script["characters"] = characters
    script["scenes"] = scenes

    _force_free_settings(script)
    script = _normalize_script_shape(script)

    project = generator.build_film_project(
        script,
        topic=topic,
        target_seconds=seconds,
        language=language,
        visual_style=style,
    )

    print(
        "[LOCAL-QWEN] Proje doğrulandı: "
        f"{project['film']['title']} "
        f"| sahne={len(project['scenes'])} "
        f"| karakter={len(project['characters'])} "
        "| maliyet=free_only",
        flush=True,
    )

    return project


def _install_project_compatibility():
    """
    Convert the legacy film JSON into the canonical pipeline schema
    before ProductionJob validates and stores the project.
    """
    import production_orchestrator
    from project_adapter import legacy_to_pipeline_project

    original_create = (
        production_orchestrator.ProductionJob.create.__func__
    )

    def compatible_create(cls, workspace, project_data):
        if (
            isinstance(project_data, dict)
            and isinstance(project_data.get("film"), dict)
            and isinstance(project_data.get("characters"), list)
            and isinstance(project_data.get("scenes"), list)
        ):
            print(
                "[LOCAL-QWEN] Eski proje biçimi üretim şemasına "
                "dönüştürülüyor.",
                flush=True,
            )

            project_data = legacy_to_pipeline_project(
                project_data
            ).to_dict()

            print(
                "[LOCAL-QWEN] Proje uyumluluk dönüşümü tamamlandı.",
                flush=True,
            )

        return original_create(
            cls,
            workspace,
            project_data,
        )

    production_orchestrator.ProductionJob.create = classmethod(
        compatible_create
    )


def main() -> int:
    import topic_project_generator as generator
    import film_studio

    # Üretim takip sistemine gönderilen eski proje biçimini dönüştür.
    _install_project_compatibility()

    generator.PROVIDERS["local_qwen"] = local_qwen_generate
    generator.provider_order = lambda: ["local_qwen"]
    generator.generate_project = _generate_project_in_batches

    film_studio.configure_free_llm_routes = lambda: ["local_qwen"]

    os.environ["LLM_PROVIDER_ORDER"] = "local_qwen"
    os.environ.setdefault("LOCAL_QWEN_MAX_TOKENS", "3600")

    return film_studio.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"LOCAL_FILM_STUDIO_ERROR={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
