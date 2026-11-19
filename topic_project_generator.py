
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate a complete validated film project from topic-only input."""

import json
import os
import re
import urllib.error
import urllib.request

from film_project import build_film_project


def log(message):
    print(f"[LLM] {message}", flush=True)


def request_json(url, headers, payload, timeout=180):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        details = exc.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code}: {details[:1200]}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Bağlantı hatası: {type(exc).__name__}: {exc}"
        ) from exc


def extract_openai_text(data):
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Model yanıtında choices bulunamadı.")

    message = choices[0].get("message") or {}
    content = message.get("content", "")

    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )

    content = str(content or "").strip()

    if not content:
        raise ValueError("Model boş yanıt döndürdü.")

    return content


def call_openrouter(prompt):
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY tanımlı değil.")

    model = os.environ.get(
        "OPENROUTER_MODEL",
        "openrouter/free",
    ).strip()

    data = request_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",
            "X-OpenRouter-Title": "Cihan Cinematic Pipeline",
        },
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional screenwriter, film director "
                        "and cinematic continuity supervisor. Return only "
                        "valid JSON. Never use Markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 12000,
        },
    )

    return extract_openai_text(data)


def call_groq(prompt):
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY tanımlı değil.")

    model = os.environ.get(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    ).strip()

    data = request_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional screenwriter, film director "
                        "and cinematic continuity supervisor. Return only "
                        "valid JSON. Never use Markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 12000,
        },
    )

    return extract_openai_text(data)


def call_gemini(prompt):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY tanımlı değil.")

    model = os.environ.get(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    ).strip()

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent"
    )

    data = request_json(
        url,
        {
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 12000,
                "responseMimeType": "application/json",
            },
        },
    )

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini yanıtında aday bulunamadı.")

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )

    result = "\n".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict)
    ).strip()

    if not result:
        raise ValueError("Gemini boş yanıt döndürdü.")

    return result


PROVIDERS = {
    "openrouter": call_openrouter,
    "groq": call_groq,
    "gemini": call_gemini,
}


def provider_order():
    raw = os.environ.get(
        "LLM_PROVIDER_ORDER",
        "openrouter,groq,gemini",
    )

    result = []
    for item in raw.split(","):
        name = item.strip().lower()
        if name in PROVIDERS and name not in result:
            result.append(name)

    return result


def generate_text(prompt):
    errors = []

    for provider in provider_order():
        try:
            log(f"{provider} deneniyor...")
            result = PROVIDERS[provider](prompt)

            if not result or not result.strip():
                raise ValueError("Sağlayıcı boş yanıt döndürdü.")

            log(f"Başarılı sağlayıcı: {provider}")
            return result, provider

        except Exception as exc:
            message = str(exc)
            errors.append(
                f"{provider}: {message[:500]}"
            )
            log(f"{provider} başarısız; sonraki sağlayıcı deneniyor.")
            log(message[:300])

    raise RuntimeError(
        "Hiçbir yapay zekâ sağlayıcısı yanıt veremedi.\n"
        + "\n".join(errors)
    )


def extract_json(raw):
    raw = str(raw or "").strip()

    raw = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"\s*```\s*$", "", raw)

    start = raw.find("{")
    if start < 0:
        raise ValueError("Model yanıtında JSON nesnesi bulunamadı.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(raw)):
        char = raw[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = raw[start:index + 1]
                return json.loads(candidate)

    raise ValueError("Model yanıtındaki JSON tamamlanmamış.")


def build_prompt(topic, style, language, seconds):
    scene_count = max(8, min(15, round(seconds / 6)))

    template = {
        "title": "Film title",
        "description": "Film description",
        "style": style,
        "character_bible": "Detailed visual continuity guide",
        "tags": [],
        "characters": [
            {
                "id": "character_1",
                "name": "Character name",
                "description": "Detailed fixed appearance, face, hair and clothing",
                "personality": "Personality and motivation",
                "voice": "Distinct voice description",
                "voice_gender": "male",
                "voice_profile": {
                    "xai": "",
                    "gemini": "",
                    "edge": "",
                },
            },
            {
                "id": "character_2",
                "name": "Second character",
                "description": "Detailed fixed appearance and clothing",
                "personality": "Personality and motivation",
                "voice": "Distinct voice description",
                "voice_gender": "female",
                "voice_profile": {
                    "xai": "",
                    "gemini": "",
                    "edge": "",
                },
            },
        ],
        "scenes": [
            {
                "scene_id": "scene_01",
                "narration": "",
                "visual_subject": "Visible subjects",
                "action": "Specific physical action",
                "location": "Detailed location",
                "props": "Important objects",
                "camera": "Shot framing and camera movement",
                "image_prompt": "Detailed cinematic first-frame image prompt",
                "motion_prompt": "Specific continuous physical movement",
                "sound": "Ambience and sound effects",
                "sound_effects": [],
                "ambience": "",
                "music": "",
                "dialogue_turns": [
                    {
                        "speaker": "character_1",
                        "text": "Natural dialogue in the requested language",
                        "emotion": "neutral",
                        "delivery": "natural",
                        "listener_ids": ["character_2"],
                    }
                ],
                "shot_plan": [],
                "transition": "Transition to next scene",
            }
        ],
        "audio": {
            "enabled": True,
            "tts_provider_order": [
                "gemini",
                "edge",
                "gtts",
                "xai",
            ],
            "narration_enabled": True,
            "subtitles": {
                "enabled": True,
                "format": "srt",
                "burn_in": False,
            },
            "music": {
                "enabled": False,
                "provider": "licensed_library",
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
                "volume": 0.45,
            },
        },
        "cost_policy": {
            "mode": "free_only",
            "paid_services_allowed": False,
            "require_approval": True,
            "show_cost_before_run": True,
            "block_unknown_prices": True,
            "currency": "USD",
            "max_project_budget": 0.0,
            "record_estimated_costs": True,
            "record_actual_costs": True,
        },
    }

    return f"""
You are an award-winning professional film director,
screenwriter, cinematographer and continuity supervisor.

Create a complete original cinematic short film.

FILM TOPIC:
{topic}

LANGUAGE:
{language}

VISUAL STYLE:
{style}

TARGET DURATION:
{seconds} seconds.

EXACT SCENE COUNT:
{scene_count}

Return exactly one valid JSON object matching this structure.
The example arrays show the required object fields; expand them
to the required number of characters and scenes.

STRUCTURE:
{json.dumps(template, ensure_ascii=False, indent=2)}

MANDATORY RULES:

1. Create 2 to 8 recurring characters.
2. Create exactly {scene_count} scenes.
3. Every character must have a unique ID and name.
4. Character faces, hair, clothing, age appearance and proportions
   must remain visually consistent across all scenes.
5. Every scene must contain image_prompt, motion_prompt and sound.
6. Every scene must contain narration or dialogue.
7. Include meaningful character dialogue in at least one scene.
8. Every dialogue speaker must match an existing character ID.
9. Use the requested language for narration and dialogue.
10. Each scene must advance the story.
11. Use cinematic shot variety and coherent screen geography.
12. Specify camera framing, movement, lighting and atmosphere.
13. Opening must engage the viewer; ending must feel complete.
14. Compose every scene for landscape 26:15 framing.
15. Do not create generic filler scenes.
16. Do not copy existing films, scripts or copyrighted characters.
17. Avoid graphic violence and explicit sexual content.
18. Return JSON only. No Markdown or explanations.

AUDIO AND COST RULES:

19. Keep the audio configuration in the structure.
20. Use this default TTS order: Gemini, Edge TTS, gTTS, then xAI.
21. xAI is a paid optional provider. Never make it the default.
22. Do not enable paid music or sound-effect generation.
23. Do not enable FAL or any other paid audio-generation service.
24. Use free or properly licensed audio assets only when available.
25. Keep cost_policy.mode as "free_only".
26. Keep paid_services_allowed as false.
27. Keep max_project_budget as 0.
28. Never change cost settings to enable a paid service.
29. Leave provider voice IDs empty unless a valid voice ID is known.
30. Do not invent prices, quotas, API keys or service availability.

The JSON must be complete and valid. Do not truncate it.
""".strip()


def generate_project(topic, style="cinematic realistic"):
    topic = str(topic or "").strip()

    if not topic:
        raise ValueError("Film konusu boş olamaz.")

    language = (
        os.environ.get("VIDEO_LANGUAGE", "Turkish").strip()
        or "Turkish"
    )

    try:
        seconds = int(
            os.environ.get("TARGET_SECONDS", "60")
        )
    except (TypeError, ValueError):
        seconds = 60

    seconds = max(15, min(seconds, 180))
    style = str(style or "cinematic realistic").strip()

    if not style:
        style = "cinematic realistic"

    original_prompt = build_prompt(
        topic,
        style,
        language,
        seconds,
    )

    current_prompt = original_prompt
    last_error = None
    last_raw = ""
    selected_provider = "unknown"

    # İlk üretim + en fazla iki otomatik düzeltme.
    for attempt in range(1, 4):
        log(f"Film projesi üretimi: deneme {attempt}/3")

        try:
            raw, provider = generate_text(current_prompt)
            last_raw = raw
            selected_provider = provider

            script = extract_json(raw)

            # Model çıktısı maliyet politikasını değiştiremez.
            # Proje daima ücretsiz modda başlar.
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

            # Ücretli müzik ve efekt servisleri model tarafından
            # etkinleştirilmiş olsa bile varsayılan olarak kapatılır.
            if isinstance(script.get("audio"), dict):
                audio = script["audio"]

                audio["tts_provider_order"] = [
                    "gemini",
                    "edge",
                    "gtts",
                    "xai",
                ]

                for section in ("music", "sound_effects"):
                    settings = audio.get(section)
                    if isinstance(settings, dict):
                        settings["enabled"] = False
                        settings["provider"] = "licensed_library"

                ambience = audio.get("ambience")
                if isinstance(ambience, dict):
                    ambience["enabled"] = False
                    ambience["provider"] = "licensed_library"

            project = build_film_project(
                script,
                topic=topic,
                target_seconds=seconds,
                language=language,
                visual_style=style,
            )

            log(
                "Proje doğrulandı: "
                + str(project["film"]["title"])
                + " | sahne: "
                + str(len(project["scenes"]))
                + " | karakter: "
                + str(len(project["characters"]))
                + " | sağlayıcı: "
                + provider
                + " | maliyet modu: "
                + project["cost_policy"]["mode"]
            )

            return project

        except Exception as exc:
            last_error = exc
            log(
                f"Deneme {attempt} başarısız: "
                f"{type(exc).__name__}: {str(exc)[:700]}"
            )

            if attempt >= 3:
                break

            current_prompt = (
                original_prompt
                + "\n\nIMPORTANT: Your previous response failed "
                "JSON parsing or project validation.\n"
                + "Correct the complete project and return valid JSON only.\n"
                + "Do not change the free-only cost policy.\n"
                + "Do not enable paid services.\n"
                + "ERROR:\n"
                + str(exc)[:5000]
                + "\n\nPREVIOUS RESPONSE:\n"
                + last_raw[:16000]
            )

    raise RuntimeError(
        "Film projesi üç denemede de oluşturulamadı. "
        "Son hata: "
        + str(last_error)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a film project from a topic."
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument(
        "--style",
        default="cinematic realistic",
    )
    parser.add_argument(
        "--output",
        default=".work/project.json",
    )

    args = parser.parse_args()

    project = generate_project(
        args.topic,
        args.style,
    )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            project,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print("PROJECT_OUTPUT=" + output_path)
    print("PROJECT_TITLE=" + project["film"]["title"])
    print("SCENE_COUNT=" + str(len(project["scenes"])))
    print("CHARACTER_COUNT=" + str(len(project["characters"])))
    print("COST_MODE=" + project["cost_policy"]["mode"])
    print(
        "PAID_SERVICES_ALLOWED="
        + str(project["cost_policy"]["paid_services_allowed"])
    )
