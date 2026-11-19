#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Topic-only film project generation with automatic LLM provider fallback."""

import json
import os
import re
import time
import urllib.error
import urllib.request

from film_project import build_film_project


def log(message):
    print(f"[LLM] {message}", flush=True)


def request_json(url, headers, payload, timeout=150):
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
        details = exc.read(1800).decode("utf-8", errors="replace")
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
                        "You are a professional screenwriter and film director. "
                        "Return only valid JSON. Never use Markdown fences."
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
                        "You are a professional screenwriter and film director. "
                        "Return only valid JSON. Never use Markdown fences."
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
        + ":generateContent?key="
        + key
    )

    data = request_json(
        url,
        {"Content-Type": "application/json"},
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


def generate_text(prompt):
    order = os.environ.get(
        "LLM_PROVIDER_ORDER",
        "openrouter,groq,gemini",
    )

    providers = [
        item.strip().lower()
        for item in order.split(",")
        if item.strip().lower() in PROVIDERS
    ]

    errors = []

    for provider in providers:
        try:
            log(f"{provider} deneniyor...")
            result = PROVIDERS[provider](prompt)
            log(f"Başarılı sağlayıcı: {provider}")
            return result, provider

        except Exception as exc:
            message = str(exc)
            errors.append(f"{provider}: {message[:500]}")
            log(f"{provider} başarısız; sıradaki sağlayıcıya geçiliyor.")
            log(message[:350])

    raise RuntimeError(
        "Hiçbir yapay zekâ sağlayıcısı yanıt veremedi.\n"
        + "\n".join(errors)
    )


def extract_json(raw):
    raw = str(raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("Model yanıtında JSON nesnesi bulunamadı.")

    return json.loads(raw[start:end + 1])


def build_prompt(topic, style, language, seconds):
    scene_count = max(8, min(15, round(seconds / 6)))

    return f"""
You are a professional film director, screenwriter,
cinematographer and character-continuity supervisor.

Create a complete, original short-film project.

TOPIC:
{topic}

LANGUAGE:
{language}

VISUAL STYLE:
{style}

TARGET DURATION:
{seconds} seconds.

SCENES:
Create exactly {scene_count} scenes.
Never create fewer than 8 or more than 15 scenes.

Return ONLY one valid JSON object with this structure:

{{
  "title": "Film title",
  "description": "Short film description",
  "style": "{style}",
  "character_bible": "Detailed visual continuity guide",
  "characters": [
    {{
      "id": "character_1",
      "name": "Character name",
      "description": "Detailed fixed appearance, face, hair and clothing",
      "personality": "Personality and motivation",
      "voice": "Distinct voice description",
      "voice_gender": "male"
    }}
  ],
  "scenes": [
    {{
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
      "dialogue_turns": [
        {{
          "speaker": "character_1",
          "text": "Natural dialogue in the requested language",
          "emotion": "neutral",
          "delivery": "natural",
          "listener_ids": ["character_2"]
        }}
      ],
      "shot_plan": [],
      "transition": "Transition to next scene"
    }}
  ]
}}

MANDATORY QUALITY RULES:

- Create 2 to 8 recurring characters.
- Every character must have a unique ID and name.
- Keep faces, hair, clothing and proportions consistent.
- Every scene needs image_prompt, motion_prompt and sound.
- Every scene needs narration or dialogue.
- Include meaningful character dialogue in at least one scene.
- Every dialogue speaker must use an existing character ID.
- Make every scene advance the story.
- Use cinematic shot variety, meaningful camera movement,
  clear physical actions and coherent screen geography.
- Make the opening engaging and the ending satisfying.
- Use landscape 26:15 composition for every scene.
- Do not create generic filler scenes.
- Do not copy existing films, scripts or copyrighted characters.
- Avoid graphic violence and explicit sexual content.
- Do not invent factual claims about real people or historical events.
- Return JSON only. No Markdown, no explanations.
""".strip()


def generate_project(topic, style="cinematic realistic"):
    topic = str(topic or "").strip()
    if not topic:
        raise ValueError("Film konusu boş olamaz.")

    language = os.environ.get("VIDEO_LANGUAGE", "Turkish").strip()
    try:
        seconds = int(os.environ.get("TARGET_SECONDS", "60"))
    except ValueError:
        seconds = 60

    seconds = max(15, min(seconds, 180))
    style = str(style or "cinematic realistic").strip()

    prompt = build_prompt(topic, style, language, seconds)

    raw, provider = generate_text(prompt)
    script = extract_json(raw)

    try:
        project = build_film_project(
            script,
            topic=topic,
            target_seconds=seconds,
            language=language,
            visual_style=style,
        )
    except Exception as first_error:
        log("İlk proje doğrulamadan geçmedi; bir kez düzeltiliyor.")

        repair_prompt = (
            prompt
            + "\n\nFIX THIS JSON. Correct every schema or validation "
            "problem listed below. Preserve the story and return the "
            "complete corrected JSON only.\n\n"
            + "VALIDATION ERROR:\n"
            + str(first_error)[:5000]
            + "\n\nINVALID JSON:\n"
            + json.dumps(script, ensure_ascii=False)[:18000]
        )

        repaired_raw, repair_provider = generate_text(repair_prompt)
        repaired_script = extract_json(repaired_raw)

        project = build_film_project(
            repaired_script,
            topic=topic,
            target_seconds=seconds,
            language=language,
            visual_style=style,
        )

        provider = repair_provider

    log(
        "Proje hazır: "
        + str(project["film"]["title"])
        + " | sahne: "
        + str(len(project["scenes"]))
        + " | karakter: "
        + str(len(project["characters"]))
        + " | sağlayıcı: "
        + provider
    )

    return project
