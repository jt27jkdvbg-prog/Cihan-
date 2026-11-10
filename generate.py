
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cinematic Video Generator
Araştırma -> Senaryo -> Görsel -> Hareket -> Diyalog/Ses -> Altyazı -> MP4
"""

import argparse
import base64
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cinematic_director import (
    build_script_prompt,
    enrich_image_prompt,
    resolve_visual_style,
    validate_script,
)
from dialogue_director import normalize_cast
from speech_director import build_voice_profiles
from camera_director import build_zoompan_filter
from quality_gate import require_real_motion, validate_motion_outputs


# ============================================================
# CONFIGURATION
# ============================================================

W, H, FPS = 1080, 1920, 30
PAD = 0.35
UTTERANCE_GAP = 0.12

def env(name, default):
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default

def env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(env(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

def env_float(name, default, minimum=None, maximum=None):
    try:
        value = float(env(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

def env_list(name, default):
    return [
        x.strip().lower()
        for x in env(name, default).split(",")
        if x.strip()
    ]

RESEARCH_MODELS = env_list(
    "GEMINI_RESEARCH_MODELS",
    "gemini-2.5-flash,gemini-2.0-flash",
)
SCRIPT_MODELS = env_list(
    "GEMINI_SCRIPT_MODELS",
    "gemini-2.5-flash,gemini-2.0-flash",
)
GEMINI_TIMEOUT_MS = env_int("GEMINI_TIMEOUT_SEC", 120, 20, 300) * 1000

IMAGE_ENGINE = env("IMAGE_ENGINE", "auto").lower()
CF_IMAGE_MODEL = env(
    "CF_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell",
)
CF_STEPS = env_int("CF_STEPS", 6, 1, 20)

MOTION = env("MOTION", "auto").lower()
MOTION_REQUIRED = env("MOTION_REQUIRED", "false").lower() == "true"

FAL_IMAGE_MODEL = env("FAL_IMAGE_MODEL", "fal-ai/flux/dev")
VIDEO_MODEL = env("VIDEO_MODEL", "creatify/boreal")
VIDEO_RES = env("VIDEO_RES", "720p")
AMBIENCE_VOLUME = env_float("AMBIENCE_VOLUME", 0.25, 0, 1)

TTS_ENGINES = env_list("TTS_ENGINES", "edge,gtts,gemini")
GEMINI_TTS_MODEL = env(
    "GEMINI_TTS_MODEL",
    "gemini-2.5-flash-preview-tts",
)
GEMINI_VOICE = env("GEMINI_VOICE", "Charon")
GEMINI_TTS_STYLE = env(
    "GEMINI_TTS_STYLE",
    "Speak naturally in Turkish with clear pronunciation and cinematic pacing.",
)
EDGE_VOICE = env("EDGE_VOICE", "tr-TR-AhmetNeural")
EDGE_FEMALE_VOICE = env("EDGE_FEMALE_VOICE", "tr-TR-EmelNeural")
EDGE_RATE = env("EDGE_RATE", "+0%")
GTTS_LANG = env("GTTS_LANG", "tr")

ELEVEN_MODEL = env("TTS_MODEL", "eleven_multilingual_v2")
ELEVEN_VOICE = env("ELEVEN_VOICE_ID", "")

TARGET_SECONDS = env_int("TARGET_SECONDS", 45, 15, 180)
VIDEO_DILI = env("VIDEO_DILI", "Turkish")
VISUAL_STYLE = env("VISUAL_STYLE", "3d")
CUSTOM_VISUAL_STYLE = env("CUSTOM_VISUAL_STYLE", "")

NEGATIVE = (
    "subtitles, text, letters, watermark, logo, distorted face, "
    "extra fingers, morphing, flicker, deformed hands, unnatural anatomy, "
    "duplicate people, inconsistent clothing"
)

UA = {"User-Agent": "cinematic-video-generator/3.0"}


# ============================================================
# GENERAL HELPERS
# ============================================================

def log(message):
    print(message, flush=True)


def run(command, cwd=None, timeout=1800):
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            "Komut başarısız: "
            + " ".join(map(str, command[:7]))
            + "\n"
            + (result.stderr or "")[-2500:]
        )
    return result


def retry(fn, label, tries=3, wait=5):
    last_error = None

    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            log(
                f"  ! {label}: deneme {attempt}/{tries} başarısız: "
                f"{str(exc)[:220]}"
            )
            if attempt < tries:
                time.sleep(wait * attempt)

    raise RuntimeError(f"{label} başarısız: {last_error}")


def check_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} sistemde bulunamadı.")


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_json(raw):
    raw = safe_text(raw)
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("Model yanıtında geçerli JSON bulunamadı.")

    return json.loads(raw[start:end + 1])


# ============================================================
# GEMINI
# ============================================================

def gemini_client():
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY tanımlı değil.")

    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )

    return client, types


def gemini_generate(
    prompt,
    models,
    search=False,
    json_mode=False,
    max_tokens=8192,
):
    client, types = gemini_client()
    last_error = None

    for model in models:
        for attempt in range(2):
            try:
                config = {
                    "temperature": 0.7,
                    "max_output_tokens": max_tokens,
                }

                if search:
                    config["tools"] = [
                        types.Tool(
                            google_search=types.GoogleSearch()
                        )
                    ]

                if json_mode:
                    config["response_mime_type"] = "application/json"

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config),
                )

                result = safe_text(response.text)

                if result:
                    log(f"     Model: {model}")
                    return result

                raise RuntimeError(f"{model} boş yanıt döndürdü.")

            except Exception as exc:
                last_error = exc
                message = str(exc).lower()

                temporary = any(
                    token in message
                    for token in (
                        "429",
                        "503",
                        "timeout",
                        "unavailable",
                        "resource_exhausted",
                        "temporarily",
                    )
                )

                log(f"     {model}: {str(exc)[:150]}")

                if temporary and attempt == 0:
                    time.sleep(8)
                else:
                    break

    raise RuntimeError(f"Gemini modelleri başarısız: {last_error}")


# ============================================================
# RESEARCH
# ============================================================

def wiki_context(topic, max_chars=3000):
    import requests

    sections = []

    for language in ("tr", "en"):
        api = f"https://{language}.wikipedia.org/w/api.php"

        try:
            response = requests.get(
                api,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": topic,
                    "format": "json",
                    "srlimit": 1,
                },
                headers=UA,
                timeout=25,
            )
            response.raise_for_status()

            results = response.json()["query"]["search"]

            if not results:
                continue

            title = results[0]["title"]

            page_response = requests.get(
                api,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exlimit": 1,
                    "titles": title,
                    "redirects": 1,
                    "format": "json",
                },
                headers=UA,
                timeout=25,
            )
            page_response.raise_for_status()

            pages = page_response.json()["query"]["pages"]
            page = next(iter(pages.values()))
            extract = safe_text(page.get("extract"))

            if extract:
                sections.append(
                    f"[Wikipedia/{language}: {title}]\n"
                    f"{extract[:max_chars]}"
                )

        except Exception as exc:
            log(
                f"     Wikipedia/{language} atlandı: "
                f"{str(exc)[:100]}"
            )

    return "\n\n".join(sections)


def research(topic):
    context = wiki_context(topic)

    prompt = f"""
Konu: {topic}

Bu konuda kısa film için kullanılabilecek güvenilir araştırma notları hazırla.
Önemli tarihleri, olayları, kişileri ve doğrulanabilir bilgileri belirt.
Emin olmadığın bilgileri gerçekmiş gibi sunma.
Kaynakları mümkünse bağlantılarıyla belirt.
Türkçe ve madde madde yaz.

Ek bağlam:
{context or "Ek kaynak bulunamadı."}
"""

    try:
        notes = gemini_generate(
            prompt,
            RESEARCH_MODELS,
            search=True,
            max_tokens=6000,
        )
    except Exception as exc:
        log(f"     Web araması kullanılamadı: {str(exc)[:120]}")
        notes = gemini_generate(
            prompt,
            SCRIPT_MODELS,
            search=False,
            max_tokens=6000,
        )

    if len(notes) < 150:
        raise RuntimeError("Araştırma metni yeterince uzun değil.")

    return notes


# ============================================================
# SCRIPT AND CHARACTER NORMALIZATION
# ============================================================

def normalize_script_speakers(data):
    characters = data.get("characters", [])

    if not isinstance(characters, list):
        raise ValueError("Karakter listesi geçersiz.")

    cast = normalize_cast(characters)

    if not cast:
        raise ValueError("Senaryoda geçerli karakter bulunamadı.")

    by_id = {}
    by_name = {}

    for character in cast:
        cid = safe_text(character.get("id"))
        name = safe_text(character.get("name"))

        if not cid or not name:
            raise ValueError("Her karakterin ID ve adı olmalı.")

        if cid in by_id:
            raise ValueError(f"Tekrarlanan karakter ID: {cid}")

        by_id[cid] = character
        by_name[name.casefold()] = cid

    data["characters"] = cast

    for scene_number, scene in enumerate(data.get("scenes", []), 1):
        turns = scene.get("dialogue_turns", [])

        if turns is None:
            turns = []

        if not isinstance(turns, list):
            raise ValueError(
                f"{scene_number}. sahnenin dialogue_turns alanı "
                "liste olmalı."
            )

        normalized_turns = []

        for turn_number, turn in enumerate(turns, 1):
            if not isinstance(turn, dict):
                raise ValueError("Geçersiz diyalog satırı.")

            speaker = safe_text(turn.get("speaker"))

            speaker_id = (
                speaker
                if speaker in by_id
                else by_name.get(speaker.casefold())
            )

            if not speaker_id:
                raise ValueError(
                    f"{scene_number}. sahne {turn_number}. konuşmacı "
                    f"karakter listesinde bulunamadı: {speaker}"
                )

            dialogue_text = safe_text(turn.get("text"))

            if not dialogue_text:
                raise ValueError(
                    f"{scene_number}. sahne {turn_number}. diyalog boş."
                )

            normalized_turns.append({
                "speaker": speaker_id,
                "text": dialogue_text,
                "emotion": safe_text(turn.get("emotion")) or "neutral",
                "delivery": safe_text(turn.get("delivery")) or "natural",
                "listener_ids": turn.get("listener_ids") or [],
            })

        scene["dialogue_turns"] = normalized_turns
        scene["narration"] = safe_text(scene.get("narration"))

    return data


def write_script(topic, notes):
    prompt = build_script_prompt(
        topic,
        notes,
        TARGET_SECONDS,
        VIDEO_DILI,
        VISUAL_STYLE,
        CUSTOM_VISUAL_STYLE,
    )

    raw = gemini_generate(
        prompt,
        SCRIPT_MODELS,
        json_mode=True,
        max_tokens=8192,
    )

    data = parse_json(raw)

    data["style"] = resolve_visual_style(
        VISUAL_STYLE,
        CUSTOM_VISUAL_STYLE,
    )

    data = normalize_script_speakers(data)

    report = validate_script(data, TARGET_SECONDS)

    if not report.get("valid"):
        errors = "; ".join(report.get("errors", []))
        raise ValueError(
            f"Senaryo kalite kontrolünden geçemedi: {errors}"
        )

    scenes = data.get("scenes", [])

    if not scenes:
        raise ValueError("Senaryoda sahne bulunamadı.")

    for index, scene in enumerate(scenes, 1):
        for field in ("image_prompt", "motion_prompt", "sound"):
            if not safe_text(scene.get(field)):
                raise ValueError(
                    f"{index}. sahnede {field} alanı eksik."
                )

        scene.setdefault("narration", "")
        scene.setdefault("dialogue_turns", [])
        scene.setdefault("visual_subject", "")
        scene.setdefault("action", "")
        scene.setdefault("location", "")
        scene.setdefault("camera", "")

    data["quality_report"] = report
    data["title"] = safe_text(data.get("title")) or topic
    data["title"] = data["title"][:95]

    return data


# ============================================================
# IMAGE GENERATION
# ============================================================

def download_file(url, path, min_bytes=1000):
    import requests

    response = requests.get(
        url,
        headers=UA,
        timeout=300,
        stream=True,
    )
    response.raise_for_status()

    temporary = Path(str(path) + ".part")
    size = 0

    try:
        with temporary.open("wb") as output:
            for chunk in response.iter_content(
                chunk_size=1024 * 256
            ):
                if chunk:
                    output.write(chunk)
                    size += len(chunk)

        if size < min_bytes:
            raise RuntimeError(
                f"İndirilen dosya çok küçük: {size} bayt."
            )

        temporary.replace(path)

    finally:
        temporary.unlink(missing_ok=True)


def make_keyframe_fal(prompt, path):
    import fal_client

    result = fal_client.subscribe(
        FAL_IMAGE_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": {
                "width": 720,
                "height": 1280,
            },
            "num_images": 1,
            "output_format": "jpeg",
            "enable_safety_checker": True,
        },
    )

    images = result.get("images") or []

    if not images or not images[0].get("url"):
        raise RuntimeError("fal.ai görsel döndürmedi.")

    flags = result.get("has_nsfw_concepts") or []

    if flags and flags[0]:
        raise RuntimeError("Görsel güvenlik filtresine takıldı.")

    download_file(images[0]["url"], path, 5000)


def make_keyframe_cloudflare(prompt, path):
    import requests

    account = os.environ.get("CF_ACCOUNT_ID")
    token = os.environ.get("CF_API_TOKEN")

    if not account or not token:
        raise RuntimeError("Cloudflare API bilgileri eksik.")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account}"
        f"/ai/run/{CF_IMAGE_MODEL}"
    )

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
        },
        json={
            "prompt": prompt[:2000],
            "steps": CF_STEPS,
        },
        timeout=180,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    if content_type.startswith("image/"):
        image_data = response.content
    else:
        result = response.json().get("result") or {}
        encoded = result.get("image")

        if not encoded:
            raise RuntimeError(
                "Cloudflare görsel verisi döndürmedi."
            )

        image_data = base64.b64decode(encoded)

    if len(image_data) < 5000:
        raise RuntimeError(
            "Cloudflare görseli bozuk veya çok küçük."
        )

    Path(path).write_bytes(image_data)


def select_image_engine():
    engine = IMAGE_ENGINE

    if engine == "auto":
        engine = (
            "fal"
            if os.environ.get("FAL_KEY")
            else "cloudflare"
        )

    if engine not in ("fal", "cloudflare"):
        raise ValueError(
            "IMAGE_ENGINE yalnızca auto, fal veya cloudflare olabilir."
        )

    if engine == "fal" and not os.environ.get("FAL_KEY"):
        raise RuntimeError(
            "fal.ai görsel motoru için FAL_KEY gerekli."
        )

    if engine == "cloudflare":
        if (
            not os.environ.get("CF_ACCOUNT_ID")
            or not os.environ.get("CF_API_TOKEN")
        ):
            raise RuntimeError(
                "Cloudflare görsel motoru için API bilgileri gerekli."
            )

    return engine


# ============================================================
# MOTION GENERATION
# ============================================================

def motion_text(style, scene):
    movement = safe_text(scene.get("motion_prompt"))
    sound = safe_text(scene.get("sound"))
    subject = safe_text(scene.get("visual_subject"))
    action = safe_text(scene.get("action"))

    return f"""
VISUAL STYLE:
{style}

VISIBLE SUBJECT:
{subject}

PHYSICAL ACTION:
{action}

ANIMATION:
{movement}

Animate the visible characters with natural body movement,
facial expressions, clothing movement and environmental motion.
Preserve character identity, proportions, faces and clothing.
Use physically coherent movement and smooth cinematic camera motion.
Do not create subtitles, captions, logos or written text.

ENVIRONMENTAL SOUND:
{sound}
"""


def make_motion(image_path, prompt, duration, output_path):
    import fal_client

    uploaded = fal_client.upload_file(str(image_path))

    result = fal_client.subscribe(
        VIDEO_MODEL,
        arguments={
            "prompt": prompt,
            "image_url": uploaded,
            "resolution": VIDEO_RES,
            "aspect_ratio": "9:16",
            "duration": duration,
            "negative_prompt": NEGATIVE,
            "enable_safety_checker": True,
        },
    )

    video = result.get("video") or {}
    url = video.get("url")

    if not url:
        raise RuntimeError(
            "Video motoru klip döndürmedi."
        )

    download_file(url, output_path, 20000)


# ============================================================
# AUDIO UTILITIES
# ============================================================

def audio_duration(path):
    result = run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])

    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Ses süresi okunamadı: {path}"
        )

    if duration <= 0:
        raise RuntimeError(
            f"Ses dosyası boş: {path}"
        )

    return duration


def estimate_words(text, duration):
    words = safe_text(text).split()

    if not words:
        return []

    weights = []

    for word in words:
        weight = len(word) + 1

        if re.search(r"[.!?…]$", word):
            weight += 6
        elif re.search(r"[,;:]$", word):
            weight += 3

        weights.append(weight)

    total_weight = sum(weights)
    available = max(duration - 0.05, 0.1)
    cursor = 0.025
    output = []

    for word, weight in zip(words, weights):
        length = available * weight / total_weight
        output.append((
            word,
            cursor,
            cursor + length,
        ))
        cursor += length

    return output


def pcm_to_mp3(pcm, sample_rate, output_path):
    raw_path = Path(str(output_path) + ".pcm")
    raw_path.write_bytes(pcm)

    try:
        run([
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-i", str(raw_path),
            "-ar", "44100",
            "-ac", "2",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            str(output_path),
        ])
    finally:
        raw_path.unlink(missing_ok=True)

    if (
        not Path(output_path).is_file()
        or Path(output_path).stat().st_size < 1000
    ):
        raise RuntimeError(
            "Gemini TTS geçerli MP3 üretemedi."
        )


# ============================================================
# TTS ENGINES
# ============================================================

def voice_gemini(text, output_path, voice_name=None):
    client, types = gemini_client()

    response = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        contents=f"{GEMINI_TTS_STYLE}\n\n{text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name or GEMINI_VOICE
                    )
                )
            ),
        ),
    )

    candidates = getattr(response, "candidates", None) or []

    if not candidates:
        raise RuntimeError(
            "Gemini TTS yanıtında aday bulunamadı."
        )

    parts = getattr(
        candidates[0].content,
        "parts",
        None,
    ) or []

    audio_part = next(
        (
            part for part in parts
            if getattr(part, "inline_data", None)
        ),
        None,
    )

    if audio_part is None:
        raise RuntimeError(
            "Gemini TTS ses verisi döndürmedi."
        )

    inline = audio_part.inline_data
    audio_data = inline.data

    if isinstance(audio_data, str):
        audio_data = base64.b64decode(audio_data)

    mime = safe_text(
        getattr(inline, "mime_type", "")
    )

    match = re.search(r"rate=(\d+)", mime)
    sample_rate = int(match.group(1)) if match else 24000

    if len(audio_data) < 2000:
        raise RuntimeError(
            "Gemini TTS ses verisi geçersiz."
        )

    pcm_to_mp3(
        audio_data,
        sample_rate,
        output_path,
    )

    return estimate_words(
        text,
        audio_duration(output_path),
    )


def voice_edge(text, output_path, voice_name=None):
    import asyncio
    import edge_tts

    async def synthesize():
        communicator = edge_tts.Communicate(
            text,
            voice_name or EDGE_VOICE,
            rate=EDGE_RATE,
        )

        audio = bytearray()
        word_times = []

        async for chunk in communicator.stream():
            if chunk.get("type") == "audio":
                audio.extend(chunk.get("data", b""))

            elif chunk.get("type") == "WordBoundary":
                start = chunk.get("offset", 0) / 10_000_000
                duration = chunk.get("duration", 0) / 10_000_000

                word_times.append((
                    safe_text(chunk.get("text")),
                    start,
                    start + duration,
                ))

        return bytes(audio), word_times

    audio_data, words = asyncio.run(synthesize())

    if len(audio_data) < 1000:
        raise RuntimeError(
            "Edge TTS ses üretmedi."
        )

    Path(output_path).write_bytes(audio_data)

    if not words:
        words = estimate_words(
            text,
            audio_duration(output_path),
        )

    return words


def voice_gtts(text, output_path, voice_name=None):
    from gtts import gTTS

    gTTS(
        text=text,
        lang=GTTS_LANG,
    ).save(str(output_path))

    if (
        not Path(output_path).is_file()
        or Path(output_path).stat().st_size < 1000
    ):
        raise RuntimeError(
            "gTTS ses dosyası oluşturamadı."
        )

    return estimate_words(
        text,
        audio_duration(output_path),
    )


def voice_elevenlabs(text, output_path, voice_name=None):
    import requests

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    selected_voice = voice_name or ELEVEN_VOICE

    if not api_key or not selected_voice:
        raise RuntimeError(
            "ElevenLabs API anahtarı veya ses ID eksik."
        )

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/"
        f"{selected_voice}/with-timestamps"
        "?output_format=mp3_44100_128"
    )

    response = requests.post(
        url,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": ELEVEN_MODEL,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.25,
                "use_speaker_boost": True,
            },
        },
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()
    encoded = data.get("audio_base64")

    if not encoded:
        raise RuntimeError(
            "ElevenLabs ses verisi döndürmedi."
        )

    Path(output_path).write_bytes(
        base64.b64decode(encoded)
    )

    alignment = data.get("alignment") or {}
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []

    words = []
    current = ""
    start_time = None
    end_time = None

    for char, start, end in zip(chars, starts, ends):
        if char.isspace():
            if current:
                words.append((
                    current,
                    start_time,
                    end_time,
                ))
                current = ""
                start_time = None
        else:
            if not current:
                start_time = start

            current += char
            end_time = end

    if current:
        words.append((
            current,
            start_time,
            end_time,
        ))

    return words or estimate_words(
        text,
        audio_duration(output_path),
    )


VOICE_FUNCTIONS = {
    "edge": voice_edge,
    "gtts": voice_gtts,
    "gemini": voice_gemini,
    "elevenlabs": voice_elevenlabs,
}


# ============================================================
# CHARACTER VOICE ASSIGNMENT
# ============================================================

def create_voice_profiles(characters):
    profiles = build_voice_profiles(characters)

    edge_voices = [
        EDGE_VOICE,
        EDGE_FEMALE_VOICE,
    ]

    for index, character in enumerate(characters):
        cid = safe_text(character.get("id"))
        profile = profiles.get(cid)

        if not profile:
            continue

        custom = character.get("voice_profile") or {}

        if not isinstance(custom, dict):
            custom = {}

        profile["edge"] = (
            safe_text(custom.get("edge"))
            or edge_voices[index % len(edge_voices)]
        )

        profile["gemini"] = (
            safe_text(custom.get("gemini"))
            or profile.get("gemini")
            or GEMINI_VOICE
        )

    return profiles


def build_scene_utterances(scene, profiles):
    utterances = []

    narration = safe_text(scene.get("narration"))

    if narration:
        utterances.append({
            "text": narration,
            "speaker_id": None,
            "voice": None,
        })

    turns = scene.get("dialogue_turns") or []

    for turn in turns:
        speaker_id = safe_text(turn.get("speaker"))
        profile = profiles.get(speaker_id)

        if not profile:
            raise ValueError(
                f"Ses profili bulunamadı: {speaker_id}"
            )

        text = safe_text(turn.get("text"))

        if not text:
            continue

        utterances.append({
            "text": text,
            "speaker_id": speaker_id,
            "voice": profile,
        })

    if not utterances:
        raise ValueError(
            "Bir sahnede anlatım veya diyalog bulunmuyor."
        )

    return utterances


# ============================================================
# COMPLETE SPEECH PIPELINE
# ============================================================

def synthesize_scene(
    scene,
    scene_index,
    work,
    engine,
    profiles,
):
    utterances = build_scene_utterances(
        scene,
        profiles,
    )

    function = VOICE_FUNCTIONS[engine]

    segment_files = []
    combined_words = []
    cursor = 0.0

    for utterance_index, utterance in enumerate(utterances):
        text = utterance["text"]
        voice_profile = utterance.get("voice") or {}

        voice_name = None

        if engine == "edge":
            voice_name = (
                voice_profile.get("edge")
                or EDGE_VOICE
            )

        elif engine == "gemini":
            voice_name = (
                voice_profile.get("gemini")
                or GEMINI_VOICE
            )

        elif engine == "elevenlabs":
            voice_name = ELEVEN_VOICE

        segment_path = (
            Path(work)
            / (
                f"speech_{scene_index:03d}_"
                f"{utterance_index:03d}.mp3"
            )
        )

        words = retry(
            lambda: function(
                text,
                segment_path,
                voice_name,
            ),
            (
                f"{engine} sahne {scene_index + 1} "
                f"konuşma {utterance_index + 1}"
            ),
            tries=2,
            wait=3,
        )

        duration = audio_duration(segment_path)
        segment_files.append(segment_path)

        for word, start, end in words:
            combined_words.append((
                word,
                cursor + start,
                cursor + end,
            ))

        cursor += duration

        if utterance_index < len(utterances) - 1:
            cursor += UTTERANCE_GAP

    output_path = (
        Path(work)
        / f"voice_{scene_index:03d}.mp3"
    )

    concat_file = (
        Path(work)
        / f"speech_list_{scene_index:03d}.txt"
    )

    lines = []

    for segment in segment_files:
        escaped = (
            str(segment.resolve())
            .replace("'", "'\\''")
        )
        lines.append(f"file '{escaped}'")

    concat_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(output_path),
    ])

    actual_duration = audio_duration(output_path)

    if actual_duration <= 0 or not combined_words:
        raise RuntimeError(
            "Sahne seslendirmesi boş üretildi."
        )

    return output_path, combined_words


def synthesize_all(scenes, work, characters):
    profiles = create_voice_profiles(characters)
    last_error = None

    engines = []

    for engine in TTS_ENGINES:
        if engine not in VOICE_FUNCTIONS:
            log(
                f"     Bilinmeyen TTS motoru atlandı: {engine}"
            )
            continue

        if engine == "elevenlabs":
            if not os.environ.get("ELEVENLABS_API_KEY"):
                continue
            if not ELEVEN_VOICE:
                continue

        if engine not in engines:
            engines.append(engine)

    if not engines:
        raise RuntimeError(
            "Kullanılabilir TTS motoru bulunamadı."
        )

    for engine in engines:
        log(f"     Ses motoru deneniyor: {engine}")

        results = []

        try:
            for index, scene in enumerate(scenes):
                audio, words = synthesize_scene(
                    scene,
                    index,
                    work,
                    engine,
                    profiles,
                )

                results.append((audio, words))

                log(
                    f"     Ses {index + 1}/{len(scenes)} hazır "
                    f"({audio_duration(audio):.1f} sn)"
                )

            return results

        except Exception as exc:
            last_error = exc

            log(
                f"     {engine} başarısız; diğer motora geçiliyor: "
                f"{str(exc)[:180]}"
            )

            for pattern in (
                "speech_*.mp3",
                "voice_*.mp3",
                "speech_list_*.txt",
            ):
                for path in Path(work).glob(pattern):
                    path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Hiçbir ses motoru çalışmadı: {last_error}"
    )


# ============================================================
# VIDEO CLIPS
# ============================================================

def make_camera_clip(
    image_path,
    frames,
    index,
    output_path,
    scene,
):
    frames = max(1, int(frames))

    camera_filter = build_zoompan_filter(
        scene or {},
        index,
        frames,
        W,
        H,
        FPS,
    )

    video_filter = (
        f"scale={W * 2}:{H * 2}:"
        "force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={W * 2}:{H * 2},"
        f"{camera_filter},"
        "eq=contrast=1.06:saturation=1.10,"
        "vignette=PI/5,"
        "noise=alls=4:allf=t+u,"
        "format=yuv420p"
    )

    run([
        "ffmpeg", "-y",
        "-i", str(image_path),
        "-vf", video_filter,
        "-frames:v", str(frames),
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-r", str(FPS),
        str(output_path),
    ])


def normalize_motion_clip(source, frames, output):
    video_filter = (
        f"scale={W}:{H}:"
        "force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={W}:{H},"
        "setsar=1,fps=30,"
        "tpad=stop_mode=clone:stop_duration=5,"
        "unsharp=3:3:0.35,"
        "format=yuv420p"
    )

    run([
        "ffmpeg", "-y",
        "-i", str(source),
        "-vf", video_filter,
        "-an",
        "-frames:v", str(frames),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "19",
        "-r", str(FPS),
        str(output),
    ])


def has_audio(path):
    result = run([
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ])

    return bool(result.stdout.strip())


def make_ambience(source, duration, output):
    if source and Path(source).is_file() and has_audio(source):
        fade_start = max(duration - 0.35, 0)

        audio_filter = (
            f"volume={AMBIENCE_VOLUME},"
            "afade=t=in:d=0.15,"
            f"afade=t=out:st={fade_start:.3f}:d=0.3,"
            "apad"
        )

        run([
            "ffmpeg", "-y",
            "-i", str(source),
            "-vn",
            "-ac", "2",
            "-ar", "44100",
            "-af", audio_filter,
            "-t", f"{duration:.4f}",
            str(output),
        ])

    else:
        run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-t", f"{duration:.4f}",
            str(output),
        ])


# ============================================================
# SUBTITLES
# ============================================================

def caption_events(words, offset, max_words=4):
    events = []
    group = []

    def flush():
        if not group:
            return

        events.append([
            offset + group[0][1],
            offset + group[-1][2],
            " ".join(item[0] for item in group),
        ])

        group.clear()

    for word in words:
        if not word or not safe_text(word[0]):
            continue

        group.append(word)

        if (
            len(group) >= max_words
            or re.search(r"[.!?…:;]$", word[0])
            or (
                len(group) >= 2
                and word[0].endswith(",")
            )
        ):
            flush()

    flush()

    for index in range(len(events) - 1):
        events[index][1] = min(
            events[index + 1][0],
            events[index][1] + 0.2,
        )

    if events:
        events[-1][1] += 0.2

    return events


def ass_time(seconds):
    centiseconds = int(
        round(max(0, seconds) * 100)
    )

    hours = centiseconds // 360000
    minutes = (centiseconds % 360000) // 6000
    secs = (centiseconds % 6000) // 100
    fraction = centiseconds % 100

    return (
        f"{hours}:{minutes:02d}:"
        f"{secs:02d}.{fraction:02d}"
    )


def write_ass(events, output):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,90,90,430,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []

    for start, end, text in events:
        clean = safe_text(text)
        clean = clean.replace("\\", "")
        clean = clean.replace("{", "").replace("}", "")
        clean = clean.replace("\n", " ")

        if end <= start:
            end = start + 0.25

        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},"
            f"Default,,0,0,0,,{clean}"
        )

    Path(output).write_text(
        header + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# FINAL ASSEMBLY
# ============================================================

def assemble(items, work, output_path, music=None):
    work = Path(work).resolve()
    output_path = Path(output_path).resolve()

    if not items:
        raise RuntimeError(
            "Montaj için sahne bulunamadı."
        )

    elapsed = 0.0
    subtitle_events = []

    for index, item in enumerate(items):
        audio = Path(item["audio"])

        if not audio.is_file():
            raise RuntimeError(
                f"Ses dosyası bulunamadı: {audio}"
            )

        duration = audio_duration(audio)
        frames = max(
            1,
            int(round((duration + PAD) * FPS)),
        )
        segment_duration = frames / FPS

        clip_output = work / f"clip_{index:03d}.mp4"

        if item.get("clip"):
            normalize_motion_clip(
                item["clip"],
                frames,
                clip_output,
            )
        else:
            make_camera_clip(
                item["image"],
                frames,
                index,
                clip_output,
                item.get("scene", {}),
            )

        narration_output = work / f"nar_{index:03d}.wav"

        run([
            "ffmpeg", "-y",
            "-i", str(audio),
            "-af", "apad",
            "-ar", "44100",
            "-ac", "2",
            "-t", f"{segment_duration:.4f}",
            str(narration_output),
        ])

        ambience_output = work / f"amb_{index:03d}.wav"

        make_ambience(
            item.get("clip"),
            segment_duration,
            ambience_output,
        )

        subtitle_events.extend(
            caption_events(
                item["words"],
                elapsed,
            )
        )

        elapsed += segment_duration

        log(
            f"  Montaj sahnesi {index + 1}/{len(items)} "
            f"({segment_duration:.1f} sn)"
        )

    total_duration = elapsed
    count = len(items)

    clip_list = work / "clips.txt"
    narration_list = work / "narration.txt"
    ambience_list = work / "ambience.txt"

    clip_list.write_text(
        "".join(
            f"file 'clip_{index:03d}.mp4'\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )

    narration_list.write_text(
        "".join(
            f"file 'nar_{index:03d}.wav'\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )

    ambience_list.write_text(
        "".join(
            f"file 'amb_{index:03d}.wav'\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "clips.txt",
        "-c", "copy",
        "video_raw.mp4",
    ], cwd=work)

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "narration.txt",
        "-c", "copy",
        "narration.wav",
    ], cwd=work)

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "ambience.txt",
        "-c", "copy",
        "ambience.wav",
    ], cwd=work)

    write_ass(
        subtitle_events,
        work / "subs.ass",
    )

    command = [
        "ffmpeg", "-y",
        "-i", "video_raw.mp4",
        "-i", "narration.wav",
        "-i", "ambience.wav",
    ]

    filters = (
        "[0:v]ass=subs.ass[v];"
        "[1:a][2:a]amix=inputs=2:duration=first:"
        "dropout_transition=0:normalize=0[nm];"
    )

    if music and Path(music).is_file():
        command.extend([
            "-stream_loop", "-1",
            "-i", str(Path(music).resolve()),
        ])

        fade_start = max(
            total_duration - 2.5,
            0,
        )

        filters += (
            f"[3:a]atrim=0:{total_duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            "volume=0.10,"
            "afade=t=in:d=1.5,"
            f"afade=t=out:st={fade_start:.3f}:d=2.5[m];"
            "[nm][m]amix=inputs=2:duration=first:"
            "dropout_transition=0:normalize=0[mix];"
        )

    else:
        filters += "[nm]anull[mix];"

    filters += (
        "[mix]loudnorm=I=-16:TP=-1.5:LRA=11[a]"
    )

    command.extend([
        "-filter_complex", filters,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-movflags", "+faststart",
        "-t", f"{total_duration:.3f}",
        str(output_path),
    ])

    run(command, cwd=work)

    if (
        not output_path.is_file()
        or output_path.stat().st_size < 50000
    ):
        raise RuntimeError(
            "Son video dosyası oluşturulamadı veya bozuk."
        )

    return total_duration


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--topic",
        required=True,
    )

    args = parser.parse_args()

    topic = safe_text(args.topic)

    if not topic:
        raise ValueError(
            "Video konusu boş olamaz."
        )

    check_tools()

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY bulunamadı."
        )

    image_engine = select_image_engine()

    fal_available = bool(
        os.environ.get("FAL_KEY")
    )

    motion_mode = (
        "real"
        if MOTION != "off" and fal_available
        else "preview"
    )

    if MOTION_REQUIRED and motion_mode != "real":
        raise RuntimeError(
            "Gerçek hareket zorunlu; FAL_KEY bulunamadı "
            "veya hareket kapalı."
        )

    require_real_motion(
        motion_mode,
        fal_available,
    )

    output_dir = Path("output")
    work_dir = Path("work")

    output_dir.mkdir(exist_ok=True)

    shutil.rmtree(
        work_dir,
        ignore_errors=True,
    )

    work_dir.mkdir(exist_ok=True)

    log("=" * 60)
    log("SİNEMATİK VİDEO ÜRETİMİ")
    log(f"Konu: {topic}")
    log(f"Hedef süre: {TARGET_SECONDS} saniye")
    log(f"Görsel motoru: {image_engine}")

    log(
        "Hareket modu: "
        + (
            "GERÇEK AI KLİBİ"
            if motion_mode == "real"
            else "KAMERA HAREKETİ"
        )
    )

    log("=" * 60)

    # 1. ARAŞTIRMA
    log("\n1/6 - Araştırma")

    notes = retry(
        lambda: research(topic),
        "Araştırma",
        tries=2,
        wait=5,
    )

    (output_dir / "research.md").write_text(
        notes,
        encoding="utf-8",
    )

    # 2. SENARYO
    log("\n2/6 - Senaryo ve karakterler")

    script = retry(
        lambda: write_script(topic, notes),
        "Senaryo",
        tries=3,
        wait=5,
    )

    scenes = script["scenes"]
    characters = script["characters"]

    write_json(
        output_dir / "script.json",
        script,
    )

    log(
        f"Başlık: {script['title']} | "
        f"Sahneler: {len(scenes)} | "
        f"Karakterler: {len(characters)}"
    )

    # 3. GÖRSELLER
    log("\n3/6 - Görsel üretimi")

    style = safe_text(script.get("style"))
    character_bible = safe_text(
        script.get("character_bible")
    )

    prompts = []

    for scene in scenes:
        prompt = enrich_image_prompt(
            style,
            character_bible,
            scene,
        )

        prompt += (
            " Premium cinematic film frame, vertical 9:16 "
            "composition, detailed environment, consistent "
            "character appearance, clear physical action, "
            "no text, no logo, no watermark."
        )

        prompts.append(prompt)

    image_paths = [
        work_dir / f"image_{index:03d}.jpg"
        for index in range(len(scenes))
    ]

    image_function = (
        make_keyframe_fal
        if image_engine == "fal"
        else make_keyframe_cloudflare
    )

    def generate_image(index):
        retry(
            lambda: image_function(
                prompts[index],
                image_paths[index],
            ),
            f"Görsel {index + 1}",
            tries=3,
            wait=5,
        )

        if not image_paths[index].is_file():
            raise RuntimeError(
                "Görsel dosyası oluşmadı."
            )

        log(
            f"     Görsel {index + 1}/{len(scenes)} hazır"
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(
            executor.map(
                generate_image,
                range(len(scenes)),
            )
        )

    # 4. SES
    log("\n4/6 - Anlatım ve karakter diyalogları")

    voices = synthesize_all(
        scenes,
        work_dir,
        characters,
    )

    if len(voices) != len(scenes):
        raise RuntimeError(
            "Ses ve sahne sayısı eşleşmiyor."
        )

    # 5. HAREKET
    log("\n5/6 - Hareketli klip üretimi")

    clip_paths = [
        work_dir / f"motion_{index:03d}.mp4"
        for index in range(len(scenes))
    ]

    motion_success = [
        False
        for _ in scenes
    ]

    if motion_mode == "real":
        durations = [
            int(
                min(
                    12,
                    max(
                        3,
                        math.ceil(
                            audio_duration(audio) + PAD
                        ),
                    ),
                )
            )
            for audio, _ in voices
        ]

        def generate_motion(index):
            prompt = motion_text(
                style,
                scenes[index],
            )

            retry(
                lambda: make_motion(
                    image_paths[index],
                    prompt,
                    durations[index],
                    clip_paths[index],
                ),
                f"Hareketli klip {index + 1}",
                tries=2,
                wait=8,
            )

            motion_success[index] = True

            log(
                f"     Klip {index + 1}/{len(scenes)} hazır"
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(
                executor.map(
                    generate_motion,
                    range(len(scenes)),
                )
            )

        valid_clips = validate_motion_outputs(
            clip_paths,
            mode="real",
        )

        if len(valid_clips) != len(scenes):
            failed = len(scenes) - len(valid_clips)

            if (
                MOTION_REQUIRED
                or failed > len(scenes) / 2
            ):
                raise RuntimeError(
                    f"{failed} hareketli klip üretilemedi. "
                    "Kalite kontrolü nedeniyle işlem durduruldu."
                )

            log(
                f"     Uyarı: {failed} klipte kamera hareketi "
                "yedek olarak kullanılacak."
            )

    else:
        log(
            "     Ücretsiz mod: gerçek karakter animasyonu "
            "yerine sinematik kamera hareketi kullanılacak."
        )

    # 6. MONTAJ
    log("\n6/6 - Video montajı")

    music_tracks = glob.glob(
        "assets/music/*.mp3"
    )

    music = (
        random.choice(music_tracks)
        if music_tracks
        else None
    )

    log(
        f"Müzik: {music or 'Ek müzik yok'}"
    )

    items = []

    for index, scene in enumerate(scenes):
        audio_path, words = voices[index]

        items.append({
            "image": image_paths[index],
            "clip": (
                clip_paths[index]
                if motion_success[index]
                else None
            ),
            "audio": audio_path,
            "words": words,
            "scene": scene,
        })

    final_video = output_dir / "video.mp4"

    assemble(
        items,
        work_dir,
        final_video,
        music,
    )

    # SON KONTROLLER
    if not final_video.is_file():
        raise RuntimeError(
            "Video çıktısı bulunamadı."
        )

    if final_video.stat().st_size < 50000:
        raise RuntimeError(
            "Video dosyası beklenenden küçük."
        )

    final_duration = audio_duration(
        final_video
    )

    if final_duration < 1:
        raise RuntimeError(
            "Video süresi geçersiz."
        )

    shutil.copy2(
        image_paths[0],
        output_dir / "thumbnail.jpg",
    )

    metadata = {
        "topic": topic,
        "title": script.get("title", topic),
        "description": script.get("description", ""),
        "tags": script.get("tags", []),
        "duration_seconds": round(final_duration, 2),
        "scene_count": len(scenes),
        "character_count": len(characters),
        "dialogue_turn_count": sum(
            len(scene.get("dialogue_turns", []))
            for scene in scenes
        ),
        "motion_clips": (
            f"{sum(motion_success)}/{len(scenes)}"
        ),
        "motion_mode": motion_mode,
        "image_engine": image_engine,
        "tts_engines": TTS_ENGINES,
    }

    write_json(
        output_dir / "metadata.json",
        metadata,
    )

    log("\n" + "=" * 60)
    log("ÜRETİM TAMAMLANDI")
    log(f"Video: {final_video}")
    log(f"Süre: {final_duration:.1f} saniye")
    log(f"Sahne: {len(scenes)}")
    log(f"Karakter: {len(characters)}")
    log(
        f"Gerçek hareketli klip: {metadata['motion_clips']}"
    )
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        log(
            "\nİşlem kullanıcı tarafından durduruldu."
        )
        sys.exit(130)

    except Exception as error:
        log("\nÜRETİM HATASI")
        log(str(error))
        sys.exit(1)

