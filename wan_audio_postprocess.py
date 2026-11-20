
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Wan film audio post-processor: TTS, subtitles, audio mixing and lip-sync timeline."""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from audio_asset_mixer import mix_audio_layers, write_mix_report


GEMINI_MODEL = os.getenv(
    "GEMINI_TTS_MODEL",
    "gemini-2.5-flash-preview-tts",
)
GEMINI_STYLE = os.getenv(
    "GEMINI_TTS_STYLE",
    "Speak naturally in Turkish with clear pronunciation and cinematic pacing.",
)
GEMINI_QUOTA_CONFIRMED = (
    os.getenv("GEMINI_TTS_FREE_QUOTA_CONFIRMED", "false").lower()
    == "true"
)
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Charon")
EDGE_VOICE = os.getenv("EDGE_VOICE", "tr-TR-AhmetNeural")
EDGE_FEMALE_VOICE = os.getenv("EDGE_FEMALE_VOICE", "tr-TR-EmelNeural")
GTTS_LANG = os.getenv("GTTS_LANG", "tr")

GAP_SECONDS = 0.12
SCENE_DURATION = 81 / 16

REPORT = {
    "status": "starting",
    "providers_requested": [],
    "provider_used": None,
    "paid_services_blocked": [
        "xAI/Grok TTS",
        "FAL music generation",
        "FAL sound effects",
    ],
    "fallbacks": [],
    "audio_duration_seconds": None,
    "video_duration_seconds": None,
    "subtitle_file": None,
    "output_file": None,
    "mix_report_file": None,
    "lip_sync_timeline_file": None,
    "mix_layers": [],
    "notes": [],
}


def run(command):
    return subprocess.run(
        [str(item) for item in command],
        check=True,
        capture_output=True,
        text=True,
    )


def require_binary(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"Gerekli program bulunamadı: {name}")


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Proje JSON bulunamadı: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def probe_duration(path):
    result = run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise RuntimeError(f"Geçersiz medya süresi: {path}")
    return duration


def clean_text(value):
    return "" if value is None else str(value).strip()


def timestamp(seconds):
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{secs:02d},{milliseconds:03d}"
    )


def estimate_word_times(text, duration):
    words = clean_text(text).split()
    if not words:
        return []

    weights = []
    for word in words:
        weight = len(word) + 1
        if re.search(r"[.!?…]$", word):
            weight += 5
        elif re.search(r"[,;:]$", word):
            weight += 2
        weights.append(weight)

    total = sum(weights)
    available = max(duration - 0.04, 0.1)
    cursor = 0.02
    result = []

    for word, weight in zip(words, weights):
        length = available * weight / total
        result.append((word, cursor, cursor + length))
        cursor += length

    return result


def emotion_instruction(emotion, delivery):
    emotion = clean_text(emotion).lower()
    delivery = clean_text(delivery).lower()

    descriptions = {
        "angry": "controlled anger and firm articulation",
        "sad": "subdued sadness and gentle delivery",
        "excited": "energetic and enthusiastic delivery",
        "fearful": "nervous but understandable delivery",
        "calm": "calm and composed delivery",
        "whispering": "soft and quiet delivery",
        "confident": "confident and decisive delivery",
        "neutral": "natural conversational delivery",
    }

    description = descriptions.get(emotion, descriptions["neutral"])
    if delivery:
        description += ", " + delivery
    return description


def build_profiles(characters):
    if not isinstance(characters, list):
        raise ValueError("Karakter listesi geçersiz.")
    if not 2 <= len(characters) <= 8:
        raise ValueError("Film kadrosunda 2-8 karakter bulunmalıdır.")

    gemini_voices = [
        "Charon", "Kore", "Fenrir", "Aoede",
        "Puck", "Leda", "Orus", "Zephyr",
    ]

    profiles = {}
    used_voices = set()

    for index, character in enumerate(characters):
        character_id = clean_text(character.get("id"))
        name = clean_text(character.get("name"))

        if not character_id or not name:
            raise ValueError("Her karakterin ID ve adı bulunmalıdır.")
        if character_id in profiles:
            raise ValueError(f"Tekrarlanan karakter ID: {character_id}")

        custom = character.get("voice_profile") or {}
        if not isinstance(custom, dict):
            custom = {}

        gender = clean_text(character.get("voice_gender")).lower()
        edge_default = (
            EDGE_FEMALE_VOICE if gender == "female" else EDGE_VOICE
        )

        gemini_voice = (
            clean_text(custom.get("gemini")) or gemini_voices[index]
        )

        if gemini_voice not in gemini_voices:
            raise ValueError(
                f"Desteklenmeyen Gemini sesi: {gemini_voice}"
            )
        if gemini_voice in used_voices:
            raise ValueError(
                f"Gemini sesi birden fazla karaktere atanmış: {gemini_voice}"
            )

        used_voices.add(gemini_voice)

        profiles[character_id] = {
            "name": name,
            "edge": clean_text(custom.get("edge")) or edge_default,
            "gemini": gemini_voice,
            "voice_description": clean_text(character.get("voice")),
        }

    return profiles


def tts_edge(text, output_path, voice, emotion="neutral", delivery="natural"):
    import edge_tts

    async def generate():
        communicator = edge_tts.Communicate(text, voice)
        await communicator.save(str(output_path))

    asyncio.run(generate())

    if (
        not Path(output_path).is_file()
        or Path(output_path).stat().st_size < 1000
    ):
        raise RuntimeError("Edge TTS ses üretemedi.")


def tts_gtts(text, output_path, voice=None, emotion="neutral", delivery="natural"):
    from gtts import gTTS

    gTTS(text=text, lang=GTTS_LANG, slow=False).save(str(output_path))

    if (
        not Path(output_path).is_file()
        or Path(output_path).stat().st_size < 1000
    ):
        raise RuntimeError("gTTS ses üretemedi.")


def tts_gemini(
    text,
    output_path,
    voice,
    emotion="neutral",
    delivery="natural",
    voice_description="",
):
    if not GEMINI_QUOTA_CONFIRMED:
        raise RuntimeError("Gemini ücretsiz kotası doğrulanmadı.")

    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY GitHub Secret bulunamadı.")

    client = genai.Client(api_key=api_key)

    instruction = (
        f"{GEMINI_STYLE}\n"
        f"Emotion: {emotion}.\n"
        f"Delivery: {emotion_instruction(emotion, delivery)}.\n"
        f"Character voice: {voice_description or 'natural voice'}.\n"
        "Speak the supplied text in Turkish. "
        "Do not add explanations or extra words."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{instruction}\n\n{text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice or GEMINI_VOICE
                    )
                )
            ),
        ),
    )

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("Gemini ses yanıtı boş.")

    parts = getattr(candidates[0].content, "parts", None) or []
    audio_part = next(
        (part for part in parts if getattr(part, "inline_data", None)),
        None,
    )
    if audio_part is None:
        raise RuntimeError("Gemini ses verisi döndürmedi.")

    inline = audio_part.inline_data
    audio_data = inline.data

    import base64
    if isinstance(audio_data, str):
        audio_data = base64.b64decode(audio_data)

    mime = clean_text(getattr(inline, "mime_type", ""))
    match = re.search(r"rate=(\d+)", mime)
    sample_rate = int(match.group(1)) if match else 24000

    raw_path = Path(str(output_path) + ".pcm")
    raw_path.write_bytes(audio_data)

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
        raise RuntimeError("Gemini MP3 dosyası oluşturulamadı.")


PROVIDERS = {
    "gemini": tts_gemini,
    "edge": tts_edge,
    "gtts": tts_gtts,
}


def build_utterances(scene, profiles):
    result = []
    narration = clean_text(scene.get("narration"))

    if narration:
        result.append({
            "text": narration,
            "speaker_id": None,
            "voice": None,
            "emotion": "neutral",
            "delivery": "natural",
        })

    turns = scene.get("dialogue_turns") or []
    if not isinstance(turns, list):
        raise ValueError("dialogue_turns listesi geçersiz.")

    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("Diyalog satırı geçersiz.")

        speaker_id = clean_text(turn.get("speaker"))
        if speaker_id not in profiles:
            raise ValueError(f"Ses profili bulunamadı: {speaker_id}")

        text = clean_text(turn.get("text"))
        if not text:
            continue

        result.append({
            "text": text,
            "speaker_id": speaker_id,
            "voice": profiles[speaker_id],
            "emotion": clean_text(turn.get("emotion")) or "neutral",
            "delivery": clean_text(turn.get("delivery")) or "natural",
        })

    if not result:
        raise ValueError("Sahnede seslendirilecek metin bulunamadı.")

    return result


def create_silence(path, duration):
    run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(max(duration, 0.01)),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(path),
    ])


def synthesize_scene(
    scene,
    scene_index,
    work_dir,
    provider,
    profiles,
    subtitle_entries,
    timeline,
    lip_sync_entries,
):
    utterances = build_utterances(scene, profiles)
    function = PROVIDERS[provider]
    segments = []
    scene_cursor = 0.0

    for turn_index, utterance in enumerate(utterances):
        profile = utterance["voice"] or {}

        if provider == "gemini":
            voice = profile.get("gemini") or GEMINI_VOICE
        elif provider == "edge":
            voice = profile.get("edge") or EDGE_VOICE
        else:
            voice = None

        segment_path = (
            work_dir
            / f"scene_{scene_index:03d}_turn_{turn_index:03d}.mp3"
        )

        if provider == "gemini":
            function(
                utterance["text"],
                segment_path,
                voice,
                utterance["emotion"],
                utterance["delivery"],
                profile.get("voice_description", ""),
            )
        else:
            function(
                utterance["text"],
                segment_path,
                voice,
                utterance["emotion"],
                utterance["delivery"],
            )

        duration = probe_duration(segment_path)
        if duration <= 0:
            raise RuntimeError(
                f"Scene {scene_index + 1}: boş ses segmenti."
            )

        local_start = scene_cursor
        local_end = local_start + duration
        scene_start = scene_index * SCENE_DURATION

        words = estimate_word_times(utterance["text"], duration)

        for word, start, end in words:
            subtitle_entries.append({
                "text": word,
                "start": scene_start + local_start + start,
                "end": scene_start + local_start + end,
            })

        # Konuşmacı bazlı senkron zaman çizelgesi.
        # Anlatıcı kayıtları dudak senkronu listesine eklenmez.
        if utterance["speaker_id"]:
            lip_sync_entries.append({
                "scene_index": scene_index,
                "turn_index": turn_index,
                "speaker_id": utterance["speaker_id"],
                "character_name": profile.get("name", ""),
                "text": utterance["text"],
                "audio_file": str(segment_path.resolve()),
                "audio_duration_seconds": round(duration, 4),
                "scene_local_start_seconds": round(local_start, 4),
                "scene_local_end_seconds": round(local_end, 4),
                "film_start_seconds": round(
                    scene_start + local_start, 4
                ),
                "film_end_seconds": round(
                    scene_start + local_end, 4
                ),
                "emotion": utterance["emotion"],
                "delivery": utterance["delivery"],
            })

        segments.append(segment_path)
        scene_cursor += duration

        if turn_index < len(utterances) - 1:
            gap_path = (
                work_dir
                / f"scene_{scene_index:03d}_gap_{turn_index:03d}.mp3"
            )
            create_silence(gap_path, GAP_SECONDS)
            segments.append(gap_path)
            scene_cursor += GAP_SECONDS

    if scene_cursor > SCENE_DURATION:
        raise RuntimeError(
            f"Scene {scene_index + 1} ses süresi aşıldı: "
            f"{scene_cursor:.2f}s / {SCENE_DURATION:.2f}s. "
            "Konuşma kesilmedi. Sahne metnini kısaltın "
            "veya video sahne süresini artırın."
        )

    concat_file = work_dir / f"scene_{scene_index:03d}_concat.txt"
    concat_file.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in segments),
        encoding="utf-8",
    )

    raw_scene_audio = work_dir / f"scene_{scene_index:03d}_raw.mp3"
    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(raw_scene_audio),
    ])

    actual_duration = probe_duration(raw_scene_audio)
    if actual_duration > SCENE_DURATION + 0.08:
        raise RuntimeError(
            f"Scene {scene_index + 1}: birleştirilmiş ses "
            f"sahne süresini aşıyor ({actual_duration:.2f}s)."
        )

    scene_audio = work_dir / f"scene_{scene_index:03d}_audio.mp3"
    run([
        "ffmpeg", "-y",
        "-i", str(raw_scene_audio),
        "-af", "apad",
        "-t", str(SCENE_DURATION),
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(scene_audio),
    ])

    timeline[0] = (scene_index + 1) * SCENE_DURATION
    return scene_audio


def write_srt(entries, output_path):
    lines = []

    for index, entry in enumerate(entries, 1):
        lines.extend([
            str(index),
            (
                timestamp(entry["start"])
                + " --> "
                + timestamp(entry["end"])
            ),
            entry["text"],
            "",
        ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def mux_audio(video_path, audio_path, output_path):
    video_duration = probe_duration(video_path)
    audio_duration = probe_duration(audio_path)

    REPORT["video_duration_seconds"] = round(video_duration, 3)
    REPORT["audio_duration_seconds"] = round(audio_duration, 3)

    if audio_duration > video_duration + 0.5:
        raise RuntimeError(
            "Ses video süresinden uzun. Konuşmalar kesilmeyecek; "
            "sahne süreleri veya kurgu yeniden düzenlenmeli. "
            f"Video={video_duration:.2f}s, Ses={audio_duration:.2f}s"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "apad",
        "-t", str(video_duration),
        "-movflags", "+faststart",
        str(output_path),
    ])

    if (
        not Path(output_path).is_file()
        or Path(output_path).stat().st_size < 10000
    ):
        raise RuntimeError("Sesli final video oluşturulamadı.")


def _licensed_asset(settings, name, project_dir):
    """Convert canonical asset settings to the mixer's input format."""
    if not isinstance(settings, dict) or not settings.get("enabled", False):
        return None

    asset_path = clean_text(settings.get("asset_path"))
    if not asset_path:
        raise RuntimeError(
            f"{name}: etkinleştirilmiş ancak asset_path boş."
        )

    asset = Path(asset_path)
    if not asset.is_absolute():
        asset = Path(project_dir) / asset

    return {
        "path": str(asset.resolve()),
        "volume": settings.get("volume", 0.2),
        "license_verified": settings.get("license_verified") is True,
    }


def build_audio_mix_configuration(audio_settings, scenes, project_dir):
    """Resolve enabled local audio assets and scene effects."""
    music_settings = audio_settings.get("music") or {}
    ambience_settings = audio_settings.get("ambience") or {}
    effects_settings = audio_settings.get("sound_effects") or {}

    music = _licensed_asset(music_settings, "Müzik", project_dir)
    ambience = _licensed_asset(ambience_settings, "Ambiyans", project_dir)

    effects = []
    ignored_text_effects = 0

    if effects_settings.get("enabled", False):
        for scene_index, scene in enumerate(scenes):
            scene_effects = scene.get("sound_effects") or []
            if not isinstance(scene_effects, list):
                continue

            for effect_index, effect in enumerate(scene_effects):
                if isinstance(effect, str):
                    if effect.strip():
                        ignored_text_effects += 1
                    continue

                if not isinstance(effect, dict):
                    continue

                asset_path = clean_text(effect.get("asset_path"))
                if not asset_path:
                    raise RuntimeError(
                        f"Sahne {scene_index + 1}, efekt {effect_index + 1}: "
                        "asset_path boş."
                    )

                if effect.get("license_verified") is not True:
                    raise RuntimeError(
                        f"Sahne {scene_index + 1}, efekt {effect_index + 1}: "
                        "efekt lisansı doğrulanmamış."
                    )

                try:
                    local_offset = float(effect.get("offset", 0.0))
                except (TypeError, ValueError, OverflowError):
                    raise RuntimeError(
                        f"Sahne {scene_index + 1}: efekt offset geçersiz."
                    )

                if local_offset < 0:
                    raise RuntimeError("Efekt offset değeri negatif olamaz.")

                effect_path = Path(asset_path)
                if not effect_path.is_absolute():
                    effect_path = Path(project_dir) / effect_path

                effects.append({
                    "path": str(effect_path.resolve()),
                    "volume": effect.get(
                        "volume",
                        effects_settings.get("volume", 0.45),
                    ),
                    "license_verified": (
                        effect.get("license_verified") is True
                    ),
                    "offset": scene_index * SCENE_DURATION + local_offset,
                })

    return {
        "music": music,
        "ambience": ambience,
        "effects": effects,
        "ignored_text_effects": ignored_text_effects,
        "mix_settings": audio_settings.get("mix") or {},
        "music_ducking": bool(music_settings.get("ducking", True)),
    }


def write_lip_sync_manifest(entries, output_path, scenes, profiles):
    """Write dialogue audio timing metadata for a later GPU lip-sync stage."""
    manifest = {
        "schema_version": "1.0",
        "status": "timeline_ready",
        "lip_sync_rendered": False,
        "description": (
            "Dialogue audio segments and timestamps. "
            "This manifest is input metadata, not a rendered lip-sync result."
        ),
        "scene_duration_seconds": SCENE_DURATION,
        "total_scenes": len(scenes),
        "total_dialogue_turns": len(entries),
        "characters": [
            {
                "speaker_id": speaker_id,
                "name": profile.get("name", ""),
            }
            for speaker_id, profile in profiles.items()
        ],
        "dialogue": sorted(
            entries,
            key=lambda item: (
                item["film_start_seconds"],
                item["scene_index"],
                item["turn_index"],
            ),
        ),
        "processing_requirements": {
            "requires_video_face_tracking": True,
            "requires_speaker_to_face_matching": True,
            "requires_gpu_video_inference": True,
            "skip_ambiguous_face_matches": True,
        },
    }
    write_json(output_path, manifest)


def process(project_path, video_path, output_path, work_dir):
    require_binary("ffmpeg")
    require_binary("ffprobe")

    project = read_json(project_path)
    audio_settings = project.get("audio") or {}
    characters = project.get("characters") or []
    scenes = project.get("scenes") or []

    if not Path(video_path).is_file():
        raise RuntimeError(f"Video bulunamadı: {video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if audio_settings.get("enabled") is False:
        shutil.copy2(video_path, output_path)

        REPORT["status"] = "success_audio_disabled"
        REPORT["output_file"] = str(output_path)
        REPORT["notes"].append(
            "Proje ses sistemi kapalı. Sessiz video değiştirilmeden aktarıldı."
        )

        report_path = output_path.with_name("audio_runtime_report.json")
        write_json(report_path, REPORT)

        print("AUDIO_POSTPROCESS_SKIPPED")
        print("Audio disabled; original video copied.")
        print("Final:", output_path)
        print("Report:", report_path)
        return

    if not scenes:
        raise RuntimeError("Film sahneleri bulunamadı.")

    profiles = build_profiles(characters)

    configured = audio_settings.get(
        "tts_provider_order",
        ["gemini", "edge", "gtts"],
    )
    if not isinstance(configured, list):
        configured = ["edge", "gtts"]

    REPORT["providers_requested"] = configured
    providers = []

    for name in configured:
        name = clean_text(name).lower()
        if name == "xai" or name not in PROVIDERS:
            continue

        if name == "gemini":
            if not GEMINI_QUOTA_CONFIRMED:
                REPORT["notes"].append(
                    "Gemini atlandı: ücretsiz kota doğrulanmadı."
                )
                continue
            if not os.getenv("GEMINI_API_KEY"):
                REPORT["notes"].append(
                    "Gemini atlandı: API anahtarı yok."
                )
                continue

        if name not in providers:
            providers.append(name)

    for fallback in ("edge", "gtts"):
        if fallback not in providers:
            providers.append(fallback)

    last_error = None

    for provider in providers:
        try:
            REPORT["notes"].append(f"{provider} sağlayıcısı deneniyor.")

            scene_audio_files = []
            subtitle_entries = []
            lip_sync_entries = []
            timeline = [0.0]

            for index, scene in enumerate(scenes):
                audio_file = synthesize_scene(
                    scene,
                    index,
                    work_dir,
                    provider,
                    profiles,
                    subtitle_entries,
                    timeline,
                    lip_sync_entries,
                )
                scene_audio_files.append(audio_file)

            concat_file = work_dir / "film_audio_concat.txt"
            concat_file.write_text(
                "".join(
                    f"file '{path.resolve()}'\n"
                    for path in scene_audio_files
                ),
                encoding="utf-8",
            )

            combined_audio = work_dir / "film_dialogue.mp3"
            run([
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                str(combined_audio),
            ])

            if probe_duration(combined_audio) <= 0:
                raise RuntimeError("Birleştirilmiş ses boş.")

            srt_path = output_path.with_suffix(".srt")
            write_srt(subtitle_entries, srt_path)

            timeline_path = output_path.with_name(
                "lip_sync_timeline.json"
            )
            write_lip_sync_manifest(
                lip_sync_entries,
                timeline_path,
                scenes,
                profiles,
            )

            video_duration = probe_duration(video_path)

            mix_config = build_audio_mix_configuration(
                audio_settings,
                scenes,
                Path(project_path).resolve().parent,
            )

            music_asset = mix_config["music"]
            ambience_asset = mix_config["ambience"]
            effect_assets = mix_config["effects"]

            mix_report = {
                "status": "dialogue_only",
                "layers": ["dialogue"],
                "ignored_text_effect_descriptions": (
                    mix_config["ignored_text_effects"]
                ),
                "notes": [],
            }

            if music_asset or ambience_asset or effect_assets:
                mixed_audio = work_dir / "film_final_mix.m4a"
                mix_settings = mix_config["mix_settings"]

                mix_report = mix_audio_layers(
                    voice_path=combined_audio,
                    output_path=mixed_audio,
                    duration=video_duration,
                    music=music_asset,
                    ambience=ambience_asset,
                    effects=effect_assets,
                    sample_rate=int(
                        mix_settings.get("sample_rate", 48000)
                    ),
                    loudness_lufs=float(
                        mix_settings.get("loudness_lufs", -16.0)
                    ),
                    true_peak_db=float(
                        mix_settings.get("true_peak_db", -1.5)
                    ),
                    music_ducking=mix_config["music_ducking"],
                )

                audio_for_video = mixed_audio
                mix_report["status"] = "success"
            else:
                audio_for_video = combined_audio
                mix_report["notes"].append(
                    "Etkin lisanslı müzik, ambiyans veya ses efekti "
                    "bulunmadığından diyalog sesi doğrudan kullanıldı."
                )

            mix_report["ignored_text_effect_descriptions"] = (
                mix_config["ignored_text_effects"]
            )

            mix_report_path = output_path.with_name(
                "audio_mix_report.json"
            )
            write_mix_report(mix_report_path, mix_report)

            mux_audio(video_path, audio_for_video, output_path)

            REPORT["mix_report_file"] = str(mix_report_path)
            REPORT["mix_layers"] = mix_report.get(
                "layers",
                ["dialogue"],
            )
            REPORT["status"] = "success"
            REPORT["provider_used"] = provider
            REPORT["subtitle_file"] = str(srt_path)
            REPORT["output_file"] = str(output_path)
            REPORT["lip_sync_timeline_file"] = str(timeline_path)

            if provider != providers[0]:
                REPORT["fallbacks"].append({
                    "selected": provider,
                    "reason": str(
                        last_error or "Önceki sağlayıcı atlandı."
                    ),
                })

            REPORT["notes"].append(
                "Konuşma ve anlatım sesleri üretildi."
            )
            REPORT["notes"].append(
                "Altyazı zamanları ses segmentlerinden oluşturuldu."
            )
            REPORT["notes"].append(
                "Konuşmacı bazlı dudak senkronu zaman çizelgesi oluşturuldu."
            )
            REPORT["notes"].append(
                "Dudak senkronu video render işlemi henüz uygulanmadı."
            )
            REPORT["notes"].append(
                "Müzik, ambiyans ve efektler yalnızca etkinleştirilmiş, "
                "dosyası bulunan ve lisansı doğrulanmış yerel varlıklardan "
                "miksaja eklenir. Metin efekt açıklamaları ses dosyası "
                "olmadığı için üretilmez."
            )

            report_path = output_path.with_name(
                "audio_runtime_report.json"
            )
            write_json(report_path, REPORT)

            print("AUDIO_POSTPROCESS_SUCCESS")
            print("Provider:", provider)
            print("Final:", output_path)
            print("Subtitles:", srt_path)
            print("Lip-sync timeline:", timeline_path)
            print("Mix report:", mix_report_path)
            print("Report:", report_path)
            return

        except Exception as exc:
            last_error = exc
            REPORT["fallbacks"].append({
                "failed_provider": provider,
                "error": str(exc)[:500],
            })
            print(
                f"{provider} başarısız: {exc}",
                file=sys.stderr,
            )

    REPORT["status"] = "failed"
    REPORT["notes"].append(
        "Hiçbir ücretsiz ses sağlayıcısı başarılı olmadı."
    )

    report_path = output_path.with_name(
        "audio_runtime_report.json"
    )
    write_json(report_path, REPORT)

    raise RuntimeError(
        f"Ses üretimi başarısız. Son hata: {last_error}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="output/final-film.mp4")
    parser.add_argument("--work-dir", default=".work/audio")
    args = parser.parse_args()

    try:
        process(
            args.project,
            args.video,
            args.output,
            args.work_dir,
        )
    except Exception as exc:
        print(
            f"AUDIO_POSTPROCESS_FAILED: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

