
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

FPS = 16
SCENE_FRAMES = 81
SCENE_SECONDS = SCENE_FRAMES / FPS
SAMPLE_RATE = 44100
CHANNELS = 2
LINE_GAP = 0.18

DEFAULT_VOICES = {
    "male": "tr-TR-AhmetNeural",
    "female": "tr-TR-EmelNeural",
}


def run(command, cwd=None):
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    return result.stdout.strip()


def require_tools():
    missing = [
        name for name in ("ffmpeg", "ffprobe")
        if not shutil.which(name)
    ]
    if missing:
        raise RuntimeError(
            "Gerekli araçlar bulunamadı: " + ", ".join(missing)
        )


def probe_duration(path):
    raw = run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Medya süresi okunamadı: {path}"
        )


def safe_filename(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))
    return value[:48] or "speaker"


def ass_time(seconds):
    centis = max(0, round(seconds * 100))
    hours, rem = divmod(centis, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, fraction = divmod(rem, 100)

    return (
        f"{hours}:{minutes:02d}:"
        f"{secs:02d}.{fraction:02d}"
    )


def srt_time(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, rem = divmod(milliseconds, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)

    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{secs:02d},{millis:03d}"
    )


def clean_caption(value):
    value = str(value or "")
    value = value.replace("\\", "")
    value = value.replace("{", "")
    value = value.replace("}", "")
    value = value.replace("\n", " ")
    return value.strip()


def write_ass(events, path):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,54,&H00FFFFFF,&H000000FF,&H00101010,&H90000000,-1,0,0,0,100,100,0,0,1,4,1,2,120,120,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []

    for start, end, caption in events:
        caption = clean_caption(caption)
        if not caption:
            continue

        lines.append(
            "Dialogue: 0,"
            + ass_time(start)
            + ","
            + ass_time(max(end, start + 0.25))
            + ",Default,,0,0,0,,"
            + caption
        )

    path.write_text(
        header + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_srt(events, path):
    blocks = []

    for index, (start, end, caption) in enumerate(events, 1):
        caption = clean_caption(caption)
        if not caption:
            continue

        blocks.append(
            f"{index}\n"
            f"{srt_time(start)} --> {srt_time(end)}\n"
            f"{caption}\n"
        )

    path.write_text(
        "\n".join(blocks),
        encoding="utf-8",
    )


async def synthesize_line(text, voice, output):
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "edge-tts kurulu değil. "
            "Önce 'pip install edge-tts' çalıştırılmalı."
        ) from exc

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%",
    )

    await communicate.save(str(output))

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(
            f"Ses dosyası üretilemedi: {output.name}"
        )


def character_voice(character):
    profile = character.get("voice_profile") or {}

    candidates = [
        profile.get("edge"),
        character.get("voice"),
    ]

    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if candidate in {
            "tr-TR-AhmetNeural",
            "tr-TR-EmelNeural",
        }:
            return candidate

    gender = str(
        character.get("voice_gender") or "male"
    ).strip().lower()

    if gender in {"female", "woman", "kadın"}:
        return DEFAULT_VOICES["female"]

    return DEFAULT_VOICES["male"]


def make_silence(path, duration):
    run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
        "-t", f"{duration:.4f}",
        "-c:a", "pcm_s16le",
        str(path),
    ])


def convert_to_wav(source, destination):
    run([
        "ffmpeg", "-y",
        "-i", str(source),
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-c:a", "pcm_s16le",
        str(destination),
    ])


def concat_wav_files(files, output, work):
    if not files:
        raise ValueError("Birleştirilecek ses dosyası bulunamadı.")

    listing = work / f"{safe_filename(output.stem)}_concat.txt"

    listing.write_text(
        "".join(
            "file '" + path.resolve().as_posix() + "'\n"
            for path in files
        ),
        encoding="utf-8",
    )

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(listing),
        "-c:a", "pcm_s16le",
        str(output),
    ])


async def create_scene_audio(
    scene,
    scene_index,
    characters_by_id,
    work,
    subtitle_events,
    timeline,
):
    turns = scene.get("dialogue_turns") or []
    narration = str(scene.get("narration") or "").strip()

    utterances = []

    if narration:
        utterances.append({
            "character": None,
            "text": narration,
        })

    for turn in turns:
        speaker_id = str(turn.get("speaker") or "").strip()
        dialogue = str(turn.get("text") or "").strip()

        if not dialogue:
            continue

        if speaker_id not in characters_by_id:
            raise ValueError(
                f"Sahne {scene_index + 1}: "
                f"bilinmeyen konuşmacı '{speaker_id}'."
            )

        utterances.append({
            "character": characters_by_id[speaker_id],
            "text": dialogue,
        })

    if not utterances:
        silence = work / f"scene_{scene_index + 1:02d}_silence.wav"
        make_silence(silence, SCENE_SECONDS)
        return silence, 0

    parts = []
    cursor = 0.0

    for turn_index, utterance in enumerate(utterances, 1):
        character = utterance["character"]
        dialogue = utterance["text"]

        if character:
            voice = character_voice(character)
            speaker_name = str(
                character.get("name") or "Karakter"
            )
            speaker_id = str(
                character.get("id") or f"speaker_{turn_index}"
            )
        else:
            voice = DEFAULT_VOICES["male"]
            speaker_name = "Anlatıcı"
            speaker_id = "narrator"

        filename = (
            f"scene_{scene_index + 1:02d}_"
            f"line_{turn_index:02d}_"
            f"{safe_filename(speaker_id)}"
        )

        mp3 = work / f"{filename}.mp3"
        wav = work / f"{filename}.wav"

        await synthesize_line(dialogue, voice, mp3)
        convert_to_wav(mp3, wav)

        duration = probe_duration(wav)

        if duration <= 0:
            raise RuntimeError(
                f"Geçersiz ses süresi: {mp3.name}"
            )

        subtitle_events.append((
            timeline + cursor,
            timeline + cursor + duration,
            f"{speaker_name}: {dialogue}",
        ))

        parts.append(wav)
        cursor += duration

        if turn_index < len(utterances):
            gap = work / (
                f"scene_{scene_index + 1:02d}_"
                f"gap_{turn_index:02d}.wav"
            )
            make_silence(gap, LINE_GAP)
            parts.append(gap)
            cursor += LINE_GAP

    if cursor > SCENE_SECONDS:
        raise RuntimeError(
            f"Sahne {scene_index + 1} için ses süresi "
            f"{cursor:.2f} saniye. Mevcut görüntü süresi "
            f"{SCENE_SECONDS:.2f} saniye. Konuşma kesilmesin "
            "diye üretim durduruldu. Diyalog kısaltılmalı "
            "veya sahneye daha uzun görüntü üretilmeli."
        )

    raw_audio = work / f"scene_{scene_index + 1:02d}_raw.wav"
    final_audio = work / f"scene_{scene_index + 1:02d}_audio.wav"

    concat_wav_files(parts, raw_audio, work)

    run([
        "ffmpeg", "-y",
        "-i", str(raw_audio),
        "-af", f"apad=pad_dur={SCENE_SECONDS - cursor:.4f}",
        "-t", f"{SCENE_SECONDS:.4f}",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-c:a", "pcm_s16le",
        str(final_audio),
    ])

    return final_audio, len(utterances)


def escape_filter_path(path):
    value = str(path.resolve()).replace("\\", "/")
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    value = value.replace(",", r"\,")
    value = value.replace("[", r"\[")
    value = value.replace("]", r"\]")
    return value


def build_audio(video, project_path, output, work, music=None):
    require_tools()

    video = Path(video)
    project_path = Path(project_path)
    output = Path(output)
    work = Path(work)

    if not video.is_file():
        raise FileNotFoundError(f"Video bulunamadı: {video}")

    if not project_path.is_file():
        raise FileNotFoundError(
            f"Proje JSON dosyası bulunamadı: {project_path}"
        )

    project = json.loads(
        project_path.read_text(encoding="utf-8")
    )

    characters = project.get("characters") or []
    scenes = project.get("scenes") or []

    if not characters or not scenes:
        raise ValueError(
            "Projede karakterler veya sahneler eksik."
        )

    characters_by_id = {
        str(character.get("id")): character
        for character in characters
        if character.get("id")
    }

    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    subtitle_events = []
    scene_audio_files = []
    timeline = 0.0
    total_dialogues = 0

    async def generate_all_scenes():
        nonlocal timeline, total_dialogues

        for scene_index, scene in enumerate(scenes):
            audio, count = await create_scene_audio(
                scene=scene,
                scene_index=scene_index,
                characters_by_id=characters_by_id,
                work=work,
                subtitle_events=subtitle_events,
                timeline=timeline,
            )

            scene_audio_files.append(audio)
            total_dialogues += count
            timeline += SCENE_SECONDS

    asyncio.run(generate_all_scenes())

    if total_dialogues == 0:
        raise RuntimeError(
            "Projede seslendirilecek diyalog bulunamadı."
        )

    speech_track = work / "complete_speech.wav"

    concat_wav_files(
        scene_audio_files,
        speech_track,
        work,
    )

    expected_duration = len(scenes) * SCENE_SECONDS
    actual_video_duration = probe_duration(video)

    if actual_video_duration + 0.25 < expected_duration:
        raise RuntimeError(
            "Video süresi proje sahne süreleriyle uyuşmuyor. "
            f"Video: {actual_video_duration:.2f} sn, "
            f"beklenen: {expected_duration:.2f} sn. "
            "Ses-video senkronu bozulmaması için işlem durduruldu."
        )

    ass_path = output.with_suffix(".ass")
    srt_path = output.with_suffix(".srt")

    write_ass(subtitle_events, ass_path)
    write_srt(subtitle_events, srt_path)

    command = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(speech_track),
    ]

    filters = [
        "[1:a]volume=1.0[speech]"
    ]

    audio_label = "[speech]"

    music_path = Path(music) if music else None

    if music_path and music_path.is_file():
        command += [
            "-stream_loop", "-1",
            "-i", str(music_path),
        ]

        filters.append(
            f"[2:a]volume=0.12,"
            f"atrim=0:{expected_duration:.3f},"
            "afade=t=in:d=1.5,"
            f"afade=t=out:st={max(0, expected_duration - 2.5):.3f}:d=2.5"
            "[music]"
        )

        filters.append(
            "[speech][music]"
            "amix=inputs=2:duration=first:normalize=0[mix]"
        )

        audio_label = "[mix]"

    filters.append(
        f"[0:v]ass='{escape_filter_path(ass_path)}'[video]"
    )

    command += [
        "-filter_complex",
        ";".join(filters),

        "-map", "[video]",
        "-map", audio_label,

        "-t", f"{expected_duration:.3f}",

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", str(SAMPLE_RATE),

        "-movflags", "+faststart",
        "-shortest",

        str(output),
    ]

    run(command)

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Sesli final video oluşturulamadı.")

    return {
        "output": str(output),
        "subtitle_file": str(srt_path),
        "scene_count": len(scenes),
        "dialogue_turn_count": total_dialogues,
        "audio_duration_seconds": round(
            probe_duration(output), 2
        ),
        "voice_engine": "Edge TTS",
        "music_added": bool(
            music_path and music_path.is_file()
        ),
        "subtitles_burned_in": True,
        "separate_srt_created": True,
        "lip_sync": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Wan video için Türkçe ses ve altyazı üretimi."
    )

    parser.add_argument(
        "--video",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--project",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--work",
        default=Path("audio_work"),
        type=Path,
    )
    parser.add_argument(
        "--music",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    report = build_audio(
        video=args.video,
        project_path=args.project,
        output=args.output,
        work=args.work,
        music=args.music,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
