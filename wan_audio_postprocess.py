from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

FPS = 16
SCENE_SECONDS = 81 / FPS

DEFAULT_VOICES = {
    "male": "tr-TR-AhmetNeural",
    "female": "tr-TR-EmelNeural",
}


def run(command):
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def probe_duration(path):
    raw = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    return float(raw)


def safe_filename(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
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
        clean = caption.replace("\\", "")
        clean = clean.replace("{", "").replace("}", "")
        clean = clean.replace("\n", " ").strip()

        if clean:
            lines.append(
                f"Dialogue: 0,{ass_time(start)},"
                f"{ass_time(max(end, start + .25))},"
                f"Default,,0,0,0,,{clean}"
            )

    path.write_text(
        header + "\n".join(lines) + "\n",
        encoding="utf-8"
    )


async def synthesize_line(text, voice, output):
    import edge_tts

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        volume="+0%"
    )

    await communicate.save(str(output))


def character_voice(character):
    profile = character.get("voice_profile") or {}

    chosen = str(profile.get("edge") or "").strip()

    if chosen:
        return chosen

    gender = str(
        character.get("voice_gender", "male")
    ).lower()

    return DEFAULT_VOICES[
        "female" if gender == "female" else "male"
    ]


def build_audio(video, project_path, output, work, music=None):

    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg bulunamadı.")

    if not shutil.which("ffprobe"):
        raise RuntimeError("FFprobe bulunamadı.")

    project = json.loads(
        project_path.read_text(encoding="utf-8")
    )

    characters = project.get("characters") or []
    scenes = project.get("scenes") or []

    if not scenes or not characters:
        raise ValueError(
            "Projede karakterler ve sahneler bulunamadı."
        )

    by_id = {
        str(c.get("id")): c
        for c in characters
    }

    work.mkdir(parents=True, exist_ok=True)

    segment_paths = []
    subtitle_events = []
    timeline = 0.0

    async def make_segments():
        nonlocal timeline

        for scene_index, scene in enumerate(scenes):

            line_files = []
            scene_cursor = 0.0

            turns = scene.get("dialogue_turns") or []
            narration = str(
                scene.get("narration") or ""
            ).strip()

            utterances = []

            if narration:
                utterances.append((None, narration))

            for turn in turns:

                speaker_id = str(
                    turn.get("speaker") or ""
                ).strip()

                text = str(
                    turn.get("text") or ""
                ).strip()

                if not text:
                    continue

                if speaker_id not in by_id:
                    raise ValueError(
                        f"Bilinmeyen konuşmacı: {speaker_id}"
                    )

                utterances.append((
                    by_id[speaker_id],
                    text
                ))

            for turn_index, (character, text) in enumerate(
                utterances
            ):

                voice = (
                    character_voice(character)
                    if character
                    else DEFAULT_VOICES["male"]
                )

                name = (
                    safe_filename(character.get("id", "narrator"))
                    if character
                    else "narrator"
                )

                mp3 = work / (
                    f"s{scene_index+1:02d}_"
                    f"t{turn_index+1:02d}_{name}.mp3"
                )

                await synthesize_line(
                    text,
                    voice,
                    mp3
                )

                line_duration = probe_duration(mp3)

                line_files.append(mp3)

                speaker_name = (
                    character.get("name")
                    if character
                    else "Anlatıcı"
                )

                subtitle_events.append((
                    timeline + scene_cursor,
                    timeline + scene_cursor + line_duration,
                    f"{speaker_name}: {text}"
                ))

                scene_cursor += line_duration + 0.18

            scene_audio = work / (
                f"scene_audio_{scene_index+1:02d}.wav"
            )

            if line_files:

                concat = work / (
                    f"scene_concat_{scene_index+1:02d}.txt"
                )

                concat.write_text(
                    "".join(
                        f"file '{p.resolve().as_posix()}'\n"
                        for p in line_files
                    ),
                    encoding="utf-8"
                )

                raw = work / (
                    f"scene_raw_{scene_index+1:02d}.wav"
                )

                run([
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat),
                    "-ar", "44100",
                    "-ac", "2",
                    str(raw)
                ])

                raw_duration = probe_duration(raw)

                if raw_duration > SCENE_SECONDS * 1.18:
                    raise RuntimeError(
                        f"{scene_index+1}. sahnenin konuşması "
                        "sahne süresine sığmıyor. "
                        "Diyalog kısaltılmalı."
                    )

                run([
                    "ffmpeg", "-y",
                    "-i", str(raw),
                    "-af",
                    f"apad=pad_dur={SCENE_SECONDS:.3f}",
                    "-t", f"{SCENE_SECONDS:.3f}",
                    "-ar", "44100",
                    "-ac", "2",
                    str(scene_audio)
                ])

            else:

                run([
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", f"{SCENE_SECONDS:.3f}",
                    str(scene_audio)
                ])

            segment_paths.append(scene_audio)
            timeline += SCENE_SECONDS

    asyncio.run(make_segments())

    concat_all = work / "all_speech.txt"

    concat_all.write_text(
        "".join(
            f"file '{p.resolve().as_posix()}'\n"
            for p in segment_paths
        ),
        encoding="utf-8"
    )

    speech_track = work / "speech.wav"

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_all),
        "-c:a", "pcm_s16le",
        str(speech_track)
    ])

    subtitles = work / "dialogue.ass"

    write_ass(
        subtitle_events,
        subtitles
    )

    command = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(speech_track)
    ]

    filters = [
        "[1:a]volume=1.0[speech]"
    ]

    if music and music.is_file():

        command += [
            "-stream_loop", "-1",
            "-i", str(music)
        ]

        filters.append(
            f"[2:a]volume=0.12,"
            f"atrim=0:{timeline:.3f},"
            "afade=t=in:d=1.5,"
            f"afade=t=out:st={max(0, timeline-2.5):.3f}:d=2.5"
            "[music]"
        )

        filters.append(
            "[speech][music]"
            "amix=inputs=2:duration=first:normalize=0[mix]"
        )

        audio_label = "[mix]"

    else:
        audio_label = "[speech]"

    filters.append(
        f"[0:v]ass='{subtitles.as_posix()}'[video]"
    )

    command += [
        "-filter_complex",
        ";".join(filters),

        "-map", "[video]",
        "-map", audio_label,

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",

        "-movflags", "+faststart",
        "-shortest",

        str(output)
    ]

    run(command)

    return {
        "output": str(output),
        "scene_count": len(scenes),
        "dialogue_turn_count": sum(
            len(s.get("dialogue_turns") or [])
            for s in scenes
        ),
        "audio_duration_seconds": round(
            probe_duration(output), 2
        ),
        "voice_engine": "Edge TTS",
        "music_added": bool(music and music.is_file())
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", default=Path("audio_work"), type=Path)
    parser.add_argument("--music", type=Path, default=None)

    args = parser.parse_args()

    report = build_audio(
        args.video,
        args.project,
        args.output,
        args.work,
        args.music
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
