
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a validated MuseTalk lip-sync input package.

Input:
- Silent source video
- lip_sync_timeline.json
- Individual dialogue audio segments

Output:
- video.mp4
- audio/segment_XXXX.mp3
- lip_sync_timeline.json (portable relative audio paths)
- manifest.json
- musetalk_input.zip

This script prepares the data package only.
It does not run MuseTalk inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


def run_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Komut başarısız:\n"
            + " ".join(command)
            + "\n"
            + result.stderr[-3000:]
        )

    return result.stdout.strip()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Gerekli program bulunamadı: {name}"
        )


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"JSON okunamadı: {path}: {exc}"
        ) from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def probe_video(video_path: Path) -> dict:
    output = run_command([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of", "json",
        str(video_path),
    ])

    data = json.loads(output)

    streams = data.get("streams") or []

    if not streams:
        raise RuntimeError(
            "Video akışı bulunamadı."
        )

    stream = streams[0]

    duration = float(
        data.get("format", {}).get(
            "duration", 0
        )
    )

    if duration <= 0:
        raise RuntimeError(
            "Video süresi geçersiz."
        )

    return {
        "duration_seconds": round(duration, 4),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "frame_rate": stream.get(
            "r_frame_rate", ""
        ),
    }


def resolve_audio_path(
    value: str,
    timeline_path: Path,
) -> Path:

    path = Path(value)

    if path.is_absolute():
        return path.resolve()

    candidates = [
        timeline_path.parent / path,
        Path.cwd() / path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Diyalog ses dosyası bulunamadı: "
        + value
    )


def validate_entry(
    entry: dict,
    index: int,
    video_duration: float,
) -> None:

    required = (
        "speaker_id",
        "audio_file",
        "film_start_seconds",
        "film_end_seconds",
    )

    for field in required:
        if field not in entry:
            raise RuntimeError(
                f"Diyalog {index}: "
                f"{field} alanı eksik."
            )

    speaker = str(
        entry.get("speaker_id", "")
    ).strip()

    if not speaker:
        raise RuntimeError(
            f"Diyalog {index}: speaker_id boş."
        )

    try:
        start = float(
            entry["film_start_seconds"]
        )

        end = float(
            entry["film_end_seconds"]
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        raise RuntimeError(
            f"Diyalog {index}: zaman bilgisi geçersiz."
        )

    if start < 0 or end <= start:
        raise RuntimeError(
            f"Diyalog {index}: zaman aralığı geçersiz."
        )

    if end > video_duration + 0.25:
        raise RuntimeError(
            f"Diyalog {index}: "
            "zaman aralığı video süresini aşıyor."
        )

    if not str(entry.get("text", "")).strip():
        raise RuntimeError(
            f"Diyalog {index}: konuşma metni boş."
        )


def prepare_package(
    video_path: Path,
    timeline_path: Path,
    output_dir: Path,
) -> dict:

    require_binary("ffmpeg")
    require_binary("ffprobe")

    video_path = video_path.resolve()
    timeline_path = timeline_path.resolve()
    output_dir = output_dir.resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video bulunamadı: {video_path}"
        )

    if not timeline_path.is_file():
        raise FileNotFoundError(
            f"Dudak senkronu zaman çizelgesi bulunamadı: "
            f"{timeline_path}"
        )

    if output_dir.exists():
        raise FileExistsError(
            "Çıktı klasörü zaten mevcut. "
            "Başka bir klasör adı kullanın: "
            + str(output_dir)
        )

    timeline = read_json(timeline_path)

    if not isinstance(timeline, dict):
        raise RuntimeError(
            "Zaman çizelgesi JSON nesnesi olmalıdır."
        )

    if timeline.get("lip_sync_rendered") is True:
        raise RuntimeError(
            "Bu zaman çizelgesi zaten işlenmiş olarak işaretli."
        )

    entries = timeline.get("dialogue")

    if not isinstance(entries, list) or not entries:
        raise RuntimeError(
            "Zaman çizelgesinde konuşmacı diyaloğu bulunamadı."
        )

    video_info = probe_video(video_path)

    video_duration = video_info[
        "duration_seconds"
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    audio_dir = output_dir / "audio"

    audio_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        video_path,
        output_dir / "video.mp4",
    )

    packaged_entries = []
    copied_audio = []

    for index, original in enumerate(entries):

        if not isinstance(original, dict):
            raise RuntimeError(
                f"Diyalog {index}: kayıt geçersiz."
            )

        validate_entry(
            original,
            index,
            video_duration,
        )

        source_audio = resolve_audio_path(
            str(original["audio_file"]),
            timeline_path,
        )

        if source_audio.stat().st_size == 0:
            raise RuntimeError(
                f"Ses dosyası boş: {source_audio}"
            )

        destination_name = (
            f"segment_{index:04d}"
            + source_audio.suffix.lower()
        )

        destination_audio = (
            audio_dir / destination_name
        )

        shutil.copy2(
            source_audio,
            destination_audio,
        )

        entry = dict(original)

        entry["audio_file"] = (
            f"audio/{destination_name}"
        )

        entry["package_audio_sha256"] = (
            sha256_file(destination_audio)
        )

        entry["package_audio_size_bytes"] = (
            destination_audio.stat().st_size
        )

        packaged_entries.append(entry)

        copied_audio.append({
            "file": f"audio/{destination_name}",
            "speaker_id": entry["speaker_id"],
            "sha256": entry[
                "package_audio_sha256"
            ],
        })

    portable_timeline = dict(timeline)

    portable_timeline.update({
        "schema_version": SCHEMA_VERSION,
        "status": "package_ready",
        "lip_sync_rendered": False,
        "description": (
            "Portable MuseTalk input timeline. "
            "This is not a rendered lip-sync result."
        ),
        "source_video": "video.mp4",
        "video_duration_seconds": video_duration,
        "dialogue": packaged_entries,
    })

    portable_timeline["processing_requirements"] = {
        "requires_video_face_tracking": True,
        "requires_speaker_to_face_matching": True,
        "requires_gpu_video_inference": True,
        "skip_ambiguous_face_matches": True,
        "preserve_original_audio_timing": True,
        "preserve_original_video_duration": True,
    }

    write_json(
        output_dir / "lip_sync_timeline.json",
        portable_timeline,
    )

    manifest = {
        "package_schema_version": SCHEMA_VERSION,
        "status": "ready_for_musetalk",
        "inference_completed": False,
        "video": {
            "file": "video.mp4",
            **video_info,
            "sha256": sha256_file(
                output_dir / "video.mp4"
            ),
        },
        "dialogue_count": len(packaged_entries),
        "audio_count": len(copied_audio),
        "audio_files": copied_audio,
        "speaker_ids": sorted({
            str(item["speaker_id"])
            for item in packaged_entries
        }),
        "timeline_file": "lip_sync_timeline.json",
        "important_note": (
            "Package preparation only. "
            "MuseTalk inference has not been executed."
        ),
    }

    write_json(
        output_dir / "manifest.json",
        manifest,
    )

    zip_path = output_dir.parent / (
        output_dir.name + ".zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for file_path in sorted(
            output_dir.rglob("*")
        ):
            if file_path.is_file():
                archive.write(
                    file_path,
                    file_path.relative_to(
                        output_dir
                    ).as_posix(),
                )

    result = {
        "status": "success",
        "package_directory": str(output_dir),
        "zip_file": str(zip_path),
        "dialogue_count": len(packaged_entries),
        "speaker_count": len(
            manifest["speaker_ids"]
        ),
        "video_duration_seconds": video_duration,
    }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    return result


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare a validated MuseTalk "
            "lip-sync input package."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Sessiz kaynak video dosyası.",
    )

    parser.add_argument(
        "--timeline",
        required=True,
        help="lip_sync_timeline.json dosyası.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Paket çıktı klasörü.",
    )

    args = parser.parse_args()

    try:
        result = prepare_package(
            Path(args.video),
            Path(args.timeline),
            Path(args.output_dir),
        )

        if result.get("status") != "success":
            raise RuntimeError(
                "Paket hazırlama başarısız."
            )

    except Exception as exc:
        print(
            f"HATA: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
