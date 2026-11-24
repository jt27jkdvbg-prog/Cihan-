
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuseTalk Film Compositor v1.0

MuseTalk tarafından üretilen konuşma kliplerini,
orijinal filmin doğru zaman aralıklarına yerleştirir.

- Orijinal video süresini korur.
- Orijinal filmin sesini korur.
- Sadece doğrulanmış MuseTalk kliplerini kullanır.
- Diyalog kliplerini kendi zaman damgalarında görüntüye bindirir.
- İşlem raporu oluşturur.

Bu dosya yüz tespiti veya konuşmacı eşleştirmesi yapmaz.
Bu işlemler önceki aşamada onaylanmış olmalıdır.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time

from pathlib import Path
from typing import Any


VERSION = "1.0.0"


def log(message: str) -> None:
    print(
        f"[MuseTalk Compositor] {message}",
        flush=True,
    )


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        fail(f"Gerekli program bulunamadı: {name}")


def run(command: list[str]) -> subprocess.CompletedProcess:

    log("Komut çalıştırılıyor:")
    log(" ".join(command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.stdout:
        print(
            result.stdout[-10000:],
            flush=True,
        )

    if result.returncode != 0:
        fail(
            "FFmpeg işlemi başarısız. "
            f"Çıkış kodu: {result.returncode}"
        )

    return result


def read_json(path: Path) -> dict[str, Any]:

    if not path.is_file():
        fail(f"JSON dosyası bulunamadı: {path}")

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        fail(f"JSON okunamadı: {exc}")

    if not isinstance(data, dict):
        fail("JSON kök verisi nesne olmalıdır.")

    return data


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:

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


def resolve_file(
    value: str,
    report_path: Path,
) -> Path:

    path = Path(value)

    if path.is_absolute() and path.is_file():
        return path.resolve()

    candidates = [
        Path.cwd() / path,
        report_path.parent / path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    fail(
        "Gerekli video dosyası bulunamadı: "
        + value
    )


def probe_media(path: Path) -> dict[str, Any]:

    result = run([
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=codec_type,width,height,codec_name",
        "-of", "json",
        str(path),
    ])

    try:
        data = json.loads(result.stdout)
    except Exception:
        fail(f"Medya bilgileri okunamadı: {path}")

    duration = float(
        data.get("format", {}).get(
            "duration", 0
        )
    )

    streams = data.get("streams") or []

    video_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
        ),
        None,
    )

    if video_stream is None:
        fail(f"Video akışı bulunamadı: {path}")

    audio_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "audio"
        ),
        None,
    )

    if duration <= 0:
        fail(f"Video süresi geçersiz: {path}")

    return {
        "duration_seconds": duration,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "video_codec": video_stream.get("codec_name"),
        "audio_present": audio_stream is not None,
        "audio_codec": (
            audio_stream.get("codec_name")
            if audio_stream
            else None
        ),
    }


def validate_report(
    report: dict[str, Any],
    report_path: Path,
) -> list[dict[str, Any]]:

    if report.get("status") != "dialogue_clips_generated":
        fail(
            "MuseTalk raporunun durumu "
            "dialogue_clips_generated değil."
        )

    if report.get("full_film_composited") is True:
        fail("Bu rapor zaten birleştirilmiş olarak işaretli.")

    tasks = report.get("dialogue_tasks")
    generated = report.get("generated_videos")

    if not isinstance(tasks, list) or not tasks:
        fail("Diyalog görev listesi boş.")

    if not isinstance(generated, list) or not generated:
        fail("MuseTalk çıktı listesi boş.")

    generated_by_index = {}

    for item in generated:

        if not isinstance(item, dict):
            fail("MuseTalk çıktı kaydı geçersiz.")

        index = item.get("dialogue_index")

        if index is None:
            fail("Çıktıda dialogue_index eksik.")

        if index in generated_by_index:
            fail(
                f"Tekrarlanan diyalog çıktısı: {index}"
            )

        generated_by_index[index] = item

    validated = []

    for task in tasks:

        if not isinstance(task, dict):
            fail("Diyalog görev kaydı geçersiz.")

        index = task.get("dialogue_index")

        if index not in generated_by_index:
            fail(
                f"Diyalog {index} için MuseTalk çıktısı yok."
            )

        start = float(
            task["film_start_seconds"]
        )

        end = float(
            task["film_end_seconds"]
        )

        if start < 0 or end <= start:
            fail(
                f"Diyalog {index}: zaman aralığı geçersiz."
            )

        output_record = generated_by_index[index]

        output_path = resolve_file(
            str(output_record["file"]),
            report_path,
        )

        if output_path.stat().st_size <= 0:
            fail(
                f"Diyalog {index}: çıktı dosyası boş."
            )

        validated.append({
            "dialogue_index": index,
            "speaker_id": task.get("speaker_id"),
            "start": start,
            "end": end,
            "output": output_path,
        })

    validated.sort(
        key=lambda item: (
            item["start"],
            item["dialogue_index"],
        )
    )

    return validated


def build_filter_graph(
    clips: list[dict[str, Any]],
) -> str:

    graph = [
        "[0:v]setpts=PTS-STARTPTS[base0]"
    ]

    previous = "base0"

    for index, clip in enumerate(clips):

        input_number = index + 1
        next_label = f"base{index + 1}"

        start = f"{clip['start']:.6f}"
        end = f"{clip['end']:.6f}"

        graph.append(
            f"[{input_number}:v]"
            "setpts=PTS-STARTPTS"
            f"[overlay{index}]"
        )

        graph.append(
            f"[{previous}]"
            f"[overlay{index}]"
            "overlay="
            "x=0:y=0:"
            "eof_action=pass:"
            "repeatlast=0:"
            "shortest=0:"
            f"enable='between(t,{start},{end})'"
            f"[{next_label}]"
        )

        previous = next_label

    graph.append(
        f"[{previous}]format=yuv420p[vout]"
    )

    return ";".join(graph)


def compose(
    source_video: Path,
    report_path: Path,
    output_video: Path,
) -> dict[str, Any]:

    started = time.time()

    require_binary("ffmpeg")
    require_binary("ffprobe")

    source_video = source_video.resolve()
    report_path = report_path.resolve()
    output_video = output_video.resolve()

    if not source_video.is_file():
        fail(f"Orijinal film bulunamadı: {source_video}")

    if not report_path.is_file():
        fail(f"MuseTalk raporu bulunamadı: {report_path}")

    if output_video.exists():
        fail(
            "Çıktı dosyası zaten mevcut. "
            "Farklı bir dosya adı kullan: "
            + str(output_video)
        )

    report = read_json(report_path)

    clips = validate_report(
        report,
        report_path,
    )

    source_info = probe_media(source_video)

    duration = source_info["duration_seconds"]

    for clip in clips:

        if clip["end"] > duration + 0.25:
            fail(
                f"Diyalog {clip['dialogue_index']} "
                "orijinal filmin süresini aşıyor."
            )

        clip_info = probe_media(clip["output"])

        if (
            clip_info["width"] != source_info["width"]
            or clip_info["height"] != source_info["height"]
        ):
            fail(
                f"Diyalog {clip['dialogue_index']}: "
                "MuseTalk klibinin çözünürlüğü "
                "ana filmle eşleşmiyor. "
                "Otomatik ölçeklendirme yapılmadı."
            )

        clip["output_duration"] = (
            clip_info["duration_seconds"]
        )

        if (
            clip["output_duration"] < 0.1
            or clip["output_duration"]
            > (clip["end"] - clip["start"]) + 1.5
        ):
            fail(
                f"Diyalog {clip['dialogue_index']}: "
                "çıktı süresi beklenen aralıkla uyuşmuyor."
            )

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source_video),
    ]

    for clip in clips:
        command.extend([
            "-i",
            str(clip["output"]),
        ])

    graph = build_filter_graph(clips)

    command.extend([
        "-filter_complex",
        graph,
        "-map",
        "[vout]",
        "-map",
        "0:a?",
        "-map_metadata",
        "0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration:.4f}",
        "-movflags",
        "+faststart",
        str(output_video),
    ])

    run(command)

    if not output_video.is_file():
        fail("Birleştirilmiş film oluşturulamadı.")

    if output_video.stat().st_size <= 0:
        fail("Birleştirilmiş film dosyası boş.")

    final_info = probe_media(output_video)

    duration_difference = abs(
        final_info["duration_seconds"] - duration
    )

    if duration_difference > 1.0:
        fail(
            "Son filmin süresi orijinal filmle "
            "uyuşmuyor. "
            f"Orijinal: {duration:.2f}s, "
            f"sonuç: {final_info['duration_seconds']:.2f}s"
        )

    result = {
        "compositor_version": VERSION,
        "status": "full_film_composited",
        "lip_sync_rendered": True,
        "full_film_composited": True,
        "source_video": str(source_video),
        "output_video": str(output_video),
        "original_video": source_info,
        "final_video": final_info,
        "composited_dialogue_count": len(clips),
        "composited_dialogue": [
            {
                "dialogue_index": clip["dialogue_index"],
                "speaker_id": clip["speaker_id"],
                "film_start_seconds": clip["start"],
                "film_end_seconds": clip["end"],
                "source_clip": str(clip["output"]),
            }
            for clip in clips
        ],
        "original_audio_preserved": True,
        "duration_difference_seconds": round(
            duration_difference, 4
        ),
        "elapsed_seconds": round(
            time.time() - started, 2
        ),
        "note": (
            "Approved MuseTalk dialogue clips were "
            "composited onto the original film. "
            "Original film audio was retained. "
            "Skipped dialogue entries remain unchanged."
        ),
    }

    output_report = output_video.with_name(
        output_video.stem + "_compositor_report.json"
    )

    write_json(
        output_report,
        result,
    )

    log("Film birleştirme işlemi tamamlandı.")

    log(
        f"İşlenen diyalog sayısı: {len(clips)}"
    )

    log(
        f"Son film: {output_video}"
    )

    return result


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "MuseTalk diyalog kliplerini ana filme "
            "yerleştirir."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Orijinal sesli film.",
    )

    parser.add_argument(
        "--report",
        required=True,
        help="musetalk_report.json dosyası.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Oluşturulacak son film.",
    )

    args = parser.parse_args()

    try:

        result = compose(
            source_video=Path(args.video),
            report_path=Path(args.report),
            output_video=Path(args.output),
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    except Exception as exc:

        log("HATA: " + str(exc))

        raise SystemExit(1)


if __name__ == "__main__":
    main()
