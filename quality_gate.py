
"""Production quality gates for generated film and motion videos.

The original motion checks are preserved. This module additionally
provides dependency-light FFprobe validation for final MP4 files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional


def require_real_motion(mode, fal_key_present):
    mode = (mode or "real").strip().lower()

    if mode not in {"real", "preview"}:
        raise ValueError("MOTION_MODE must be 'real' or 'preview'.")

    if mode == "real" and not fal_key_present:
        raise RuntimeError(
            "Real motion requires a configured video-generation provider. "
            "A still image with zoom or pan is not real character animation. "
            "Configure FAL_KEY or explicitly select preview mode."
        )

    return mode == "real"


def validate_motion_outputs(paths, mode="real"):
    mode = (mode or "real").strip().lower()

    if mode not in {"real", "preview"}:
        raise ValueError("MOTION_MODE must be 'real' or 'preview'.")

    paths = list(paths)

    existing = [
        str(path)
        for path in paths
        if Path(path).is_file() and Path(path).stat().st_size > 0
    ]

    if mode == "real" and len(existing) != len(paths):
        raise RuntimeError(
            f"Expected {len(paths)} valid motion clips, "
            f"but found {len(existing)}."
        )

    return existing


def _require_ffprobe():
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError(
            "ffprobe bulunamadı. Final video kalite kontrolü için "
            "FFmpeg/FFprobe kurulmalıdır."
        )
    return executable


def _parse_frame_rate(value: Any) -> float:
    """Parse FFprobe frame-rate values such as 16/1 or 30000/1001."""
    if value in (None, "", "0/0", "N/A"):
        return 0.0

    try:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator = float(denominator)
            if denominator == 0:
                return 0.0
            return float(numerator) / denominator

        return float(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def inspect_media(path: str | Path) -> dict[str, Any]:
    """Return verified media metadata using FFprobe."""
    media_path = Path(path)

    if not media_path.is_file():
        raise RuntimeError(f"Medya dosyası bulunamadı: {media_path}")

    file_size = media_path.stat().st_size
    if file_size <= 0:
        raise RuntimeError(f"Medya dosyası boş: {media_path}")

    ffprobe = _require_ffprobe()

    command = [
        ffprobe,
        "-v", "error",
        "-show_entries",
        (
            "format=duration,format_name,size:"
            "stream=codec_type,codec_name,width,height,"
            "r_frame_rate,avg_frame_rate,sample_rate,channels"
        ),
        "-of", "json",
        str(media_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFprobe medya dosyasını okuyamadı: "
            + (result.stderr.strip()[-2000:] or "Bilinmeyen FFprobe hatası")
        )

    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"FFprobe çıktısı geçerli JSON değil: {exc}"
        ) from exc

    format_info = metadata.get("format") or {}
    streams = metadata.get("streams") or []

    video_streams = [
        stream for stream in streams
        if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in streams
        if stream.get("codec_type") == "audio"
    ]

    if not video_streams:
        raise RuntimeError("Dosyada video akışı bulunamadı.")

    video = video_streams[0]
    audio = audio_streams[0] if audio_streams else None

    try:
        duration = float(format_info.get("duration", 0))
    except (TypeError, ValueError, OverflowError):
        duration = 0.0

    if duration <= 0:
        raise RuntimeError("Video süresi geçersiz veya okunamıyor.")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)

    if width <= 0 or height <= 0:
        raise RuntimeError("Video çözünürlüğü okunamadı.")

    frame_rate = _parse_frame_rate(
        video.get("avg_frame_rate") or video.get("r_frame_rate")
    )

    return {
        "path": str(media_path.resolve()),
        "file_size_bytes": file_size,
        "duration_seconds": round(duration, 3),
        "container": format_info.get("format_name"),
        "video": {
            "codec": video.get("codec_name"),
            "width": width,
            "height": height,
            "fps": round(frame_rate, 3),
        },
        "audio": {
            "present": audio is not None,
            "codec": audio.get("codec_name") if audio else None,
            "sample_rate": audio.get("sample_rate") if audio else None,
            "channels": audio.get("channels") if audio else None,
        },
    }


def validate_final_video(
    path: str | Path,
    *,
    require_audio: bool = True,
    min_width: int = 640,
    min_height: int = 360,
    min_fps: float = 10.0,
    expected_duration: Optional[float] = None,
    duration_tolerance: Optional[float] = None,
) -> dict[str, Any]:
    """Validate a final film file and return a machine-readable report.

    This checks technical integrity, not subjective artistic quality.
    It does not claim to verify story quality, visual consistency,
    lip synchronization, or whether dialogue sounds natural.
    """
    metadata = inspect_media(path)
    errors: list[str] = []
    warnings: list[str] = []

    video = metadata["video"]
    audio = metadata["audio"]

    if video["width"] < int(min_width):
        errors.append(
            f"Çözünürlük genişliği yetersiz: {video['width']} < {min_width}"
        )

    if video["height"] < int(min_height):
        errors.append(
            f"Çözünürlük yüksekliği yetersiz: {video['height']} < {min_height}"
        )

    if video["fps"] < float(min_fps):
        errors.append(
            f"FPS yetersiz veya okunamadı: {video['fps']} < {min_fps}"
        )

    if require_audio and not audio["present"]:
        errors.append("Final MP4 dosyasında ses akışı bulunamadı.")

    if audio["present"] and not audio["codec"]:
        errors.append("Ses akışının codec bilgisi okunamadı.")

    if expected_duration is not None:
        expected = float(expected_duration)

        if expected <= 0:
            raise ValueError("expected_duration pozitif olmalıdır.")

        tolerance = (
            float(duration_tolerance)
            if duration_tolerance is not None
            else max(2.0, expected * 0.05)
        )

        difference = abs(metadata["duration_seconds"] - expected)

        if difference > tolerance:
            errors.append(
                "Video süresi hedefle uyuşmuyor: "
                f"gerçek={metadata['duration_seconds']:.2f}s, "
                f"hedef={expected:.2f}s, "
                f"tolerans={tolerance:.2f}s"
            )

    if video["width"] < 832 or video["height"] < 480:
        warnings.append(
            "Video 832x480 hedef çözünürlüğünün altında."
        )

    if video["fps"] < 24:
        warnings.append(
            "Video 24 FPS altında. Bu teknik kontrol sinematik "
            "akışkanlığı garanti etmez."
        )

    report = {
        "status": "passed" if not errors else "failed",
        "valid": not errors,
        "file": metadata,
        "checks": {
            "video_stream": True,
            "audio_stream": audio["present"],
            "resolution": (
                video["width"] >= int(min_width)
                and video["height"] >= int(min_height)
            ),
            "frame_rate": video["fps"] >= float(min_fps),
            "duration": not any("Video süresi" in item for item in errors),
        },
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            "Bu kontrol görsel sanat kalitesini ölçmez.",
            "Karakter yüzü ve kıyafet tutarlılığını doğrulamaz.",
            "Dudak senkronizasyonunun doğruluğunu ölçmez.",
            "Diyalogların anlaşılabilirliğini veya ses kalitesini dinleyerek değerlendirmez.",
            "Kamera hareketleri ve görsel efektlerin estetik başarısını garanti etmez.",
        ],
    }

    if errors:
        raise RuntimeError(
            "Final video kalite kontrolünden geçemedi: "
            + " | ".join(errors)
        )

    return report

