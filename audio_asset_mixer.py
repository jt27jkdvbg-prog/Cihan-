
import json
import shutil
import subprocess
from pathlib import Path


def _duration(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("Geçersiz film süresi.")
    if value <= 0:
        raise ValueError("Film süresi sıfırdan büyük olmalı.")
    return value


def _asset(asset, name):
    if not isinstance(asset, dict):
        raise ValueError(f"{name}: ses dosyası bilgisi geçersiz.")

    if asset.get("license_verified") is not True:
        raise ValueError(
            f"{name}: kullanım lisansı doğrulanmamış."
        )

    path = Path(str(asset.get("path", "")))

    if not path.is_file():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {path}")

    try:
        volume = float(asset.get("volume", 0.2))
    except (TypeError, ValueError):
        raise ValueError(f"{name}: ses seviyesi geçersiz.")

    if not 0 <= volume <= 1:
        raise ValueError(f"{name}: ses seviyesi 0-1 arasında olmalı.")

    return path, volume


def mix_audio_layers(
    voice_path,
    output_path,
    duration,
    music=None,
    ambience=None,
    effects=None,
    sample_rate=48000,
    loudness_lufs=-16.0,
    true_peak_db=-1.5,
):
    """Konuşma, müzik, ortam ve efektleri tek ses dosyasında birleştirir."""

    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg bulunamadı.")

    duration = _duration(duration)
    voice = Path(voice_path)

    if not voice.is_file():
        raise FileNotFoundError(f"Konuşma sesi bulunamadı: {voice}")

    if sample_rate not in (44100, 48000):
        raise ValueError("Desteklenmeyen örnekleme frekansı.")

    effects = effects or []

    if not isinstance(effects, list):
        raise ValueError("Efekt listesi geçersiz.")

    assets = []

    if music is not None:
        assets.append(("music", music, 0))

    if ambience is not None:
        assets.append(("ambience", ambience, 0))

    for index, effect in enumerate(effects):
        assets.append((f"effect_{index}", effect, index))

    validated = []

    for name, item, _ in assets:
        path, volume = _asset(item, name)

        offset = 0.0

        if name.startswith("effect_"):
            try:
                offset = float(item.get("offset", 0))
            except (TypeError, ValueError):
                raise ValueError(f"{name}: efekt başlangıç zamanı geçersiz.")

            if not 0 <= offset < duration:
                raise ValueError(f"{name}: efekt film süresinin dışında.")

        validated.append((name, path, volume, offset))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    command = ["ffmpeg", "-y", "-i", str(voice)]
    indexes = {}

    for name, path, _, _ in validated:
        if name in ("music", "ambience"):
            command.extend(["-stream_loop", "-1"])

        command.extend(["-i", str(path)])
        indexes[name] = len(indexes) + 1

    filters = [
        f"[0:a]aresample={sample_rate},"
        f"aformat=channel_layouts=stereo,"
        f"apad,atrim=0:{duration:.6f},"
        "asetpts=PTS-STARTPTS[voice]"
    ]

    mix_inputs = ["[voice]"]
    has_music = False

    for name, path, volume, offset in validated:
        index = indexes[name]
        label = name.replace("-", "_")

        if name == "music":
            filters.append(
                f"[{index}:a]aresample={sample_rate},"
                f"aformat=channel_layouts=stereo,"
                f"volume={volume},"
                f"atrim=0:{duration:.6f},"
                f"asetpts=PTS-STARTPTS[music]"
            )
            has_music = True

        elif name == "ambience":
            filters.append(
                f"[{index}:a]aresample={sample_rate},"
                f"aformat=channel_layouts=stereo,"
                f"volume={volume},"
                f"atrim=0:{duration:.6f},"
                f"asetpts=PTS-STARTPTS[ambience]"
            )
            mix_inputs.append("[ambience]")

        else:
            delay = int(offset * 1000)

            filters.append(
                f"[{index}:a]aresample={sample_rate},"
                f"aformat=channel_layouts=stereo,"
                f"volume={volume},"
                f"adelay={delay}|{delay},"
                f"apad,atrim=0:{duration:.6f},"
                f"asetpts=PTS-STARTPTS[{label}]"
            )

            mix_inputs.append(f"[{label}]")

    if has_music:
        filters.append(
            "[music][voice]sidechaincompress="
            "threshold=0.025:ratio=8:attack=20:release=450"
            "[ducked_music]"
        )
        mix_inputs.append("[ducked_music]")

    filters.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:"
        "duration=longest:dropout_transition=2,"
        + f"loudnorm=I={loudness_lufs}:TP={true_peak_db}:LRA=11,"
        + f"atrim=0:{duration:.6f},"
        + f"aformat=sample_rates={sample_rate}:"
        "channel_layouts=stereo[mix]"
    )

    command.extend([
        "-filter_complex",
        ";".join(filters),
        "-map", "[mix]",
        "-c:a", "aac",
        "-b:a", "256k",
        "-ar", str(sample_rate),
        "-ac", "2",
        "-t", str(duration),
        str(output),
    ])

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Ses miksajı oluşturulamadı.")

    return {
        "output": str(output),
        "duration_seconds": duration,
        "layers": ["dialogue"] + [
            item[0] for item in validated
        ],
        "music_ducking": has_music,
        "licence_verified": True,
    }


def write_mix_report(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
