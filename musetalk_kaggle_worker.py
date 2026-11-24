
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuseTalk 1.5 Kaggle GPU Worker - v1.1

Her diyalog için:
1. Ana videodan doğru zaman aralığını keser.
2. İlgili konuşmanın ses dosyasını kullanır.
3. MuseTalk'a yalnızca o kısa video klibini gönderir.
4. Her klip için ayrı çıktı oluşturur.
5. Çıktıları doğrular ve raporlar.

ÖNEMLİ:
Bu sürüm senkronize diyalog klipleri üretir.
Henüz bunları ana filmin üzerine birleştirmez.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

from pathlib import Path
from typing import Any


VERSION = "1.1.0"
FPS = 25

MODEL_FILES = {
    "unet": "models/musetalkV15/unet.pth",
    "unet_config": "models/musetalkV15/musetalk.json",
    "vae_config": "models/sd-vae/config.json",
    "vae_weights": "models/sd-vae/diffusion_pytorch_model.bin",
    "whisper_config": "models/whisper/config.json",
    "whisper_weights": "models/whisper/pytorch_model.bin",
    "whisper_processor": "models/whisper/preprocessor_config.json",
    "dwpose": "models/dwpose/dw-ll_ucoco_384.pth",
    "syncnet": "models/syncnet/latentsync_syncnet.pt",
    "face_parse": "models/face-parse-bisent/79999_iter.pth",
    "resnet": "models/face-parse-bisent/resnet18-5c106cde.pth",
}


def log(message: str) -> None:
    print(f"[MuseTalk Worker] {message}", flush=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(
    command: list[str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:

    log("Komut: " + " ".join(command))

    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout[-12000:], flush=True)

    if result.returncode != 0:
        fail(
            f"Komut başarısız. Çıkış kodu: "
            f"{result.returncode}"
        )

    return result


def read_json(path: Path) -> dict[str, Any]:

    if not path.is_file():
        fail(f"JSON bulunamadı: {path}")

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


def require_binary(name: str) -> None:

    if shutil.which(name) is None:
        fail(f"Gerekli program bulunamadı: {name}")


def probe_duration(path: Path) -> float:

    result = run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])

    try:
        duration = float(result.stdout.strip())
    except Exception:
        fail(f"Medya süresi okunamadı: {path}")

    if duration <= 0:
        fail(f"Medya süresi geçersiz: {path}")

    return duration


def check_gpu() -> dict[str, Any]:

    try:
        import torch
    except ImportError:
        fail("PyTorch kurulu değil.")

    if not torch.cuda.is_available():
        fail(
            "CUDA GPU bulunamadı. "
            "Kaggle Notebook GPU ayarını kontrol et."
        )

    device = torch.cuda.get_device_properties(0)

    result = {
        "cuda_available": True,
        "device_name": device.name,
        "total_memory_gb": round(
            device.total_memory / (1024 ** 3),
            2,
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    log("GPU: " + result["device_name"])

    return result


def check_models(
    root: Path,
) -> dict[str, str]:

    checked = {}

    for name, relative in MODEL_FILES.items():

        path = root / relative

        if not path.is_file():
            fail(f"Model dosyası eksik: {path}")

        if path.stat().st_size == 0:
            fail(f"Model dosyası boş: {path}")

        checked[name] = str(path.resolve())

    return checked


def validate_package(
    package: Path,
) -> tuple[Path, dict[str, Any]]:

    package = package.resolve()

    video = package / "video.mp4"
    timeline_path = package / "lip_sync_timeline.json"

    if not video.is_file():
        fail(f"Kaynak video bulunamadı: {video}")

    timeline = read_json(timeline_path)

    if timeline.get("status") != "package_ready":
        fail("Paket durumu package_ready değil.")

    if timeline.get("lip_sync_rendered") is True:
        fail("Paket zaten işlenmiş olarak işaretli.")

    dialogue = timeline.get("dialogue")

    if not isinstance(dialogue, list) or not dialogue:
        fail("Diyalog listesi boş.")

    return video, timeline


def inspect_face_policy(
    entry: dict[str, Any],
) -> tuple[bool, str]:

    """
    Otomatik konuşmacı-yüz eşleştirmesi yapılmadığı için
    yalnızca açıkça onaylanmış tek yüzlü çekimler işlenir.
    """

    face_count = entry.get("face_count")

    confirmed = (
        entry.get("target_face_confirmed") is True
    )

    if face_count is None:
        return False, "Yüz sayısı doğrulanmamış."

    try:
        face_count = int(face_count)
    except (TypeError, ValueError):
        return False, "Yüz sayısı geçersiz."

    if face_count != 1:
        return False, "Çekimde tam olarak bir yüz yok."

    if not confirmed:
        return False, "Hedef konuşmacı yüzü onaylanmamış."

    return True, "Tek yüz ve hedef konuşmacı onaylı."


def create_dialogue_clip(
    source_video: Path,
    output_video: Path,
    start: float,
    duration: float,
) -> None:

    if start < 0 or duration <= 0:
        fail("Video kesme zamanları geçersiz.")

    output_video.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run([
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source_video),
        "-ss", f"{start:.4f}",
        "-t", f"{duration:.4f}",
        "-an",
        "-vf", f"fps={FPS}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_video),
    ])

    if not output_video.is_file():
        fail("Diyalog video klibi oluşturulamadı.")

    if output_video.stat().st_size == 0:
        fail("Oluşturulan video klibi boş.")


def yaml_quote(value: str) -> str:

    return json.dumps(
        value,
        ensure_ascii=False,
    )


def write_inference_config(
    tasks: list[dict[str, Any]],
    output_path: Path,
) -> None:

    lines = []

    for index, task in enumerate(tasks):

        lines.append(f"task_{index}:")

        lines.append(
            "  video_path: "
            + yaml_quote(task["video_path"])
        )

        lines.append(
            "  audio_path: "
            + yaml_quote(task["audio_path"])
        )

        lines.append(
            "  result_name: "
            + yaml_quote(task["result_name"])
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_inference(
    root: Path,
    config: Path,
    result_dir: Path,
) -> None:

    inference_script = root / "scripts" / "inference.py"

    if not inference_script.is_file():
        fail("MuseTalk inference.py bulunamadı.")

    command = [
        sys.executable,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config.resolve()),
        "--result_dir",
        str(result_dir.resolve()),
        "--unet_model_path",
        str(
            root / "models/musetalkV15/unet.pth"
        ),
        "--unet_config",
        str(
            root / "models/musetalkV15/musetalk.json"
        ),
        "--whisper_dir",
        str(root / "models/whisper"),
        "--version",
        "v15",
        "--use_float16",
    ]

    run(command, cwd=root)


def worker(
    package: Path,
    project_root: Path,
    output_dir: Path,
    allow_approved_faces: bool,
) -> dict[str, Any]:

    started = time.time()

    require_binary("ffmpeg")
    require_binary("ffprobe")

    gpu = check_gpu()

    video, timeline = validate_package(package)

    project_root = project_root.resolve()

    models = check_models(project_root)

    package = package.resolve()

    dialogue = timeline["dialogue"]

    accepted = []
    skipped = []

    for index, entry in enumerate(dialogue):

        if not isinstance(entry, dict):
            skipped.append({
                "dialogue_index": index,
                "reason": "Diyalog kaydı geçersiz.",
            })
            continue

        allowed, reason = inspect_face_policy(entry)

        if not allowed:
            skipped.append({
                "dialogue_index": index,
                "speaker_id": entry.get("speaker_id"),
                "reason": reason,
            })
            continue

        if not allow_approved_faces:
            skipped.append({
                "dialogue_index": index,
                "speaker_id": entry.get("speaker_id"),
                "reason": "Onaylı yüz işleme izni verilmedi.",
            })
            continue

        accepted.append((index, entry))

    if not accepted:
        fail(
            "İşlenebilecek onaylı diyalog bulunamadı. "
            "Güvenlik kontrolü nedeniyle hiçbir yüz "
            "otomatik olarak seçilmeyecek."
        )

    if output_dir.exists():
        fail(
            "Çıktı klasörü zaten mevcut: "
            + str(output_dir)
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    clip_dir = output_dir / "dialogue_clips"
    clip_dir.mkdir(parents=True)

    tasks = []
    task_records = []

    for index, entry in accepted:

        speaker_id = str(entry["speaker_id"])

        start = float(entry["film_start_seconds"])
        end = float(entry["film_end_seconds"])

        audio_relative = Path(
            str(entry["audio_file"])
        )

        audio_path = (
            package / audio_relative
        ).resolve()

        try:
            audio_path.relative_to(package)
        except ValueError:
            fail(
                f"Diyalog {index}: ses dosyası paket dışına çıkıyor."
            )

        if not audio_path.is_file():
            fail(
                f"Diyalog {index}: ses dosyası bulunamadı."
            )

        audio_duration = probe_duration(audio_path)

        timeline_duration = end - start

        if abs(audio_duration - timeline_duration) > 1.0:
            fail(
                f"Diyalog {index}: ses süresi ile "
                "zaman çizelgesi arasında 1 saniyeden "
                "fazla fark var. Otomatik kırpma yapılmadı. "
                f"Ses: {audio_duration:.2f}s, "
                f"zaman aralığı: {timeline_duration:.2f}s"
            )

        clip_name = f"dialogue_{index:04d}.mp4"

        clip_path = clip_dir / clip_name

        create_dialogue_clip(
            source_video=video,
            output_video=clip_path,
            start=start,
            duration=audio_duration,
        )

        clip_duration = probe_duration(clip_path)

        if abs(clip_duration - audio_duration) > 1.0:
            fail(
                f"Diyalog {index}: oluşturulan klibin "
                "süresi ses ile uyuşmuyor."
            )

        result_name = (
            f"dialogue_{index:04d}_synced.mp4"
        )

        task = {
            "video_path": str(clip_path.resolve()),
            "audio_path": str(audio_path.resolve()),
            "result_name": result_name,
        }

        tasks.append(task)

        task_records.append({
            "dialogue_index": index,
            "speaker_id": speaker_id,
            "text": str(entry.get("text", "")),
            "film_start_seconds": start,
            "film_end_seconds": end,
            "audio_duration_seconds": round(
                audio_duration, 4
            ),
            "input_clip": str(clip_path),
            "expected_output": result_name,
        })

        log(
            f"Diyalog {index}: kısa video klibi hazır."
        )

    config_path = output_dir / "inference.yaml"

    write_inference_config(
        tasks,
        config_path,
    )

    result_dir = output_dir / "musetalk_results"

    run_inference(
        root=project_root,
        config=config_path,
        result_dir=result_dir,
    )

    generated = []

    for task, record in zip(tasks, task_records):

        expected = (
            result_dir
            / "v15"
            / task["result_name"]
        )

        if not expected.is_file():
            fail(
                "MuseTalk beklenen çıktı dosyasını üretmedi: "
                + str(expected)
            )

        if expected.stat().st_size == 0:
            fail(
                "MuseTalk boş çıktı oluşturdu: "
                + str(expected)
            )

        output_duration = probe_duration(expected)

        if abs(
            output_duration
            - record["audio_duration_seconds"]
        ) > 1.5:
            fail(
                "MuseTalk çıktı süresi beklenen ses "
                "süresiyle uyuşmuyor: "
                + str(expected)
            )

        generated.append({
            "dialogue_index": record["dialogue_index"],
            "speaker_id": record["speaker_id"],
            "file": str(expected),
            "size_bytes": expected.stat().st_size,
            "duration_seconds": round(
                output_duration, 4
            ),
        })

    report = {
        "worker_version": VERSION,
        "status": "dialogue_clips_generated",
        "lip_sync_rendered": False,
        "full_film_composited": False,
        "gpu": gpu,
        "models": models,
        "total_dialogue": len(dialogue),
        "accepted_dialogue": len(accepted),
        "skipped_dialogue": len(skipped),
        "skipped": skipped,
        "dialogue_tasks": task_records,
        "generated_videos": generated,
        "inference_config": str(config_path),
        "elapsed_seconds": round(
            time.time() - started, 2
        ),
        "next_required_step": (
            "Generated dialogue clips must be composited "
            "at their original film timestamps. The original "
            "film audio must be preserved. This worker does "
            "not create the final full-length film."
        ),
    }

    write_json(
        output_dir / "musetalk_report.json",
        report,
    )

    log("Diyalog klipleri üretildi ve doğrulandı.")

    return report


def main() -> None:

    parser = argparse.ArgumentParser(
        description="MuseTalk 1.5 Kaggle GPU Worker"
    )

    parser.add_argument(
        "--package",
        required=True,
    )

    parser.add_argument(
        "--musetalk-root",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--allow-approved-faces",
        action="store_true",
    )

    args = parser.parse_args()

    try:

        report = worker(
            package=Path(args.package),
            project_root=Path(args.musetalk_root),
            output_dir=Path(args.output),
            allow_approved_faces=args.allow_approved_faces,
        )

        print(
            json.dumps(
                report,
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

