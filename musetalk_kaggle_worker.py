
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuseTalk 1.5 Kaggle GPU Worker

- Validates a prepared MuseTalk package.
- Checks GPU and model files.
- Validates dialogue timing and audio references.
- Skips ambiguous multi-face tasks unless explicitly approved.
- Creates the official MuseTalk inference YAML.
- Runs MuseTalk 1.5 inference.
- Writes a machine-readable processing report.

This worker does not claim that skipped or failed tasks
have been lip-synced.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from pathlib import Path
from typing import Any


VERSION = "1.0.0"

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
    print(
        f"[MuseTalk Worker] {message}",
        flush=True,
    )


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"JSON dosyası bulunamadı: {path}")

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        fail(f"JSON okunamadı: {path}: {exc}")

    if not isinstance(data, dict):
        fail(f"JSON nesnesi geçersiz: {path}")

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


def run(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:

    log("Komut çalıştırılıyor:")
    log(" ".join(command))

    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
    )

    if process.returncode != 0:
        fail(
            "Komut başarısız oldu. "
            f"Çıkış kodu: {process.returncode}"
        )


def check_binary(name: str) -> None:
    if shutil.which(name) is None:
        fail(
            f"Gerekli program bulunamadı: {name}"
        )


def check_gpu() -> dict[str, Any]:

    try:
        import torch
    except ImportError:
        fail(
            "PyTorch kurulu değil."
        )

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

    log(
        "GPU: "
        + result["device_name"]
    )

    log(
        "GPU belleği: "
        + str(result["total_memory_gb"])
        + " GB"
    )

    return result


def check_models(
    project_root: Path,
) -> dict[str, str]:

    checked = {}

    for name, relative_path in MODEL_FILES.items():

        path = project_root / relative_path

        if not path.is_file():
            fail(
                f"Model dosyası eksik: {path}"
            )

        if path.stat().st_size == 0:
            fail(
                f"Model dosyası boş: {path}"
            )

        checked[name] = str(path)

    return checked


def validate_package(
    package: Path,
) -> tuple[Path, Path, dict[str, Any]]:

    package = package.resolve()

    video = package / "video.mp4"
    timeline_path = (
        package / "lip_sync_timeline.json"
    )

    if not video.is_file():
        fail(
            f"Video bulunamadı: {video}"
        )

    timeline = read_json(timeline_path)

    if timeline.get("lip_sync_rendered") is True:
        fail(
            "Bu paket zaten işlenmiş olarak işaretli."
        )

    if timeline.get("status") != "package_ready":
        fail(
            "Paket durumu package_ready değil."
        )

    dialogue = timeline.get("dialogue")

    if not isinstance(dialogue, list) or not dialogue:
        fail(
            "Diyalog listesi boş."
        )

    for index, item in enumerate(dialogue):

        if not isinstance(item, dict):
            fail(
                f"Diyalog kaydı geçersiz: {index}"
            )

        for key in (
            "speaker_id",
            "audio_file",
            "film_start_seconds",
            "film_end_seconds",
        ):
            if key not in item:
                fail(
                    f"Diyalog {index}: {key} eksik."
                )

        start = float(
            item["film_start_seconds"]
        )

        end = float(
            item["film_end_seconds"]
        )

        if start < 0 or end <= start:
            fail(
                f"Diyalog {index}: zaman aralığı geçersiz."
            )

        audio = (
            package / str(item["audio_file"])
        ).resolve()

        try:
            audio.relative_to(package)
        except ValueError:
            fail(
                f"Diyalog {index}: ses dosyası paket dışına çıkıyor."
            )

        if not audio.is_file():
            fail(
                f"Diyalog {index}: ses dosyası bulunamadı: {audio}"
            )

        if audio.stat().st_size == 0:
            fail(
                f"Diyalog {index}: ses dosyası boş."
            )

    return video, timeline_path, timeline


def inspect_face_policy(
    entry: dict[str, Any],
) -> tuple[bool, str]:

    """
    MuseTalk's standard inference pipeline does not
    provide reliable speaker-to-face identity matching
    for arbitrary multi-person shots.

    Therefore, this worker requires explicit approval
    for the target face before processing a dialogue.
    """

    face_count = entry.get(
        "face_count"
    )

    target_confirmed = (
        entry.get("target_face_confirmed")
        is True
    )

    if face_count is None:
        return (
            False,
            "Yüz sayısı doğrulanmamış.",
        )

    try:
        face_count = int(face_count)
    except (TypeError, ValueError):
        return (
            False,
            "Yüz sayısı geçersiz.",
        )

    if face_count != 1:
        return (
            False,
            "Tek yüzlü çekim değil; otomatik eşleştirme güvenli değil.",
        )

    if not target_confirmed:
        return (
            False,
            "Hedef konuşmacının yüzü onaylanmamış.",
        )

    return (
        True,
        "Tek yüz ve konuşmacı eşleştirmesi onaylandı.",
    )


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

        lines.append(
            f"task_{index}:"
        )

        lines.append(
            "  video_path: "
            + yaml_quote(task["video_path"])
        )

        lines.append(
            "  audio_path: "
            + yaml_quote(task["audio_path"])
        )

        if task.get("bbox_shift") is not None:
            lines.append(
                "  bbox_shift: "
                + str(int(task["bbox_shift"]))
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
    project_root: Path,
    config_path: Path,
    result_dir: Path,
) -> None:

    inference_script = (
        project_root / "scripts" / "inference.py"
    )

    if not inference_script.is_file():
        fail(
            "MuseTalk scripts/inference.py bulunamadı."
        )

    command = [
        sys.executable,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config_path),
        "--result_dir",
        str(result_dir),
        "--unet_model_path",
        str(
            project_root
            / "models/musetalkV15/unet.pth"
        ),
        "--unet_config",
        str(
            project_root
            / "models/musetalkV15/musetalk.json"
        ),
        "--version",
        "v15",
        "--use_float16",
    ]

    run(
        command,
        cwd=project_root,
    )


def worker(
    package: Path,
    project_root: Path,
    output_dir: Path,
    allow_approved_faces: bool,
) -> dict[str, Any]:

    started = time.time()

    check_binary("ffmpeg")
    check_binary("ffprobe")

    gpu = check_gpu()

    video, timeline_path, timeline = (
        validate_package(package)
    )

    models = check_models(project_root)

    dialogue = timeline["dialogue"]

    accepted = []
    skipped = []

    for index, entry in enumerate(dialogue):

        allowed, reason = inspect_face_policy(
            entry
        )

        if not allowed:
            skipped.append({
                "dialogue_index": index,
                "speaker_id": entry.get(
                    "speaker_id"
                ),
                "reason": reason,
            })
            continue

        if not allow_approved_faces:
            skipped.append({
                "dialogue_index": index,
                "speaker_id": entry.get(
                    "speaker_id"
                ),
                "reason": (
                    "İşleme izni verilmedi."
                ),
            })
            continue

        accepted.append(
            (index, entry)
        )

    if not accepted:
        fail(
            "Güvenli şekilde işlenebilecek diyalog yok. "
            "Her kayıt için face_count=1 ve "
            "target_face_confirmed=true gereklidir."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    task_dir = output_dir / "tasks"
    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tasks = []

    for index, entry in accepted:

        audio = (
            package / str(entry["audio_file"])
        ).resolve()

        task = {
            "original_dialogue_index": index,
            "speaker_id": str(
                entry["speaker_id"]
            ),
            "video_path": str(
                video.resolve()
            ),
            "audio_path": str(
                audio
            ),
            "bbox_shift": entry.get(
                "bbox_shift"
            ),
            "film_start_seconds": float(
                entry["film_start_seconds"]
            ),
            "film_end_seconds": float(
                entry["film_end_seconds"]
            ),
            "text": str(
                entry.get("text", "")
            ),
        }

        tasks.append(task)

    config_path = (
        output_dir / "inference.yaml"
    )

    write_inference_config(
        tasks,
        config_path,
    )

    result_dir = (
        output_dir / "musetalk_results"
    )

    run_inference(
        project_root,
        config_path,
        result_dir,
    )

    generated = []

    for path in sorted(
        result_dir.rglob("*.mp4")
    ):
        generated.append({
            "file": str(path),
            "size_bytes": path.stat().st_size,
        })

    if len(generated) != len(tasks):
        fail(
            "MuseTalk çıktı sayısı görev sayısıyla eşleşmiyor. "
            f"Görev: {len(tasks)}, çıktı: {len(generated)}"
        )

    for item in generated:
        if item["size_bytes"] <= 0:
            fail(
                "MuseTalk boş video çıktısı oluşturdu."
            )

    report = {
        "worker_version": VERSION,
        "status": "inference_completed",
        "lip_sync_rendered": True,
        "gpu": gpu,
        "models": models,
        "total_dialogue": len(dialogue),
        "accepted_dialogue": len(accepted),
        "skipped_dialogue": len(skipped),
        "skipped": skipped,
        "generated_videos": generated,
        "inference_config": str(config_path),
        "elapsed_seconds": round(
            time.time() - started,
            2,
        ),
        "important_note": (
            "MuseTalk clips have been generated. "
            "They have not yet been composited into "
            "the original full-length film."
        ),
    }

    write_json(
        output_dir / "musetalk_report.json",
        report,
    )

    log(
        "MuseTalk inference tamamlandı."
    )

    return report


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "MuseTalk 1.5 Kaggle GPU Worker"
        )
    )

    parser.add_argument(
        "--package",
        required=True,
        help="Hazırlanmış MuseTalk paket klasörü.",
    )

    parser.add_argument(
        "--musetalk-root",
        required=True,
        help="MuseTalk kaynak kodu ve modellerinin bulunduğu klasör.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Çıktı klasörü.",
    )

    parser.add_argument(
        "--allow-approved-faces",
        action="store_true",
        help="Yalnızca açıkça onaylanmış tek yüzlü görevleri işle.",
    )

    args = parser.parse_args()

    try:
        report = worker(
            package=Path(args.package),
            project_root=Path(
                args.musetalk_root
            ).resolve(),
            output_dir=Path(args.output),
            allow_approved_faces=(
                args.allow_approved_faces
            ),
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
        log(f"HATA: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
