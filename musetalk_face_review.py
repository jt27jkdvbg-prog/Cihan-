#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuseTalk Face Review v1.1

Diyalog kliplerindeki yüzleri inceler.
Otomatik konuşmacı eşleştirmesi veya yüz onayı yapmaz.

Çıktılar:
- face_review.json
- previews/
- face_review_output.zip
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import zipfile

from pathlib import Path

import cv2
import numpy as np

try:
    from face_detection import FaceAlignment, LandmarksType
except ImportError as exc:
    raise SystemExit(
        "MuseTalk face_detection paketi bulunamadı. "
        "Bu dosyayı MuseTalk kurulu ortamda çalıştır."
    ) from exc


VERSION = "1.1.0"


def run(command):
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


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "JSON kök verisi nesne olmalı."
        )

    return data


def write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def extract_frame(video, timestamp, destination):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run([
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{timestamp:.4f}",
        "-i", str(video),
        "-frames:v", "1",
        "-q:v", "2",
        str(destination),
    ])

    if (
        not destination.is_file()
        or destination.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Görüntü çıkarılamadı: {destination}"
        )

    frame = cv2.imread(str(destination))

    if frame is None:
        raise RuntimeError(
            f"Görüntü okunamadı: {destination}"
        )

    return frame


def detect_faces(detector, frame):
    """
    Bir karedeki tüm yüz kutularını döndürür.

    get_detections_for_batch yerine doğrudan
    face_detector kullanılır; böylece yalnızca
    ilk yüzün döndürülmesi sorunu önlenir.
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    face_detector = getattr(
        detector,
        "face_detector",
        None,
    )

    if face_detector is None:
        raise RuntimeError(
            "Yüz algılama motoru bulunamadı."
        )

    if not hasattr(
        face_detector,
        "detect_from_batch",
    ):
        raise RuntimeError(
            "Yüz algılama motorunda "
            "detect_from_batch metodu bulunamadı."
        )

    detections = face_detector.detect_from_batch(
        np.asarray([rgb])
    )

    if detections is None or len(detections) == 0:
        return []

    frame_detections = detections[0]

    if frame_detections is None:
        return []

    boxes = np.asarray(frame_detections)

    if boxes.size == 0:
        return []

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    result = []

    height, width = frame.shape[:2]

    for box in boxes:
        values = np.asarray(box).reshape(-1)

        if len(values) < 4:
            continue

        coordinates = values[:4]

        if not np.all(np.isfinite(coordinates)):
            continue

        x1, y1, x2, y2 = [
            int(round(float(value)))
            for value in coordinates
        ]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        result.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return result


def annotate_frame(frame, boxes, label):
    image = frame.copy()

    for index, box in enumerate(boxes):
        cv2.rectangle(
            image,
            (box["x1"], box["y1"]),
            (box["x2"], box["y2"]),
            (0, 220, 0),
            3,
        )

        cv2.putText(
            image,
            f"Face {index + 1}",
            (
                box["x1"],
                max(25, box["y1"] - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return image


def create_review_zip(output_dir):
    """
    İnceleme raporunu ve önizleme görsellerini
    tek bir indirilebilir ZIP dosyasında toplar.
    """

    zip_path = output_dir.parent / (
        output_dir.name + ".zip"
    )

    if zip_path.exists():
        raise FileExistsError(
            f"ZIP dosyası zaten mevcut: {zip_path}"
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
            if not file_path.is_file():
                continue

            archive_name = (
                Path(output_dir.name)
                / file_path.relative_to(output_dir)
            )

            archive.write(
                file_path,
                archive_name.as_posix(),
            )

    if (
        not zip_path.is_file()
        or zip_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "İnceleme ZIP paketi oluşturulamadı."
        )

    return zip_path


def review(
    package: Path,
    output_dir: Path,
    device: str,
):
    package = package.resolve()
    output_dir = output_dir.resolve()

    video = package / "video.mp4"
    timeline_path = (
        package / "lip_sync_timeline.json"
    )

    if not video.is_file():
        raise FileNotFoundError(video)

    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)

    if output_dir.exists():
        raise FileExistsError(
            f"Çıktı klasörü zaten var: {output_dir}"
        )

    timeline = read_json(timeline_path)

    if timeline.get("status") != "package_ready":
        raise RuntimeError(
            "Zaman çizelgesi package_ready "
            "durumunda değil."
        )

    dialogue = timeline.get("dialogue")

    if (
        not isinstance(dialogue, list)
        or not dialogue
    ):
        raise RuntimeError(
            "Diyalog listesi boş."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    preview_dir = output_dir / "previews"

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = FaceAlignment(
        LandmarksType._2D,
        flip_input=False,
        device=device,
    )

    results = []

    for index, entry in enumerate(dialogue):

        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Diyalog {index} geçersiz."
            )

        try:
            start = float(
                entry["film_start_seconds"]
            )

            end = float(
                entry["film_end_seconds"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            raise RuntimeError(
                f"Diyalog {index} zaman bilgileri geçersiz."
            )

        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise RuntimeError(
                f"Diyalog {index} zaman aralığı geçersiz."
            )

        duration = end - start

        sample_times = [
            start + duration * 0.20,
            start + duration * 0.50,
            start + duration * 0.80,
        ]

        frame_results = []

        for sample_index, timestamp in enumerate(
            sample_times
        ):

            name = (
                f"dialogue_{index:04d}_"
                f"sample_{sample_index + 1}.jpg"
            )

            image_path = preview_dir / name

            frame = extract_frame(
                video,
                timestamp,
                image_path,
            )

            boxes = detect_faces(
                detector,
                frame,
            )

            annotated = annotate_frame(
                frame,
                boxes,
                (
                    f"Dialogue {index} | "
                    f"{timestamp:.2f}s"
                ),
            )

            if not cv2.imwrite(
                str(image_path),
                annotated,
            ):
                raise RuntimeError(
                    f"Önizleme kaydedilemedi: {image_path}"
                )

            frame_results.append({
                "sample_index": sample_index + 1,
                "timestamp_seconds": round(
                    timestamp,
                    4,
                ),
                "face_count": len(boxes),
                "face_boxes_xyxy": boxes,
                "preview": (
                    f"previews/{name}"
                ),
            })

        counts = [
            item["face_count"]
            for item in frame_results
        ]

        stable_single_face = all(
            count == 1
            for count in counts
        )

        results.append({
            "dialogue_index": index,
            "speaker_id": entry.get("speaker_id"),
            "text": entry.get("text", ""),
            "film_start_seconds": start,
            "film_end_seconds": end,
            "sample_frames": frame_results,
            "face_count": (
                1 if stable_single_face else None
            ),
            "target_face_confirmed": False,
            "selected_face_index": None,
            "review_status": (
                "single_face_candidate"
                if stable_single_face
                else "manual_review_required"
            ),
            "human_approved": False,
            "automatic_approval": False,
        })

        print(
            f"Diyalog {index}: "
            f"yüz sayıları={counts}",
            flush=True,
        )

    report = {
        "version": VERSION,
        "status": "face_review_ready",
        "automatic_speaker_matching": False,
        "automatic_face_approval": False,
        "total_dialogue": len(results),
        "dialogue": results,
        "important_note": (
            "Bu rapor yalnızca yüz tespiti ve "
            "inceleme görselleri üretir. Yüzün "
            "konuşmacıya ait olduğunu kanıtlamaz. "
            "Her diyalog insan tarafından "
            "doğrulanmadan işlenmemelidir."
        ),
    }

    report_path = (
        output_dir / "face_review.json"
    )

    write_json(
        report_path,
        report,
    )

    zip_path = create_review_zip(
        output_dir
    )

    print(
        json.dumps(
            {
                "status": "success",
                "version": VERSION,
                "report": str(report_path),
                "preview_directory": str(
                    preview_dir
                ),
                "zip_file": str(zip_path),
                "dialogue_count": len(results),
                "automatic_approval": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MuseTalk yüz inceleme ve "
            "önizleme paketi oluşturucu."
        )
    )

    parser.add_argument(
        "--package",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "FFmpeg bulunamadı."
        )

    try:
        review(
            package=Path(args.package),
            output_dir=Path(args.output),
            device=args.device,
        )

    except Exception as exc:
        raise SystemExit(
            f"HATA: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
