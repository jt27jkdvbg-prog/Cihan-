
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuseTalk Face Review v1.0

Diyalog kliplerindeki yüzleri inceler.
Otomatik konuşmacı eşleştirmesi veya yüz onayı yapmaz.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess

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
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError("JSON kök verisi nesne olmalı.")

    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temp.replace(path)


def extract_frame(video, timestamp, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

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

    if not destination.is_file() or destination.stat().st_size == 0:
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = detector.get_detections_for_batch(
        np.asarray([rgb])
    )

    if not detections:
        return []

    boxes = detections[0]

    if boxes is None:
        return []

    result = []

    for box in boxes:
        if box is None:
            continue

        values = np.asarray(box).reshape(-1)

        if len(values) < 4:
            continue

        x1, y1, x2, y2 = [
            int(round(float(v))) for v in values[:4]
        ]

        height, width = frame.shape[:2]

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
            (box["x1"], max(25, box["y1"] - 10)),
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


def review(
    package: Path,
    output_dir: Path,
    device: str,
):
    package = package.resolve()
    output_dir = output_dir.resolve()

    video = package / "video.mp4"
    timeline_path = package / "lip_sync_timeline.json"

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
            "Zaman çizelgesi package_ready durumunda değil."
        )

    dialogue = timeline.get("dialogue")

    if not isinstance(dialogue, list) or not dialogue:
        raise RuntimeError("Diyalog listesi boş.")

    output_dir.mkdir(parents=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir()

    detector = FaceAlignment(
        LandmarksType._2D,
        flip_input=False,
        device=device,
    )

    results = []

    for index, entry in enumerate(dialogue):
        if not isinstance(entry, dict):
            continue

        start = float(entry["film_start_seconds"])
        end = float(entry["film_end_seconds"])

        if end <= start:
            continue

        duration = end - start

        sample_times = [
            start + duration * 0.20,
            start + duration * 0.50,
            start + duration * 0.80,
        ]

        frame_results = []

        for sample_index, timestamp in enumerate(sample_times):
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

            boxes = detect_faces(detector, frame)

            annotated = annotate_frame(
                frame,
                boxes,
                f"Dialogue {index} | {timestamp:.2f}s",
            )

            cv2.imwrite(str(image_path), annotated)

            frame_results.append({
                "sample_index": sample_index + 1,
                "timestamp_seconds": round(timestamp, 4),
                "face_count": len(boxes),
                "face_boxes_xyxy": boxes,
                "preview": f"previews/{name}",
            })

        counts = [
            item["face_count"]
            for item in frame_results
        ]

        stable_single_face = all(
            count == 1 for count in counts
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
            "automatic_approval": False,
        })

        print(
            f"Diyalog {index}: "
            f"yüz sayıları={counts}",
            flush=True,
        )

    report = {
        "version": "1.0.0",
        "status": "face_review_ready",
        "automatic_speaker_matching": False,
        "automatic_face_approval": False,
        "total_dialogue": len(results),
        "dialogue": results,
        "important_note": (
            "Bu rapor yalnızca yüz tespiti ve inceleme "
            "görselleri üretir. Yüzün konuşmacıya ait "
            "olduğunu kanıtlamaz. Her diyalog insan "
            "tarafından doğrulanmadan işlenmemelidir."
        ),
    }

    report_path = output_dir / "face_review.json"
    write_json(report_path, report)

    print(
        json.dumps(
            {
                "status": "success",
                "report": str(report_path),
                "preview_directory": str(preview_dir),
                "dialogue_count": len(results),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("FFmpeg bulunamadı.")

    try:
        review(
            Path(args.package),
            Path(args.output),
            args.device,
        )
    except Exception as exc:
        raise SystemExit(f"HATA: {exc}") from exc


if __name__ == "__main__":
    main()
