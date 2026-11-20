
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuseTalk Face Approval v1.0

Manuel yüz onaylarını kontrol eder ve
onaylanan diyalogları yeni pakete aktarır.

Otomatik yüz veya karakter onayı yapmaz.
"""

from __future__ import annotations

import argparse
import json
import shutil

from pathlib import Path
from typing import Any


VERSION = "1.0.0"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise RuntimeError(f"Geçersiz JSON: {path}")

    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def approve(
    package: Path,
    review_path: Path,
    output_package: Path,
) -> dict[str, Any]:

    package = package.resolve()
    review_path = review_path.resolve()
    output_package = output_package.resolve()

    timeline_path = package / "lip_sync_timeline.json"
    video_path = package / "video.mp4"

    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if not review_path.is_file():
        raise FileNotFoundError(review_path)

    if output_package.exists():
        raise FileExistsError(
            f"Çıktı klasörü zaten mevcut: {output_package}"
        )

    timeline = read_json(timeline_path)
    review = read_json(review_path)

    if timeline.get("status") != "package_ready":
        raise RuntimeError(
            "Kaynak zaman çizelgesi package_ready değil."
        )

    if review.get("status") != "face_review_ready":
        raise RuntimeError(
            "Yüz inceleme raporu face_review_ready değil."
        )

    original_dialogue = timeline.get("dialogue")
    reviewed_dialogue = review.get("dialogue")

    if not isinstance(original_dialogue, list):
        raise RuntimeError("Kaynak diyalog listesi geçersiz.")

    if not isinstance(reviewed_dialogue, list):
        raise RuntimeError("İnceleme diyalog listesi geçersiz.")

    review_by_index = {}

    for item in reviewed_dialogue:
        if not isinstance(item, dict):
            continue

        index = item.get("dialogue_index")

        if index is None or index in review_by_index:
            raise RuntimeError(
                "İnceleme raporunda eksik veya tekrarlanan diyalog indeksi."
            )

        review_by_index[index] = item

    approved_count = 0
    skipped = []
    updated_dialogue = []

    for index, original in enumerate(original_dialogue):

        if not isinstance(original, dict):
            skipped.append({
                "dialogue_index": index,
                "reason": "Kaynak diyalog kaydı geçersiz.",
            })
            continue

        reviewed = review_by_index.get(index)

        if reviewed is None:
            skipped.append({
                "dialogue_index": index,
                "reason": "Yüz inceleme kaydı bulunamadı.",
            })
            continue

        samples = reviewed.get("sample_frames")

        if not isinstance(samples, list) or len(samples) != 3:
            skipped.append({
                "dialogue_index": index,
                "reason": "Üç örnek kare incelemesi tamamlanmamış.",
            })
            continue

        counts = []

        try:
            counts = [
                int(sample["face_count"])
                for sample in samples
            ]
        except (KeyError, TypeError, ValueError):
            skipped.append({
                "dialogue_index": index,
                "reason": "Yüz sayısı bilgisi geçersiz.",
            })
            continue

        if counts != [1, 1, 1]:
            skipped.append({
                "dialogue_index": index,
                "reason": "Üç örnek karenin tamamında tek yüz yok.",
                "face_counts": counts,
            })
            continue

        if reviewed.get("human_approved") is not True:
            skipped.append({
                "dialogue_index": index,
                "reason": "İnsan onayı verilmemiş.",
            })
            continue

        expected_speaker = str(
            original.get("speaker_id", "")
        ).strip()

        confirmed_speaker = str(
            reviewed.get("confirmed_speaker_id", "")
        ).strip()

        if not expected_speaker or confirmed_speaker != expected_speaker:
            skipped.append({
                "dialogue_index": index,
                "reason": "Onaylanan konuşmacı kimliği eşleşmiyor.",
            })
            continue

        selected_face = reviewed.get("selected_face_index")

        if selected_face != 0:
            skipped.append({
                "dialogue_index": index,
                "reason": (
                    "MuseTalk tek yüzlü çekimlerde "
                    "ilk yüzü işler; seçili yüz indeksi 0 olmalı."
                ),
            })
            continue

        entry = dict(original)

        entry["face_count"] = 1
        entry["target_face_confirmed"] = True
        entry["selected_face_index"] = 0
        entry["face_approval_method"] = "manual_review"
        entry["face_approval_version"] = VERSION

        updated_dialogue.append(entry)
        approved_count += 1

    if approved_count == 0:
        raise RuntimeError(
            "Onaylanan diyalog bulunamadı. "
            "Hiçbir çıktı paketi oluşturulmadı."
        )

    output_package.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(package, output_package)

    updated_timeline = dict(timeline)
    updated_timeline["dialogue"] = updated_dialogue
    updated_timeline["status"] = "package_ready"
    updated_timeline["lip_sync_rendered"] = False
    updated_timeline["face_approval"] = {
        "version": VERSION,
        "method": "manual_review",
        "approved_dialogue_count": approved_count,
        "skipped_dialogue_count": len(skipped),
        "automatic_approval": False,
    }

    write_json(
        output_package / "lip_sync_timeline.json",
        updated_timeline,
    )

    report = {
        "version": VERSION,
        "status": "approval_completed",
        "approved_dialogue_count": approved_count,
        "skipped_dialogue_count": len(skipped),
        "approved_dialogue_indices": [
            item.get("dialogue_index")
            for item in reviewed_dialogue
            if (
                isinstance(item, dict)
                and item.get("human_approved") is True
            )
        ],
        "skipped": skipped,
        "output_package": str(output_package),
        "note": (
            "Only manually approved single-face dialogue "
            "entries were copied into the new package. "
            "No automatic speaker identity approval was performed."
        ),
    }

    write_json(
        output_package / "face_approval_report.json",
        report,
    )

    print(
        json.dumps(report, ensure_ascii=False, indent=2),
        flush=True,
    )

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--package", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output-package", required=True)

    args = parser.parse_args()

    try:
        approve(
            package=Path(args.package),
            review_path=Path(args.review),
            output_package=Path(args.output_package),
        )
    except Exception as exc:
        raise SystemExit(f"HATA: {exc}") from exc


if __name__ == "__main__":
    main()
