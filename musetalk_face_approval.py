
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuseTalk Face Approval v1.1

Yüz inceleme raporunu doğrular ve yalnızca insan
tarafından onaylanan diyalogları yeni pakete aktarır.

--interactive kullanılırsa onayları terminalden toplar.
Otomatik konuşmacı-yüz eşleştirmesi yapmaz.
"""

from __future__ import annotations

import argparse
import json
import shutil

from pathlib import Path
from typing import Any


VERSION = "1.1.0"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise RuntimeError(f"Geçersiz JSON: {path}")

    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def validate_review_record(
    original: dict[str, Any],
    reviewed: dict[str, Any],
    index: int,
) -> tuple[bool, str]:

    if reviewed.get("dialogue_index") != index:
        return False, "Diyalog indeksi eşleşmiyor."

    expected_speaker = str(
        original.get("speaker_id", "")
    ).strip()

    reviewed_speaker = str(
        reviewed.get("speaker_id", "")
    ).strip()

    if not expected_speaker or reviewed_speaker != expected_speaker:
        return False, "Konuşmacı kimliği kaynak kayıtla eşleşmiyor."

    try:
        original_start = float(original["film_start_seconds"])
        original_end = float(original["film_end_seconds"])
        reviewed_start = float(reviewed["film_start_seconds"])
        reviewed_end = float(reviewed["film_end_seconds"])
    except (KeyError, TypeError, ValueError):
        return False, "Zaman bilgileri geçersiz."

    if abs(original_start - reviewed_start) > 0.05:
        return False, "Başlangıç zamanı kaynak kayıtla eşleşmiyor."

    if abs(original_end - reviewed_end) > 0.05:
        return False, "Bitiş zamanı kaynak kayıtla eşleşmiyor."

    samples = reviewed.get("sample_frames")

    if not isinstance(samples, list) or len(samples) != 3:
        return False, "Üç örnek kare incelemesi bulunamadı."

    try:
        counts = [int(sample["face_count"]) for sample in samples]
    except (KeyError, TypeError, ValueError):
        return False, "Örnek kare yüz sayıları geçersiz."

    if counts != [1, 1, 1]:
        return False, "Üç örnek karenin tamamında tek yüz bulunmuyor."

    return True, "İnceleme kaydı geçerli."


def collect_interactive_approvals(
    original_dialogue: list,
    reviewed_dialogue: list,
    review_path: Path,
) -> None:

    review_by_index = {}

    for item in reviewed_dialogue:
        if isinstance(item, dict):
            index = item.get("dialogue_index")
            if index is not None:
                review_by_index[index] = item

    print("\nMANUEL YÜZ ONAYI")
    print("Her diyalog için üç önizleme görselini kontrol et.")
    print("Görsellerin yolları aşağıda gösterilecek.")
    print("Onay için konuşmacı kimliğini aynen yaz.")
    print("Atlamak için ENTER tuşuna bas.\n")

    for index, original in enumerate(original_dialogue):
        reviewed = review_by_index.get(index)

        if not isinstance(original, dict) or not isinstance(reviewed, dict):
            continue

        valid, reason = validate_review_record(
            original,
            reviewed,
            index,
        )

        if not valid:
            print(f"Diyalog {index}: ATLANDI — {reason}")
            continue

        speaker_id = str(original.get("speaker_id", "")).strip()

        print("=" * 60)
        print(f"Diyalog: {index}")
        print(f"Konuşmacı kimliği: {speaker_id}")
        print(f"Metin: {original.get('text', '')}")
        print(
            f"Zaman: {original.get('film_start_seconds')} - "
            f"{original.get('film_end_seconds')} saniye"
        )

        for sample in reviewed["sample_frames"]:
            preview = sample.get("preview", "")
            preview_path = (review_path.parent / preview).resolve()
            print(f"Önizleme: {preview_path}")

        answer = input(
            "\nBu üç görüntüdeki kişi gerçekten bu konuşmacı mı? "
            "Onay için speaker_id yaz: "
        ).strip()

        if answer == speaker_id:
            reviewed["human_approved"] = True
            reviewed["confirmed_speaker_id"] = speaker_id
            reviewed["selected_face_index"] = 0
            print("İnsan onayı kaydedildi.\n")
        else:
            reviewed["human_approved"] = False
            reviewed["confirmed_speaker_id"] = ""
            reviewed["selected_face_index"] = None
            print("Bu diyalog onaylanmadı.\n")

    updated_review = read_json(review_path)
    updated_review["dialogue"] = reviewed_dialogue
    updated_review["interactive_approval_completed"] = True

    write_json(review_path, updated_review)


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
        raise RuntimeError("Kaynak zaman çizelgesi package_ready değil.")

    if review.get("status") != "face_review_ready":
        raise RuntimeError("Yüz inceleme raporu face_review_ready değil.")

    original_dialogue = timeline.get("dialogue")
    reviewed_dialogue = review.get("dialogue")

    if not isinstance(original_dialogue, list) or not original_dialogue:
        raise RuntimeError("Kaynak diyalog listesi geçersiz veya boş.")

    if not isinstance(reviewed_dialogue, list):
        raise RuntimeError("İnceleme diyalog listesi geçersiz.")

    review_by_index = {}

    for item in reviewed_dialogue:
        if not isinstance(item, dict):
            continue

        index = item.get("dialogue_index")

        if not isinstance(index, int) or index in review_by_index:
            raise RuntimeError(
                "İnceleme raporunda eksik veya tekrarlanan diyalog indeksi."
            )

        review_by_index[index] = item

    approved_dialogue = []
    approved_indices = []
    skipped = []

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

        valid, reason = validate_review_record(
            original,
            reviewed,
            index,
        )

        if not valid:
            skipped.append({
                "dialogue_index": index,
                "reason": reason,
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

        if reviewed.get("selected_face_index") != 0:
            skipped.append({
                "dialogue_index": index,
                "reason": "Tek yüzlü çekim için yüz indeksi 0 olmalı.",
            })
            continue

        entry = dict(original)
        entry["face_count"] = 1
        entry["target_face_confirmed"] = True
        entry["selected_face_index"] = 0
        entry["face_approval_method"] = "manual_review"
        entry["face_approval_version"] = VERSION

        approved_dialogue.append(entry)
        approved_indices.append(index)

    if not approved_dialogue:
        raise RuntimeError(
            "Onaylanan diyalog bulunamadı. "
            "Hiçbir çıktı paketi oluşturulmadı."
        )

    output_package.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, output_package)

    updated_timeline = dict(timeline)
    updated_timeline["dialogue"] = approved_dialogue
    updated_timeline["status"] = "package_ready"
    updated_timeline["lip_sync_rendered"] = False
    updated_timeline["face_approval"] = {
        "version": VERSION,
        "method": "manual_review",
        "approved_dialogue_count": len(approved_dialogue),
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
        "approved_dialogue_count": len(approved_dialogue),
        "skipped_dialogue_count": len(skipped),
        "approved_dialogue_indices": approved_indices,
        "skipped": skipped,
        "output_package": str(output_package),
        "note": (
            "Only manually approved single-face dialogue entries "
            "were copied. No automatic speaker identity matching "
            "or approval was performed."
        ),
    }

    write_json(
        output_package / "face_approval_report.json",
        report,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--package", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output-package", required=True)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Yüz onaylarını terminalden tek tek al.",
    )

    args = parser.parse_args()

    package = Path(args.package).resolve()
    review_path = Path(args.review).resolve()

    try:
        if args.interactive:
            timeline = read_json(package / "lip_sync_timeline.json")
            review = read_json(review_path)

            original_dialogue = timeline.get("dialogue")
            reviewed_dialogue = review.get("dialogue")

            if not isinstance(original_dialogue, list):
                raise RuntimeError("Kaynak diyalog listesi geçersiz.")

            if not isinstance(reviewed_dialogue, list):
                raise RuntimeError("İnceleme diyalog listesi geçersiz.")

            collect_interactive_approvals(
                original_dialogue,
                reviewed_dialogue,
                review_path,
            )

        approve(
            package=package,
            review_path=review_path,
            output_package=Path(args.output_package),
        )

    except Exception as exc:
        raise SystemExit(f"HATA: {exc}") from exc


if __name__ == "__main__":
    main()

