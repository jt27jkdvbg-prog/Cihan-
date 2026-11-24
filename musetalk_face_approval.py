
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MuseTalk Face Approval v1.2

İnsan tarafından incelenmiş yüzleri onaylar.
Otomatik yüz veya konuşmacı eşleştirmesi yapmaz.

Aşamalar:
1. --create-template ile onay şablonu üret.
2. JSON dosyasını manuel olarak düzenle.
3. --approval-manifest ile onayları uygula.
"""

from __future__ import annotations

import argparse
import json
import shutil

from pathlib import Path
from typing import Any


VERSION = "1.2.0"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {path}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"JSON okunamadı: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Geçersiz JSON: {path}"
        )

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

    if (
        not expected_speaker
        or reviewed_speaker != expected_speaker
    ):
        return False, "Konuşmacı kimliği eşleşmiyor."

    try:
        original_start = float(
            original["film_start_seconds"]
        )
        original_end = float(
            original["film_end_seconds"]
        )
        reviewed_start = float(
            reviewed["film_start_seconds"]
        )
        reviewed_end = float(
            reviewed["film_end_seconds"]
        )
    except (KeyError, TypeError, ValueError):
        return False, "Zaman bilgileri geçersiz."

    if abs(original_start - reviewed_start) > 0.05:
        return False, "Başlangıç zamanı eşleşmiyor."

    if abs(original_end - reviewed_end) > 0.05:
        return False, "Bitiş zamanı eşleşmiyor."

    samples = reviewed.get("sample_frames")

    if not isinstance(samples, list) or len(samples) != 3:
        return False, "Üç örnek kare bulunamadı."

    try:
        counts = [
            int(sample["face_count"])
            for sample in samples
        ]
    except (KeyError, TypeError, ValueError):
        return False, "Yüz sayıları geçersiz."

    if counts != [1, 1, 1]:
        return False, (
            "Üç örnek karenin tamamında tek yüz bulunmuyor."
        )

    return True, "İnceleme geçerli."


def build_review_index(
    review_dialogue: list,
) -> dict[int, dict]:

    result = {}

    for item in review_dialogue:

        if not isinstance(item, dict):
            raise RuntimeError(
                "İnceleme kaydı geçersiz."
            )

        index = item.get("dialogue_index")

        if not isinstance(index, int) or index < 0:
            raise RuntimeError(
                "Diyalog indeksi geçersiz."
            )

        if index in result:
            raise RuntimeError(
                f"Tekrarlanan diyalog indeksi: {index}"
            )

        result[index] = item

    return result


def create_approval_template(
    timeline_path: Path,
    review_path: Path,
    output_path: Path,
) -> dict[str, Any]:

    timeline = read_json(timeline_path)
    review = read_json(review_path)

    if timeline.get("status") != "package_ready":
        raise RuntimeError(
            "Kaynak paket package_ready durumunda değil."
        )

    if review.get("status") != "face_review_ready":
        raise RuntimeError(
            "Yüz raporu face_review_ready durumunda değil."
        )

    original_dialogue = timeline.get("dialogue")
    reviewed_dialogue = review.get("dialogue")

    if (
        not isinstance(original_dialogue, list)
        or not original_dialogue
    ):
        raise RuntimeError(
            "Kaynak diyalog listesi boş."
        )

    if not isinstance(reviewed_dialogue, list):
        raise RuntimeError(
            "İnceleme diyalog listesi geçersiz."
        )

    review_by_index = build_review_index(
        reviewed_dialogue
    )

    approvals = []

    for index, original in enumerate(original_dialogue):

        if not isinstance(original, dict):
            continue

        reviewed = review_by_index.get(index)

        if reviewed is None:
            continue

        valid, reason = validate_review_record(
            original,
            reviewed,
            index,
        )

        samples = reviewed.get("sample_frames", [])

        previews = [
            item.get("preview", "")
            for item in samples
            if isinstance(item, dict)
        ]

        approvals.append({
            "dialogue_index": index,
            "speaker_id": str(
                original.get("speaker_id", "")
            ),
            "text": str(
                original.get("text", "")
            ),
            "preview_images": previews,
            "eligible_for_review": valid,
            "validation_note": reason,
            "approved": False,
            "confirmed_speaker_id": "",
            "selected_face_index": None,
            "human_review_note": "",
        })

    manifest = {
        "version": VERSION,
        "status": "awaiting_human_review",
        "instructions": (
            "Görselleri incele. Onayladığın kayıtta "
            "approved değerini true yap ve "
            "confirmed_speaker_id alanına speaker_id "
            "değerini aynen yaz. Tek yüzlü örneklerde "
            "selected_face_index değeri 0 olmalıdır. "
            "Emin olmadığın kayıtları onaylama."
        ),
        "automatic_approval": False,
        "approvals": approvals,
    }

    write_json(
        output_path,
        manifest,
    )

    result = {
        "status": "template_created",
        "template": str(output_path.resolve()),
        "dialogue_count": len(approvals),
        "eligible_count": sum(
            1 for item in approvals
            if item["eligible_for_review"]
        ),
        "approved_count": 0,
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


def apply_approval_manifest(
    package: Path,
    review_path: Path,
    manifest_path: Path,
    output_package: Path,
) -> dict[str, Any]:

    package = package.resolve()
    review_path = review_path.resolve()
    manifest_path = manifest_path.resolve()
    output_package = output_package.resolve()

    timeline_path = (
        package / "lip_sync_timeline.json"
    )

    video_path = package / "video.mp4"

    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if output_package.exists():
        raise FileExistsError(
            f"Çıktı klasörü zaten mevcut: {output_package}"
        )

    timeline = read_json(timeline_path)
    review = read_json(review_path)
    manifest = read_json(manifest_path)

    if timeline.get("status") != "package_ready":
        raise RuntimeError(
            "Kaynak zaman çizelgesi package_ready değil."
        )

    if review.get("status") != "face_review_ready":
        raise RuntimeError(
            "Yüz inceleme raporu geçerli değil."
        )

    if manifest.get("version") != VERSION:
        raise RuntimeError(
            "Onay manifest sürümü uyumsuz. "
            f"Beklenen: {VERSION}"
        )

    if manifest.get("automatic_approval") is not False:
        raise RuntimeError(
            "Otomatik onay kabul edilmiyor."
        )

    original_dialogue = timeline.get("dialogue")
    reviewed_dialogue = review.get("dialogue")
    approvals = manifest.get("approvals")

    if not isinstance(original_dialogue, list):
        raise RuntimeError(
            "Kaynak diyalog listesi geçersiz."
        )

    if not isinstance(reviewed_dialogue, list):
        raise RuntimeError(
            "İnceleme diyalog listesi geçersiz."
        )

    if not isinstance(approvals, list):
        raise RuntimeError(
            "Onay listesi geçersiz."
        )

    review_by_index = build_review_index(
        reviewed_dialogue
    )

    approval_by_index = {}

    for item in approvals:

        if not isinstance(item, dict):
            raise RuntimeError(
                "Onay kaydı geçersiz."
            )

        index = item.get("dialogue_index")

        if (
            not isinstance(index, int)
            or index < 0
            or index in approval_by_index
        ):
            raise RuntimeError(
                "Eksik veya tekrarlanan onay indeksi."
            )

        approval_by_index[index] = item

    approved_dialogue = []
    approved_indices = []
    skipped = []

    for index, original in enumerate(original_dialogue):

        if not isinstance(original, dict):
            skipped.append({
                "dialogue_index": index,
                "reason": "Kaynak kayıt geçersiz.",
            })
            continue

        approval = approval_by_index.get(index)
        reviewed = review_by_index.get(index)

        if approval is None or reviewed is None:
            skipped.append({
                "dialogue_index": index,
                "reason": "Onay veya inceleme kaydı yok.",
            })
            continue

        if approval.get("approved") is not True:
            skipped.append({
                "dialogue_index": index,
                "reason": "İnsan onayı verilmedi.",
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

        expected_speaker = str(
            original.get("speaker_id", "")
        ).strip()

        confirmed_speaker = str(
            approval.get("confirmed_speaker_id", "")
        ).strip()

        if (
            not expected_speaker
            or confirmed_speaker != expected_speaker
            or approval.get("speaker_id") != expected_speaker
        ):
            skipped.append({
                "dialogue_index": index,
                "reason": "Konuşmacı onayı eşleşmiyor.",
            })
            continue

        if approval.get("eligible_for_review") is not True:
            skipped.append({
                "dialogue_index": index,
                "reason": "Kayıt incelemeye uygun değil.",
            })
            continue

        if approval.get("selected_face_index") != 0:
            skipped.append({
                "dialogue_index": index,
                "reason": "Yüz indeksi 0 olarak onaylanmalı.",
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

    output_package.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        package,
        output_package,
    )

    updated_timeline = dict(timeline)

    updated_timeline["dialogue"] = approved_dialogue
    updated_timeline["status"] = "package_ready"
    updated_timeline["lip_sync_rendered"] = False

    updated_timeline["face_approval"] = {
        "version": VERSION,
        "method": "manual_review_manifest",
        "approved_dialogue_count": len(
            approved_dialogue
        ),
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
        "approved_dialogue_count": len(
            approved_dialogue
        ),
        "skipped_dialogue_count": len(skipped),
        "approved_dialogue_indices": approved_indices,
        "skipped": skipped,
        "output_package": str(output_package),
        "automatic_approval": False,
        "note": (
            "Only manually approved dialogue entries "
            "were copied. No automatic face or speaker "
            "identity matching was performed."
        ),
    }

    write_json(
        output_package / "face_approval_report.json",
        report,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    return report


def main() -> None:

    parser = argparse.ArgumentParser(
        description="MuseTalk manuel yüz onay sistemi."
    )

    parser.add_argument(
        "--package",
        required=True,
    )

    parser.add_argument(
        "--review",
        required=True,
    )

    parser.add_argument(
        "--output-package",
    )

    parser.add_argument(
        "--create-template",
        action="store_true",
    )

    parser.add_argument(
        "--approval-manifest",
    )

    args = parser.parse_args()

    package = Path(args.package).resolve()
    review_path = Path(args.review).resolve()

    try:

        if args.create_template:

            output_path = Path(
                args.approval_manifest
                or "face_approval_manifest.json"
            )

            create_approval_template(
                package / "lip_sync_timeline.json",
                review_path,
                output_path,
            )

            return

        if not args.approval_manifest:
            raise RuntimeError(
                "--approval-manifest gerekli."
            )

        if not args.output_package:
            raise RuntimeError(
                "--output-package gerekli."
            )

        apply_approval_manifest(
            package=package,
            review_path=review_path,
            manifest_path=Path(
                args.approval_manifest
            ),
            output_package=Path(
                args.output_package
            ),
        )

    except Exception as exc:
        raise SystemExit(
            f"HATA: {exc}"
        ) from exc


if __name__ == "__main__":
    main()


