
"""Prepare, validate and privately publish character references for Wan."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from character_asset_generator import generate_character_assets
from project_adapter import legacy_to_pipeline_project
from production_orchestrator import ProductionJob
from wan_kaggle_payload import prepare_reference_payload
from wan_kaggle_publisher import publish_reference_dataset
from wan_orchestrator_bridge import _register_character_assets
from wan_project_input import normalize_wan_project
from wan_reference_plan import write_vace_reference_plan


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_slug(value: str, label: str) -> str:
    value = value.strip().lower()

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,49}", value):
        raise ValueError(
            f"{label} geçersiz. Yalnızca küçük harf, rakam ve tire kullan."
        )

    return value


def prepare_and_publish(
    project_json: str | Path,
    workspace: str | Path,
    payload_output: str | Path,
    dataset_output: str | Path,
    *,
    kaggle_owner: str,
    dataset_slug: str,
    kernel_metadata: str | Path,
) -> dict[str, Any]:

    project_file = Path(project_json).resolve()
    workspace_path = Path(workspace).resolve()
    payload_path = Path(payload_output).resolve()
    dataset_path = Path(dataset_output).resolve()
    metadata_path = Path(kernel_metadata).resolve()

    if not project_file.is_file():
        raise FileNotFoundError(
            f"Film proje JSON dosyası bulunamadı: {project_file}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Kaggle metadata dosyası bulunamadı: {metadata_path}"
        )

    for path, label in (
        (workspace_path, "Üretim çalışma alanı"),
        (payload_path, "Referans paketi"),
        (dataset_path, "Kaggle veri kümesi klasörü"),
    ):
        if path.exists():
            raise FileExistsError(
                f"{label} zaten mevcut; üzerine yazılmayacak: {path}"
            )

    owner = _safe_slug(kaggle_owner, "Kaggle kullanıcı adı")
    slug = _safe_slug(dataset_slug, "Kaggle veri kümesi adı")

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Kaggle metadata okunamadı: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError("Kaggle metadata bir JSON nesnesi olmalıdır.")

    if metadata.get("is_private") is not True:
        raise ValueError(
            "Güvenlik kontrolü: Kaggle kernel private olmalıdır."
        )

    if not isinstance(metadata.get("dataset_sources"), list):
        raise ValueError(
            "Kaggle dataset_sources alanı liste olmalıdır."
        )

    raw_project = json.loads(
        project_file.read_text(encoding="utf-8")
    )

    # Hem desteklenen modern hem de eski proje biçimini doğrula.
    legacy_project = normalize_wan_project(raw_project)
    project = legacy_to_pipeline_project(legacy_project)

    if not project.characters:
        raise ValueError(
            "Referans üretimi için projede karakter bulunmalıdır."
        )

    if not project.scenes:
        raise ValueError(
            "Referans üretimi için projede sahne bulunmalıdır."
        )

    all_character_ids = [
        character.id for character in project.characters
    ]

    # Diyalog bilgisi olmayan sahnelerde proje karakterlerini kullan.
    for scene in project.scenes:
        if not scene.character_ids:
            scene.character_ids = list(all_character_ids)

    project_data = project.to_dict()

    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.mkdir(parents=True, exist_ok=False)

    job = ProductionJob.create(
        workspace_path,
        project_data,
    )

    inputs = workspace_path / "artifacts" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    # 1. Proje doğrulama aşaması.
    job.start_stage("project_validation")

    validation_path = inputs / "project_validation.json"

    _write_json(validation_path, {
        "valid": True,
        "project_id": project.project_id,
        "project_title": project.title,
        "character_count": len(project.characters),
        "scene_count": len(project.scenes),
        "source": "wan_kaggle_workflow_prepare",
    })

    job.complete_stage(
        "project_validation",
        ["artifacts/inputs/project_validation.json"],
    )

    # 2. Senaryo/proje girdisi aşaması.
    job.start_stage("script")

    script_path = inputs / "script_input.json"
    _write_json(script_path, project_data)

    definitions_path = inputs / "character_definitions.json"

    _write_json(definitions_path, [
        {
            "id": character.id,
            "name": character.name,
            "visual_description": character.visual_description,
            "appearance": character.appearance,
            "clothing": character.clothing,
            "personality": character.personality,
            "continuity_prompt": character.continuity_prompt,
        }
        for character in project.characters
    ])

    job.complete_stage(
        "script",
        [
            "artifacts/inputs/script_input.json",
            "artifacts/inputs/character_definitions.json",
        ],
    )

    # 3. Karakter referanslarını üret ve doğrula.
    job.start_stage("character_assets")

    source_assets = workspace_path / "generated_character_assets"

    try:
        manifest = generate_character_assets(
            project_data,
            source_assets,
            engines=["cloudflare"],
        )

        if manifest.get("all_assets_validated") is not True:
            raise RuntimeError(
                "Karakter görsellerinin tamamı doğrulanmadı."
            )

        if manifest.get("asset_count") != len(project.characters):
            raise RuntimeError(
                "Üretilen karakter görseli sayısı proje ile eşleşmiyor."
            )

        registered_artifacts = _register_character_assets(
            job.project,
            source_assets,
            job,
        )

        job.complete_stage(
            "character_assets",
            registered_artifacts,
        )

    except Exception as exc:
        try:
            job.fail_stage("character_assets", exc)
        except Exception:
            pass
        raise

    # 4. Sahne-karakter referans planını oluştur.
    plan_path = write_vace_reference_plan(
        job.project.to_dict(),
        workspace_path,
    )

    if not plan_path.is_file():
        raise RuntimeError(
            "Karakter referans planı oluşturulamadı."
        )

    # 5. Kaggle'a taşınabilir, hash doğrulamalı paket hazırla.
    bundle_path = prepare_reference_payload(
        plan_path,
        workspace_path,
        payload_path,
    )

    if not bundle_path.is_file():
        raise RuntimeError(
            "Referans paketi oluşturulamadı."
        )

    bundle = json.loads(
        bundle_path.read_text(encoding="utf-8")
    )

    if bundle.get("scene_count") != len(project.scenes):
        raise RuntimeError(
            "Referans paketindeki sahne sayısı eşleşmiyor."
        )

    if bundle.get("unique_reference_count", 0) < 1:
        raise RuntimeError(
            "Referans paketinde kullanılabilir görsel bulunamadı."
        )

    # 6. Kaggle'a private veri kümesi olarak yükle.
    publication = publish_reference_dataset(
        payload_dir=payload_path,
        output_dir=dataset_path,
        owner=owner,
        slug=slug,
        kernel_metadata_path=metadata_path,
        title=f"Cihan Film {slug[-20:]}",
    )

    if publication.get("privacy") != "private":
        raise RuntimeError(
            "Güvenlik kontrolü başarısız: veri kümesi private değil."
        )

    if publication.get("kernel_metadata_updated") is not True:
        raise RuntimeError(
            "Veri kümesi yüklendi ancak Worker metadata bağlantısı "
            "doğrulanamadı."
        )

    result = {
        "status": "published",
        "project_id": project.project_id,
        "project_title": project.title,
        "character_count": len(project.characters),
        "generated_reference_count": manifest["asset_count"],
        "scene_count": bundle["scene_count"],
        "scene_reference_count": bundle["scene_reference_count"],
        "dataset_id": publication["dataset_id"],
        "dataset_privacy": publication["privacy"],
        "kernel_metadata_updated": publication[
            "kernel_metadata_updated"
        ],
        "reference_plan": str(plan_path),
        "reference_bundle": str(bundle_path),
    }

    print(
        json.dumps(result, ensure_ascii=False, indent=2),
        flush=True,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Karakter referanslarını üretir, doğrular ve "
            "Kaggle'a private olarak yükler."
        )
    )

    parser.add_argument("--project-json", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--payload-output", required=True)
    parser.add_argument("--dataset-output", required=True)
    parser.add_argument("--kaggle-owner", required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--kernel-metadata", required=True)

    args = parser.parse_args()

    prepare_and_publish(
        args.project_json,
        args.workspace,
        args.payload_output,
        args.dataset_output,
        kaggle_owner=args.kaggle_owner,
        dataset_slug=args.dataset_slug,
        kernel_metadata=args.kernel_metadata,
    )


if __name__ == "__main__":
    main()

