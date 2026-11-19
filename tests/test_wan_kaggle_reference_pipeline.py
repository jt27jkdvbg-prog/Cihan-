
"""End-to-end tests for the Wan reference packaging pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wan_kaggle_dataset import prepare_kaggle_dataset
from wan_kaggle_payload import prepare_reference_payload
from wan_kaggle_reference_loader import (
    WanReferenceLoadError,
    load_reference_bundle,
)


def _write_test_image(path: Path) -> str:
    """Create a deterministic JPEG-like fixture for header validation."""
    path.parent.mkdir(parents=True, exist_ok=True)

    content = (
        b"\xff\xd8\xff"
        + b"CIHAN-WAN-REFERENCE-TEST-" * 300
        + b"\xff\xd9"
    )

    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _create_workspace(root: Path) -> tuple[Path, str]:
    workspace = root / "workspace"
    workspace.mkdir()

    image_path = workspace / "source_assets" / "character_anchor.jpg"
    digest = _write_test_image(image_path)

    relative_image = image_path.relative_to(workspace).as_posix()

    plan = {
        "schema_version": 1,
        "project_id": "integration-test-project",
        "project_title": "Wan Reference Integration Test",
        "status": "ready",
        "engine": "wan-vace-r2v",
        "model_profile": "Wan2.1-VACE-1.3B",
        "conditioning_mode": "reference_to_video",
        "resolution": "832*480",
        "frame_num": 81,
        "character_count": 1,
        "scene_count": 2,
        "unique_reference_count": 1,
        "scene_reference_count": 2,
        "scenes": [
            {
                "scene_id": "scene-001",
                "order": 1,
                "duration_seconds": 5,
                "character_ids": ["character-001"],
                "reference_images": [
                    {
                        "character_id": "character-001",
                        "character_name": "Test Character",
                        "path": relative_image,
                        "sha256": digest,
                        "format": "jpg",
                    }
                ],
                "reference_conditioning_required": True,
                "visual_prompt": "A cinematic establishing shot.",
            },
            {
                "scene_id": "scene-002",
                "order": 2,
                "duration_seconds": 5,
                "character_ids": ["character-001"],
                "reference_images": [
                    {
                        "character_id": "character-001",
                        "character_name": "Test Character",
                        "path": relative_image,
                        "sha256": digest,
                        "format": "jpg",
                    }
                ],
                "reference_conditioning_required": True,
                "visual_prompt": "The same character walks forward.",
            },
        ],
    }

    plan_path = workspace / "artifacts" / "wan" / "character_reference_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return workspace, digest


def test_reference_plan_payload_dataset_loader_round_trip(tmp_path: Path):
    """The payload and dataset builders must produce loader-compatible data."""
    workspace, expected_hash = _create_workspace(tmp_path)

    payload_dir = tmp_path / "payload"
    dataset_dir = tmp_path / "kaggle_dataset"

    plan_path = (
        workspace
        / "artifacts"
        / "wan"
        / "character_reference_plan.json"
    )

    bundle_path = prepare_reference_payload(
        plan_path=plan_path,
        workspace=workspace,
        output_dir=payload_dir,
    )

    assert bundle_path.is_file()

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["project_id"] == "integration-test-project"
    assert bundle["scene_count"] == 2
    assert bundle["unique_reference_count"] == 1
    assert bundle["scene_reference_count"] == 2

    metadata_path = prepare_kaggle_dataset(
        payload_dir=payload_dir,
        output_dir=dataset_dir,
        owner="cihan-test",
        slug="wan-reference-test",
    )

    assert metadata_path.is_file()
    assert (dataset_dir / "reference_bundle.json").is_file()
    assert (dataset_dir / "reference_assets").is_dir()

    loaded = load_reference_bundle(
        dataset_dir,
        expected_project_id="integration-test-project",
    )

    assert loaded["runtime_validation"]["status"] == "verified"
    assert loaded["runtime_validation"]["images_verified"] == 1
    assert loaded["runtime_validation"]["scene_references_verified"] == 2

    assert len(loaded["scenes"]) == 2

    for scene in loaded["scenes"]:
        references = scene["reference_images"]
        assert len(references) == 1
        assert references[0]["sha256"] == expected_hash
        assert Path(references[0]["local_path"]).is_file()


def test_loader_rejects_tampered_dataset_image(tmp_path: Path):
    """The runtime loader must reject an image changed after packaging."""
    workspace, _ = _create_workspace(tmp_path)

    payload_dir = tmp_path / "payload"
    dataset_dir = tmp_path / "kaggle_dataset"

    plan_path = (
        workspace
        / "artifacts"
        / "wan"
        / "character_reference_plan.json"
    )

    prepare_reference_payload(
        plan_path=plan_path,
        workspace=workspace,
        output_dir=payload_dir,
    )

    prepare_kaggle_dataset(
        payload_dir=payload_dir,
        output_dir=dataset_dir,
        owner="cihan-test",
        slug="wan-reference-test",
    )

    image_path = next(
        (dataset_dir / "reference_assets").iterdir()
    )

    image_path.write_bytes(
        image_path.read_bytes() + b"TAMPERED"
    )

    with pytest.raises(WanReferenceLoadError):
        load_reference_bundle(
            dataset_dir,
            expected_project_id="integration-test-project",
        )


def test_loader_rejects_wrong_project_id(tmp_path: Path):
    """A dataset from another project must not be accepted."""
    workspace, _ = _create_workspace(tmp_path)

    payload_dir = tmp_path / "payload"
    dataset_dir = tmp_path / "kaggle_dataset"

    plan_path = (
        workspace
        / "artifacts"
        / "wan"
        / "character_reference_plan.json"
    )

    prepare_reference_payload(
        plan_path=plan_path,
        workspace=workspace,
        output_dir=payload_dir,
    )

    prepare_kaggle_dataset(
        payload_dir=payload_dir,
        output_dir=dataset_dir,
        owner="cihan-test",
        slug="wan-reference-test",
    )

    with pytest.raises(WanReferenceLoadError):
        load_reference_bundle(
            dataset_dir,
            expected_project_id="a-different-project",
        )
