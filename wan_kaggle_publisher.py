
"""Publish a verified Wan reference bundle to Kaggle and attach it to a kernel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from wan_kaggle_dataset import (
    WanDatasetError,
    prepare_kaggle_dataset,
)


class WanKagglePublishError(RuntimeError):
    """Raised when a private Kaggle reference dataset cannot be published."""


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def _attach_dataset_to_kernel(
    metadata_path: str | Path,
    dataset_id: str,
) -> None:
    """Add a published dataset to Kaggle kernel metadata safely."""

    path = Path(metadata_path).resolve()

    if not path.is_file():
        raise WanKagglePublishError(
            f"Kaggle kernel metadata not found: {path}"
        )

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WanKagglePublishError(
            "Kaggle kernel metadata is unreadable."
        ) from exc

    if not isinstance(metadata, dict):
        raise WanKagglePublishError(
            "Kaggle kernel metadata must be a JSON object."
        )

    if metadata.get("is_private") is not True:
        raise WanKagglePublishError(
            "The Kaggle kernel must remain private."
        )

    sources = metadata.get("dataset_sources")

    if not isinstance(sources, list):
        raise WanKagglePublishError(
            "dataset_sources must be a JSON list."
        )

    if not all(isinstance(item, str) for item in sources):
        raise WanKagglePublishError(
            "dataset_sources contains an invalid entry."
        )

    if dataset_id not in sources:
        sources.append(dataset_id)

    metadata["dataset_sources"] = sources
    _write_json_atomic(path, metadata)


def publish_reference_dataset(
    payload_dir: str | Path,
    output_dir: str | Path,
    *,
    owner: str,
    slug: str,
    kernel_metadata_path: str | Path | None = None,
    title: str | None = None,
    kaggle_executable: str = "kaggle",
) -> dict[str, Any]:
    """Build and publish a private dataset, then optionally attach it to a kernel.

    Kaggle datasets are private by default. This function deliberately never
    passes the --public flag.
    """

    executable = shutil.which(kaggle_executable)

    if executable is None:
        raise WanKagglePublishError(
            f"Kaggle CLI executable was not found: {kaggle_executable}"
        )

    if not os.environ.get("KAGGLE_API_TOKEN") and not (
        Path.home() / ".kaggle" / "access_token"
    ).is_file() and not (Path.home() / ".kaggle" / "kaggle.json").is_file():
        raise WanKagglePublishError(
            "Kaggle authentication is missing. Configure the Kaggle API "
            "token in the environment or the standard Kaggle credentials file."
        )

    dataset_dir = Path(output_dir).resolve()

    try:
        prepare_kaggle_dataset(
            payload_dir=payload_dir,
            output_dir=dataset_dir,
            owner=owner,
            slug=slug,
            title=title,
        )
    except WanDatasetError:
        raise
    except Exception as exc:
        raise WanKagglePublishError(
            f"Could not prepare the Kaggle dataset: {exc}"
        ) from exc

    dataset_id = f"{owner}/{slug}"

    command = [
        executable,
        "datasets",
        "create",
        "-p",
        str(dataset_dir),
        "-r",
        "zip",
    ]

    print("Publishing private Kaggle dataset:", dataset_id, flush=True)
    print("Dataset directory:", dataset_dir, flush=True)
    print("Privacy: private by default; --public is not used.", flush=True)

    try:
        subprocess.run(
            command,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise WanKagglePublishError(
            "Kaggle dataset upload failed. Check the Kaggle CLI output "
            "and confirm that the dataset ID is not already in use."
        ) from exc
    except OSError as exc:
        raise WanKagglePublishError(
            f"Could not start Kaggle CLI: {exc}"
        ) from exc

    if kernel_metadata_path is not None:
        _attach_dataset_to_kernel(
            kernel_metadata_path,
            dataset_id,
        )

    result = {
        "dataset_id": dataset_id,
        "dataset_directory": str(dataset_dir),
        "privacy": "private",
        "published": True,
        "kernel_metadata_updated": kernel_metadata_path is not None,
    }

    print("Kaggle dataset published successfully.", flush=True)
    print("Dataset ID:", dataset_id, flush=True)

    if kernel_metadata_path is not None:
        print("Kernel metadata updated.", flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a verified Wan reference bundle as a private "
            "Kaggle dataset."
        )
    )

    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--kernel-metadata")
    parser.add_argument("--title")
    parser.add_argument("--kaggle-executable", default="kaggle")

    args = parser.parse_args()

    result = publish_reference_dataset(
        payload_dir=args.payload,
        output_dir=args.output,
        owner=args.owner,
        slug=args.slug,
        kernel_metadata_path=args.kernel_metadata,
        title=args.title,
        kaggle_executable=args.kaggle_executable,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
