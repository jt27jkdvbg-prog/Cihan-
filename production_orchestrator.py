
"""Durable job state and checkpoint orchestration for film production."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from pipeline_config import FilmProject, ProjectConfigError


ORCHESTRATOR_SCHEMA_VERSION = 1

STAGES = (
    "planning",
    "script",
    "visual_assets",
    "video_generation",
    "audio_generation",
    "post_production",
    "quality_control",
    "publishing",
)

DEPENDENCIES = {
    "planning": (),
    "script": ("planning",),
    "visual_assets": ("script",),
    "video_generation": ("visual_assets",),
    "audio_generation": ("script",),
    "post_production": (
        "video_generation",
        "audio_generation",
    ),
    "quality_control": ("post_production",),
    "publishing": ("quality_control",),
}


class OrchestrationError(RuntimeError):
    """Raised when a production job cannot safely advance."""


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


@contextmanager
def _job_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        descriptor = os.open(
            str(path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise OrchestrationError(
            "This job is already being modified by another process."
        ) from exc

    try:
        os.write(
            descriptor,
            str(os.getpid()).encode("ascii"),
        )
        yield
    finally:
        os.close(descriptor)

        try:
            path.unlink()
        except FileNotFoundError:
            pass


class ProductionJob:
    """Validated project and durable production checkpoints."""

    def __init__(
        self,
        workspace: str | Path,
    ):
        self.workspace = Path(workspace).resolve()

        self.manifest_path = (
            self.workspace / "job_manifest.json"
        )

        self.project_path = (
            self.workspace / "project.json"
        )

        self.lock_path = (
            self.workspace / ".job.lock"
        )

    @classmethod
    def create(
        cls,
        workspace: str | Path,
        project_data: Mapping[str, Any],
    ) -> "ProductionJob":

        project = FilmProject.from_dict(project_data)

        job = cls(workspace)

        job.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

        if job.manifest_path.exists():
            raise OrchestrationError(
                f"A job already exists at {job.workspace}."
            )

        project_dict = project.to_dict()

        _atomic_json(
            job.project_path,
            project_dict,
        )

        now = time.time()

        manifest = {
            "orchestrator_schema_version": (
                ORCHESTRATOR_SCHEMA_VERSION
            ),
            "job_id": re.sub(
                r"[^a-zA-Z0-9._-]+",
                "-",
                project.project_id,
            ).strip("-"),
            "project_title": project.title,
            "project_sha256": _json_hash(project_dict),
            "created_at": now,
            "updated_at": now,
            "status": "queued",
            "stages": {
                name: {
                    "status": "pending",
                    "attempts": 0,
                    "artifacts": [],
                    "error": None,
                }
                for name in STAGES
            },
        }

        _atomic_json(
            job.manifest_path,
            manifest,
        )

        return job

    @classmethod
    def open(
        cls,
        workspace: str | Path,
    ) -> "ProductionJob":

        job = cls(workspace)

        manifest = job._read_manifest()
        project = job._read_project()

        if _json_hash(project) != manifest["project_sha256"]:
            raise OrchestrationError(
                "Project JSON changed after job creation. "
                "Unsafe resume refused."
            )

        return job

    def _read_manifest(self) -> dict[str, Any]:

        if not self.manifest_path.is_file():
            raise OrchestrationError(
                "Job manifest was not found."
            )

        try:
            data = json.loads(
                self.manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise OrchestrationError(
                f"Job manifest is unreadable: {exc}"
            ) from exc

        if (
            data.get("orchestrator_schema_version")
            != ORCHESTRATOR_SCHEMA_VERSION
        ):
            raise OrchestrationError(
                "Unsupported job manifest version."
            )

        if set(data.get("stages", {})) != set(STAGES):
            raise OrchestrationError(
                "Job manifest stage set is invalid."
            )

        return data

    def _read_project(self) -> dict[str, Any]:

        try:
            data = json.loads(
                self.project_path.read_text(
                    encoding="utf-8"
                )
            )

            FilmProject.from_dict(data)

            return data

        except (
            OSError,
            json.JSONDecodeError,
            ProjectConfigError,
        ) as exc:
            raise OrchestrationError(
                f"Stored project is invalid: {exc}"
            ) from exc

    @property
    def manifest(self) -> dict[str, Any]:
        return self._read_manifest()

    @property
    def project(self) -> FilmProject:
        return FilmProject.from_dict(
            self._read_project()
        )

    def start_stage(self, stage: str) -> int:

        if stage not in DEPENDENCIES:
            raise OrchestrationError(
                f"Unknown production stage: {stage}"
            )

        with _job_lock(self.lock_path):

            data = self._read_manifest()
            current = data["stages"][stage]

            if current["status"] == "running":
                raise OrchestrationError(
                    f"Stage '{stage}' is already running."
                )

            if current["status"] == "succeeded":
                raise OrchestrationError(
                    f"Stage '{stage}' already succeeded."
                )

            incomplete = [
                dependency
                for dependency in DEPENDENCIES[stage]
                if (
                    data["stages"][dependency]["status"]
                    != "succeeded"
                )
            ]

            if incomplete:
                raise OrchestrationError(
                    "Stage dependencies are incomplete: "
                    + ", ".join(incomplete)
                )

            current.update(
                status="running",
                attempts=current["attempts"] + 1,
                started_at=time.time(),
                error=None,
            )

            data["status"] = "running"
            data["updated_at"] = time.time()

            _atomic_json(
                self.manifest_path,
                data,
            )

            return current["attempts"]

    def complete_stage(
        self,
        stage: str,
        artifacts: list[str] | None = None,
    ) -> None:

        if stage not in DEPENDENCIES:
            raise OrchestrationError(
                f"Unknown production stage: {stage}"
            )

        verified = []

        for relative in artifacts or []:

            candidate = (
                self.workspace / relative
            ).resolve()

            if self.workspace not in candidate.parents:
                raise OrchestrationError(
                    "Artifact path must stay inside the job workspace."
                )

            if (
                not candidate.is_file()
                or candidate.stat().st_size == 0
            ):
                raise OrchestrationError(
                    f"Required artifact is missing or empty: {relative}"
                )

            digest = hashlib.sha256(
                candidate.read_bytes()
            ).hexdigest()

            verified.append({
                "path": str(
                    candidate.relative_to(self.workspace)
                ),
                "size_bytes": candidate.stat().st_size,
                "sha256": digest,
            })

        with _job_lock(self.lock_path):

            data = self._read_manifest()
            current = data["stages"][stage]

            if current["status"] != "running":
                raise OrchestrationError(
                    f"Stage '{stage}' must be running."
                )

            current.update(
                status="succeeded",
                finished_at=time.time(),
                artifacts=verified,
                error=None,
            )

            data["status"] = (
                "completed"
                if all(
                    item["status"] == "succeeded"
                    for item in data["stages"].values()
                )
                else "running"
            )

            data["updated_at"] = time.time()

            _atomic_json(
                self.manifest_path,
                data,
            )

    def fail_stage(
        self,
        stage: str,
        error: BaseException | str,
    ) -> None:

        if stage not in DEPENDENCIES:
            raise OrchestrationError(
                f"Unknown production stage: {stage}"
            )

        message = (
            str(error).strip()
            or type(error).__name__
        )

        with _job_lock(self.lock_path):

            data = self._read_manifest()
            current = data["stages"][stage]

            if current["status"] != "running":
                raise OrchestrationError(
                    f"Stage '{stage}' is not running."
                )

            current.update(
                status="failed",
                finished_at=time.time(),
                error=message[:2000],
            )

            data["status"] = "failed"
            data["updated_at"] = time.time()

            _atomic_json(
                self.manifest_path,
                data,
            )

    def retry_failed_stage(
        self,
        stage: str,
    ) -> None:

        if stage not in DEPENDENCIES:
            raise OrchestrationError(
                f"Unknown production stage: {stage}"
            )

        with _job_lock(self.lock_path):

            data = self._read_manifest()
            current = data["stages"][stage]

            if current["status"] != "failed":
                raise OrchestrationError(
                    "Only a failed stage can be reset for retry."
                )

            current.update(
                status="pending",
                error=None,
                artifacts=[],
            )

            data["status"] = "queued"
            data["updated_at"] = time.time()

            _atomic_json(
                self.manifest_path,
                data,
            )

    def verify_artifacts(self) -> list[str]:

        problems = []
        data = self._read_manifest()

        for stage_name, stage in data["stages"].items():

            if stage["status"] != "succeeded":
                continue

            for artifact in stage["artifacts"]:

                path = (
                    self.workspace / artifact["path"]
                ).resolve()

                if (
                    self.workspace not in path.parents
                    or not path.is_file()
                ):
                    problems.append(
                        f"{stage_name}: missing artifact "
                        f"{artifact['path']}"
                    )
                    continue

                digest = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()

                if digest != artifact["sha256"]:
                    problems.append(
                        f"{stage_name}: checksum mismatch "
                        f"{artifact['path']}"
                    )

        return problems


__all__ = [
    "DEPENDENCIES",
    "ORCHESTRATOR_SCHEMA_VERSION",
    "OrchestrationError",
    "ProductionJob",
    "STAGES",
]
