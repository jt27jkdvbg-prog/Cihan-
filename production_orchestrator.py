
"""Durable job state and checkpoint orchestration for film production."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from pipeline_config import FilmProject, ProjectConfigError


ORCHESTRATOR_SCHEMA_VERSION = 1
HASH_CHUNK_SIZE = 1024 * 1024

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
    "post_production": ("video_generation", "audio_generation"),
    "quality_control": ("post_production",),
    "publishing": ("quality_control",),
}

VALID_STAGE_STATUSES = {
    "pending",
    "running",
    "succeeded",
    "failed",
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
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> tuple[str, int]:
    """Hash a file incrementally without loading it entirely into RAM."""
    digest = hashlib.sha256()
    total_size = 0

    with path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total_size += len(chunk)

    return digest.hexdigest(), total_size


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON to a temporary file and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(
                value,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)

        # Best-effort directory sync: protects the rename on POSIX systems.
        if os.name == "posix":
            try:
                directory_fd = os.open(
                    str(path.parent),
                    os.O_RDONLY,
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass

    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _job_lock(path: Path) -> Iterator[None]:
    """
    Acquire an operating-system lock.

    The lock file may remain on disk after use. The actual lock is held by
    the operating system and is automatically released if the process exits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.open("a+b")

    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()

        lock_file.seek(0)

        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(
                    lock_file.fileno(),
                    msvcrt.LK_NBLCK,
                    1,
                )
            elif os.name == "posix":
                import fcntl

                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            else:
                raise OrchestrationError(
                    "This operating system does not support job locking."
                )

        except (OSError, BlockingIOError) as exc:
            raise OrchestrationError(
                "This job is already being modified by another process."
            ) from exc

        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(
                    lock_file.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_UN,
                )

    finally:
        lock_file.close()


class ProductionJob:
    """Validated project and durable production checkpoints."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.manifest_path = self.workspace / "job_manifest.json"
        self.project_path = self.workspace / "project.json"
        self.lock_path = self.workspace / ".job.lock"

    @classmethod
    def create(
        cls,
        workspace: str | Path,
        project_data: Mapping[str, Any],
    ) -> "ProductionJob":
        project = FilmProject.from_dict(project_data)
        job = cls(workspace)
        job.workspace.mkdir(parents=True, exist_ok=True)

        project_dict = project.to_dict()

        with _job_lock(job.lock_path):
            if job.manifest_path.exists() or job.project_path.exists():
                raise OrchestrationError(
                    f"A job or incomplete job already exists at {job.workspace}."
                )

            now = time.time()
            manifest = {
                "orchestrator_schema_version": ORCHESTRATOR_SCHEMA_VERSION,
                "job_id": project.project_id,
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

            _atomic_json(job.project_path, project_dict)
            _atomic_json(job.manifest_path, manifest)

        return job

    @classmethod
    def open(cls, workspace: str | Path) -> "ProductionJob":
        job = cls(workspace)
        manifest = job._read_manifest()
        project = job._read_project()

        if _json_hash(project) != manifest["project_sha256"]:
            raise OrchestrationError(
                "Project JSON changed after job creation. Unsafe resume refused."
            )

        problems = job.verify_artifacts()
        if problems:
            raise OrchestrationError(
                "Stored production artifacts failed verification: "
                + "; ".join(problems[:10])
            )

        return job

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise OrchestrationError("Job manifest was not found.")

        try:
            data = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise OrchestrationError(
                f"Job manifest is unreadable: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise OrchestrationError("Job manifest must be a JSON object.")

        if (
            data.get("orchestrator_schema_version")
            != ORCHESTRATOR_SCHEMA_VERSION
        ):
            raise OrchestrationError("Unsupported job manifest version.")

        stages = data.get("stages")
        if not isinstance(stages, dict) or set(stages) != set(STAGES):
            raise OrchestrationError("Job manifest stage set is invalid.")

        for name in STAGES:
            stage = stages[name]
            if not isinstance(stage, dict):
                raise OrchestrationError(
                    f"Manifest data for stage '{name}' is invalid."
                )
            if stage.get("status") not in VALID_STAGE_STATUSES:
                raise OrchestrationError(
                    f"Invalid status for stage '{name}'."
                )
            attempts = stage.get("attempts")
            if (
                isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 0
            ):
                raise OrchestrationError(
                    f"Invalid attempt count for stage '{name}'."
                )
            if not isinstance(stage.get("artifacts"), list):
                raise OrchestrationError(
                    f"Invalid artifact list for stage '{name}'."
                )

        return data

    def _read_project(self) -> dict[str, Any]:
        try:
            data = json.loads(
                self.project_path.read_text(encoding="utf-8")
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
        return FilmProject.from_dict(self._read_project())

    def _refresh_job_status(self, data: dict[str, Any]) -> None:
        statuses = [
            data["stages"][name]["status"]
            for name in STAGES
        ]

        if all(status == "succeeded" for status in statuses):
            data["status"] = "completed"
        elif "failed" in statuses:
            data["status"] = "failed"
        elif "running" in statuses:
            data["status"] = "running"
        else:
            data["status"] = "queued"

        data["updated_at"] = time.time()

    def start_stage(self, stage: str) -> int:
        if stage not in DEPENDENCIES:
            raise OrchestrationError(
                f"Unknown production stage: {stage}"
            )

        with _job_lock(self.lock_path):
            data = self._read_manifest()
            current = data["stages"][stage]

            if current["status"] != "pending":
                raise OrchestrationError(
                    f"Stage '{stage}' must be pending before it can start "
                    f"(current status: {current['status']})."
                )

            incomplete = [
                dependency
                for dependency in DEPENDENCIES[stage]
                if data["stages"][dependency]["status"] != "succeeded"
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

            self._refresh_job_status(data)
            _atomic_json(self.manifest_path, data)
            return current["attempts"]

    def _resolve_artifact(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative.strip():
            raise OrchestrationError("Artifact path must be a non-empty string.")

        raw_path = Path(relative)

        if raw_path.is_absolute() or ".." in raw_path.parts:
            raise OrchestrationError(
                "Artifact path must be relative to the job workspace."
            )

        candidate = (self.workspace / raw_path).resolve()

        if self.workspace not in candidate.parents:
            raise OrchestrationError(
                "Artifact path must stay inside the job workspace."
            )

        return candidate

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
        seen_paths = set()

        for relative in artifacts or []:
            candidate = self._resolve_artifact(relative)
            normalized = str(candidate)

            if normalized in seen_paths:
                raise OrchestrationError(
                    f"Duplicate artifact path: {relative}"
                )
            seen_paths.add(normalized)

            if not candidate.is_file():
                raise OrchestrationError(
                    f"Required artifact is missing: {relative}"
                )

            digest, size = _file_sha256(candidate)

            if size <= 0:
                raise OrchestrationError(
                    f"Required artifact is empty: {relative}"
                )

            verified.append({
                "path": candidate.relative_to(self.workspace).as_posix(),
                "size_bytes": size,
                "sha256": digest,
            })

        with _job_lock(self.lock_path):
            data = self._read_manifest()
            current = data["stages"][stage]

            if current["status"] != "running":
                raise OrchestrationError(
                    f"Stage '{stage}' must be running before completion."
                )

            current.update(
                status="succeeded",
                finished_at=time.time(),
                artifacts=verified,
                error=None,
            )

            self._refresh_job_status(data)
            _atomic_json(self.manifest_path, data)

    def fail_stage(
        self,
        stage: str,
        error: BaseException | str,
    ) -> None:
        if stage not in DEPENDENCIES:
            raise OrchestrationError(
                f"Unknown production stage: {stage}"
            )

        message = str(error).strip() or type(error).__name__

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

            self._refresh_job_status(data)
            _atomic_json(self.manifest_path, data)

    def retry_failed_stage(self, stage: str) -> None:
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

            self._refresh_job_status(data)
            _atomic_json(self.manifest_path, data)

    def verify_artifacts(self) -> list[str]:
        problems = []
        data = self._read_manifest()

        for stage_name, stage in data["stages"].items():
            if stage["status"] != "succeeded":
                continue

            for artifact in stage["artifacts"]:
                try:
                    relative = artifact["path"]
                    expected_hash = artifact["sha256"]
                    expected_size = artifact["size_bytes"]

                    path = self._resolve_artifact(relative)

                    if not path.is_file():
                        problems.append(
                            f"{stage_name}: missing artifact {relative}"
                        )
                        continue

                    actual_hash, actual_size = _file_sha256(path)

                    if actual_size != expected_size:
                        problems.append(
                            f"{stage_name}: size mismatch {relative}"
                        )
                    elif actual_hash != expected_hash:
                        problems.append(
                            f"{stage_name}: checksum mismatch {relative}"
                        )

                except (KeyError, TypeError, OrchestrationError) as exc:
                    problems.append(
                        f"{stage_name}: invalid artifact record ({exc})"
                    )

        return problems


__all__ = [
    "DEPENDENCIES",
    "HASH_CHUNK_SIZE",
    "ORCHESTRATOR_SCHEMA_VERSION",
    "OrchestrationError",
    "ProductionJob",
    "STAGES",
]

