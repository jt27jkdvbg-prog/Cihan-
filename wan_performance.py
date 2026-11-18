
"""Performance profiling utilities for the Wan film pipeline."""

from __future__ import annotations

import json
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class PerformanceRecorder:
    """Collect stage timings, scene metrics and optional GPU telemetry."""

    def __init__(
        self,
        output_dir: str | Path,
        project_title: str = "",
        profile: str = "unknown",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = time.perf_counter()
        self.started_iso = datetime.now(timezone.utc).isoformat()

        self.project_title = project_title
        self.profile = profile
        self.stages: list[dict[str, Any]] = []
        self.scenes: list[dict[str, Any]] = []
        self.gpu_samples: list[dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure one named operation without hiding its exceptions."""
        started = time.perf_counter()
        status = "success"
        error_type = None

        try:
            yield
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            self.stages.append({
                "name": str(name),
                "elapsed_seconds": round(
                    time.perf_counter() - started, 3
                ),
                "status": status,
                "error_type": error_type,
            })

    def record_scene(
        self,
        scene_index: int,
        elapsed_seconds: float,
        frame_count: int,
        width: int,
        height: int,
        sampling_steps: int,
        seed: int | None = None,
    ) -> None:
        """Record measurable output details for a generated scene."""
        values = {
            "scene_index": scene_index,
            "elapsed_seconds": elapsed_seconds,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "sampling_steps": sampling_steps,
        }

        if any(value < 0 for value in values.values()):
            raise ValueError("Scene metrics cannot contain negative values.")

        self.scenes.append({
            **values,
            "seed": seed,
        })

    def capture_gpu(self, torch_module: Any = None) -> dict[str, Any]:
        """Capture CUDA device and memory information when available."""
        try:
            if torch_module is None:
                import torch as torch_module

            if not torch_module.cuda.is_available():
                sample = {"available": False}
                self.gpu_samples.append(sample)
                return sample

            device_index = torch_module.cuda.current_device()
            properties = torch_module.cuda.get_device_properties(
                device_index
            )

            free_bytes, total_bytes = torch_module.cuda.mem_get_info(
                device_index
            )

            sample = {
                "available": True,
                "device_index": device_index,
                "device_name": properties.name,
                "total_memory_bytes": int(total_bytes),
                "free_memory_bytes": int(free_bytes),
                "allocated_memory_bytes": int(
                    torch_module.cuda.memory_allocated(device_index)
                ),
                "reserved_memory_bytes": int(
                    torch_module.cuda.memory_reserved(device_index)
                ),
                "peak_allocated_memory_bytes": int(
                    torch_module.cuda.max_memory_allocated(device_index)
                ),
            }

        except Exception as exc:
            sample = {
                "available": False,
                "error": type(exc).__name__,
            }

        self.gpu_samples.append(sample)
        return sample

    def build_report(
        self,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a JSON-serializable performance report."""
        elapsed = time.perf_counter() - self.started_at

        successful_stages = [
            item for item in self.stages
            if item["status"] == "success"
        ]

        report = {
            "report_version": 1,
            "project_title": self.project_title,
            "profile": self.profile,
            "started_at_utc": self.started_iso,
            "finished_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "total_elapsed_seconds": round(elapsed, 3),
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "stage_count": len(self.stages),
            "successful_stage_count": len(successful_stages),
            "failed_stage_count": len(self.stages)
            - len(successful_stages),
            "scene_count": len(self.scenes),
            "average_scene_seconds": (
                round(
                    sum(item["elapsed_seconds"] for item in self.scenes)
                    / len(self.scenes),
                    3,
                )
                if self.scenes else None
            ),
            "stages": self.stages,
            "scenes": self.scenes,
            "gpu_samples": self.gpu_samples,
            "extra": extra or {},
        }

        return report

    def write_report(
        self,
        filename: str = "performance_report.json",
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write the report atomically and return its path."""
        destination = self.output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")

        report = self.build_report(extra=extra)

        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)

        return destination


__all__ = ["PerformanceRecorder"]
