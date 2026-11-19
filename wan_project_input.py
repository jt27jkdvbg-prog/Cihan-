
"""Normalize modern and legacy project JSON for the Wan workflow."""

from __future__ import annotations

from typing import Any, Mapping

from film_project import validate_film_project
from pipeline_config import FilmProject, ProjectConfigError
from project_adapter import pipeline_project_to_legacy
from copy import deepcopy

def normalize_wan_project(data: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either project schema and return validated legacy Wan data."""

    if not isinstance(data, Mapping):
        raise ProjectConfigError("Project JSON must contain an object.")

    candidate = dict(data)

    # Preserve the existing legacy input path unchanged.
    if "film" in candidate and "production" in candidate:
        legacy_project = candidate

    # Convert the modern pipeline schema to the exact legacy Wan format.
    else:
        modern_project = FilmProject.from_dict(candidate)
        legacy_project = pipeline_project_to_legacy(modern_project)

    report = validate_film_project(legacy_project)

    if not report["valid"]:
        raise ProjectConfigError(
            "Wan project validation failed: "
            + "; ".join(report["errors"])
        )
    # Preserve audio and cost settings from modern projects.
    if isinstance(candidate.get("audio"), Mapping):
        legacy_project["audio"] = deepcopy(candidate["audio"])
    elif isinstance(candidate.get("audio_settings"), Mapping):
        legacy_project["audio"] = deepcopy(candidate["audio_settings"])

    if isinstance(candidate.get("cost_policy"), Mapping):
        legacy_project["cost_policy"] = deepcopy(candidate["cost_policy"])
    return legacy_project


__all__ = ["normalize_wan_project"]
