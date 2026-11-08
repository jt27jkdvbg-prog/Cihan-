
"""Quality checks for generated motion videos."""
from pathlib import Path


def require_real_motion(mode, fal_key_present):
    mode = (mode or "real").strip().lower()

    if mode not in {"real", "preview"}:
        raise ValueError("MOTION_MODE must be 'real' or 'preview'.")

    if mode == "real" and not fal_key_present:
        raise RuntimeError(
            "Real motion requires a configured video-generation provider. "
            "A still image with zoom or pan is not real character animation. "
            "Configure FAL_KEY or explicitly select preview mode."
        )

    return mode == "real"


def validate_motion_outputs(paths, mode="real"):
    mode = (mode or "real").strip().lower()

    existing = [
        str(path)
        for path in paths
        if Path(path).is_file() and Path(path).stat().st_size > 0
    ]

    if mode == "real" and len(existing) != len(paths):
        raise RuntimeError(
            f"Expected {len(paths)} valid motion clips, "
            f"but found {len(existing)}."
        )

    return existing
