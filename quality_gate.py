
"""Provider-neutral quality checks for generated motion videos."""

from pathlib import Path


VALID_MOTION_MODES = {"real", "preview"}


def require_real_motion(mode, provider_available):
    """
    Validate the requested motion mode.

    This function does not depend on a specific provider such as FAL.
    Any configured video-generation provider can be used.
    """

    mode = (mode or "real").strip().lower()

    if mode not in VALID_MOTION_MODES:
        raise ValueError(
            "MOTION_MODE must be 'real' or 'preview'."
        )

    if mode == "real" and not provider_available:
        raise RuntimeError(
            "Real motion requires an available video-generation "
            "provider. Configure a supported provider before "
            "starting production. A still image with zoom or pan "
            "is not real character animation."
        )

    return mode == "real"


def validate_motion_outputs(paths, mode="real"):
    """
    Validate generated motion clips.

    In real mode, every expected clip must exist and be non-empty.
    In preview mode, only valid existing clips are returned.
    """

    mode = (mode or "real").strip().lower()

    if mode not in VALID_MOTION_MODES:
        raise ValueError(
            "MOTION_MODE must be 'real' or 'preview'."
        )

    paths = list(paths)

    existing = []

    for path in paths:
        file_path = Path(path)

        if (
            file_path.is_file()
            and file_path.stat().st_size > 0
        ):
            existing.append(str(file_path))

    if mode == "real" and len(existing) != len(paths):
        raise RuntimeError(
            f"Expected {len(paths)} valid motion clips, "
            f"but found {len(existing)}."
        )

    return existing

