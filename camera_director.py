"""Scene-aware camera movement presets for the FFmpeg fallback.

These moves animate the camera framing of a still image.
They do not create real 3D movement or character animation.
"""

PAN_PATHS = (
    ("center", "0.5", "0.5"),
    ("left_to_right", "on/{d}", "0.5"),
    ("right_to_left", "1-on/{d}", "0.5"),
    ("top_to_bottom", "0.5", "on/{d}"),
    ("bottom_to_top", "0.5", "1-on/{d}"),
    ("diagonal_tl_br", "on/{d}", "on/{d}"),
    ("diagonal_tr_bl", "1-on/{d}", "on/{d}"),
    ("diagonal_bl_tr", "on/{d}", "1-on/{d}"),
    ("diagonal_br_tl", "1-on/{d}", "1-on/{d}"),
)

ZOOM_CURVES = (
    ("push_in", "1+0.18*on/{d}"),
    ("pull_out", "1.18-0.18*on/{d}"),
    ("steady_tight", "1.16"),
    ("gentle_push", "1.04+0.08*on/{d}"),
)

CATEGORY_PAN_ORDER = {
    "closeup": (0, 4, 1, 2, 3, 5, 6, 7, 8),
    "wide": (5, 6, 7, 8, 1, 2, 3, 4, 0),
    "tracking": (1, 2, 5, 6, 7, 8, 3, 4, 0),
    "dramatic": (6, 7, 8, 5, 1, 2, 3, 4, 0),
    "general": (0, 1, 2, 3, 4, 5, 6, 7, 8),
}

HINTS = {
    "closeup": (
        "close-up", "close up", "extreme close", "portrait",
        "face", "eyes", "detail", "macro",
        "yakın plan", "yüz", "göz", "detay",
    ),
    "wide": (
        "wide shot", "establishing", "aerial", "drone",
        "landscape", "panorama", "location reveal",
        "genel plan", "kuş bakışı", "manzara",
    ),
    "tracking": (
        "tracking", "follow", "chase", "running", "walking",
        "moving", "takip", "kovala", "koş", "yürü", "ilerle",
    ),
    "dramatic": (
        "dramatic", "reveal", "impact", "battle", "storm",
        "fall", "gerilim", "savaş", "fırtına", "çarpıcı",
    ),
}


def camera_category(scene):
    """Choose a camera movement family from the scene description."""
    if not isinstance(scene, dict):
        return "general"

    description = " ".join(
        str(scene.get(key, ""))
        for key in ("camera", "action", "visual_subject", "narration")
    ).lower()

    for category in ("closeup", "wide", "tracking", "dramatic"):
        if any(hint in description for hint in HINTS[category]):
            return category

    return "general"


def build_camera_plan(scene, scene_index=0):
    """Return a deterministic camera pan and zoom combination."""
    category = camera_category(scene)
    order = CATEGORY_PAN_ORDER[category]
    index = int(scene_index) % len(order)

    pan_index = order[index]

    zoom_offsets = {
        "closeup": 0,
        "wide": 1,
        "tracking": 2,
        "dramatic": 3,
        "general": 0,
    }

    zoom_index = (
        int(scene_index) + zoom_offsets[category]
    ) % len(ZOOM_CURVES)

    pan_name, x_expr, y_expr = PAN_PATHS[pan_index]
    zoom_name, zoom_expr = ZOOM_CURVES[zoom_index]

    return {
        "category": category,
        "name": f"{category}_{pan_name}_{zoom_name}",
        "pan": pan_name,
        "zoom": zoom_name,
        "x": x_expr,
        "y": y_expr,
        "z": zoom_expr,
    }


def build_zoompan_filter(scene, scene_index, frames, width, height, fps):
    """Build an FFmpeg zoompan filter for a single generated image."""
    frames = max(1, int(frames))
    plan = build_camera_plan(scene, scene_index)

    x_expr = plan["x"].format(d=frames)
    y_expr = plan["y"].format(d=frames)
    z_expr = plan["z"].format(d=frames)

    return (
        f"zoompan=z='{z_expr}':"
        f"x='(iw-iw/zoom)*({x_expr})':"
        f"y='(ih-ih/zoom)*({y_expr})':"
        f"d={frames}:"
        f"s={int(width)}x{int(height)}:"
        f"fps={int(fps)}"
    )


def all_camera_preset_names():
    """List all supported camera combinations."""
    names = []

    for category, order in CATEGORY_PAN_ORDER.items():
        for pan_index in order:
            pan_name = PAN_PATHS[pan_index][0]

            for zoom_name, _ in ZOOM_CURVES:
                names.append(
                    f"{category}_{pan_name}_{zoom_name}"
                )

    return tuple(names)
