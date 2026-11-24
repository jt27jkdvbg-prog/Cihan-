
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"

PROFILES: dict[str, dict[str, int | str]] = {
    "landscape": {
        "width": 1920,
        "height": 1080,
        "fps": 24,
        "aspect_ratio": "16:9",
    },
    "landscape_30": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "aspect_ratio": "16:9",
    },
    "vertical": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "aspect_ratio": "9:16",
    },
    "square": {
        "width": 1080,
        "height": 1080,
        "fps": 30,
        "aspect_ratio": "1:1",
    },
}


class ProjectConfigError(ValueError):
    """Raised when a film project configuration is invalid."""


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    default: str = "",
) -> str:
    if value is None:
        value = default

    if not isinstance(value, str):
        raise ProjectConfigError(
            f"{field_name} must be a string."
        )

    value = value.strip()

    if required and not value:
        raise ProjectConfigError(
            f"{field_name} is required."
        )

    return value


def _number(
    value: Any,
    field_name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    default: float | None = None,
) -> float:
    if value is None:
        value = default

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ProjectConfigError(
            f"{field_name} must be a number."
        )

    result = float(value)

    if not math.isfinite(result):
        raise ProjectConfigError(
            f"{field_name} must be finite."
        )

    if result < minimum:
        raise ProjectConfigError(
            f"{field_name} must be at least {minimum}."
        )

    if maximum is not None and result > maximum:
        raise ProjectConfigError(
            f"{field_name} must be at most {maximum}."
        )

    return result


def _optional_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise ProjectConfigError(
            f"{field_name} must be an object."
        )

    return dict(value)


def _string_list(
    value: Any,
    field_name: str,
) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ProjectConfigError(
            f"{field_name} must be a list."
        )

    return [
        _text(item, f"{field_name}[]", required=True)
        for item in value
    ]


def _mapping_list(
    value: Any,
    field_name: str,
) -> list[dict[str, Any]]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ProjectConfigError(
            f"{field_name} must be a list."
        )

    result = []

    for index, item in enumerate(value, 1):
        if not isinstance(item, Mapping):
            raise ProjectConfigError(
                f"{field_name}[{index}] must be an object."
            )

        result.append(dict(item))

    return result


@dataclass
class Character:
    id: str
    name: str
    visual_description: str
    voice_description: str = ""
    reference_image: str | None = None

    age: int | None = None
    appearance: str = ""
    clothing: str = ""
    personality: str = ""
    motivation: str = ""
    voice_profile: dict[str, Any] = field(default_factory=dict)
    reference_images: list[str] = field(default_factory=list)
    continuity_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "Character":
        if not isinstance(data, Mapping):
            raise ProjectConfigError(
                "Each character must be an object."
            )

        reference_image = data.get("reference_image")

        if reference_image is not None:
            reference_image = _text(
                reference_image,
                "character.reference_image",
            ) or None

        age = data.get("age")

        if age is not None:
            if (
                isinstance(age, bool)
                or not isinstance(age, int)
                or not 0 <= age <= 120
            ):
                raise ProjectConfigError(
                    "character.age must be an integer between 0 and 120."
                )

        return cls(
            id=_text(
                data.get("id"),
                "character.id",
                required=True,
            ),
            name=_text(
                data.get("name"),
                "character.name",
                required=True,
            ),
            visual_description=_text(
                data.get("visual_description"),
                "character.visual_description",
                required=True,
            ),
            voice_description=_text(
                data.get("voice_description"),
                "character.voice_description",
            ),
            reference_image=reference_image,
            age=age,
            appearance=_text(
                data.get("appearance"),
                "character.appearance",
            ),
            clothing=_text(
                data.get("clothing"),
                "character.clothing",
            ),
            personality=_text(
                data.get("personality"),
                "character.personality",
            ),
            motivation=_text(
                data.get("motivation"),
                "character.motivation",
            ),
            voice_profile=_optional_mapping(
                data.get("voice_profile"),
                "character.voice_profile",
            ),
            reference_images=_string_list(
                data.get("reference_images"),
                "character.reference_images",
            ),
            continuity_prompt=_text(
                data.get("continuity_prompt"),
                "character.continuity_prompt",
            ),
            metadata=_optional_mapping(
                data.get("metadata"),
                "character.metadata",
            ),
        )


@dataclass
class Scene:
    id: str
    order: int
    duration_seconds: float
    visual_prompt: str

    dialogue: str = ""
    character_ids: list[str] = field(default_factory=list)
    location: str = ""
    camera: dict[str, Any] = field(default_factory=dict)
    sound_design: dict[str, Any] = field(default_factory=dict)
    transition_out: str = "cut"
    seed: int | None = None

    visual_subject: str = ""
    action: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    lighting: dict[str, Any] = field(default_factory=dict)
    motion: dict[str, Any] = field(default_factory=dict)
    composition: dict[str, Any] = field(default_factory=dict)
    shot_plan: list[dict[str, Any]] = field(default_factory=list)
    dialogue_turns: list[dict[str, Any]] = field(default_factory=list)
    props: list[str] = field(default_factory=list)
    reference_images: list[str] = field(default_factory=list)
    visual_style: str = ""
    continuity_notes: str = ""
    negative_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "Scene":
        if not isinstance(data, Mapping):
            raise ProjectConfigError(
                "Each scene must be an object."
            )

        raw_order = data.get("order")

        if (
            isinstance(raw_order, bool)
            or not isinstance(raw_order, int)
            or raw_order < 1
        ):
            raise ProjectConfigError(
                "scene.order must be a positive integer."
            )

        duration = _number(
            data.get("duration_seconds"),
            "scene.duration_seconds",
            minimum=1,
            maximum=300,
        )

        raw_seed = data.get("seed")

        if raw_seed is not None:
            if (
                isinstance(raw_seed, bool)
                or not isinstance(raw_seed, int)
                or raw_seed < 0
            ):
                raise ProjectConfigError(
                    "scene.seed must be a non-negative integer."
                )

        return cls(
            id=_text(
                data.get("id"),
                "scene.id",
                required=True,
            ),
            order=raw_order,
            duration_seconds=duration,
            visual_prompt=_text(
                data.get("visual_prompt"),
                "scene.visual_prompt",
                required=True,
            ),
            dialogue=_text(
                data.get("dialogue"),
                "scene.dialogue",
            ),
            character_ids=_string_list(
                data.get("character_ids"),
                "scene.character_ids",
            ),
            location=_text(
                data.get("location"),
                "scene.location",
            ),
            camera=_optional_mapping(
                data.get("camera"),
                "scene.camera",
            ),
            sound_design=_optional_mapping(
                data.get("sound_design"),
                "scene.sound_design",
            ),
            transition_out=_text(
                data.get("transition_out"),
                "scene.transition_out",
                default="cut",
            ),
            seed=raw_seed,
            visual_subject=_text(
                data.get("visual_subject"),
                "scene.visual_subject",
            ),
            action=_text(
                data.get("action"),
                "scene.action",
            ),
            environment=_optional_mapping(
                data.get("environment"),
                "scene.environment",
            ),
            lighting=_optional_mapping(
                data.get("lighting"),
                "scene.lighting",
            ),
            motion=_optional_mapping(
                data.get("motion"),
                "scene.motion",
            ),
            composition=_optional_mapping(
                data.get("composition"),
                "scene.composition",
            ),
            shot_plan=_mapping_list(
                data.get("shot_plan"),
                "scene.shot_plan",
            ),
            dialogue_turns=_mapping_list(
                data.get("dialogue_turns"),
                "scene.dialogue_turns",
            ),
            props=_string_list(
                data.get("props"),
                "scene.props",
            ),
            reference_images=_string_list(
                data.get("reference_images"),
                "scene.reference_images",
            ),
            visual_style=_text(
                data.get("visual_style"),
                "scene.visual_style",
            ),
            continuity_notes=_text(
                data.get("continuity_notes"),
                "scene.continuity_notes",
            ),
            negative_prompt=_text(
                data.get("negative_prompt"),
                "scene.negative_prompt",
            ),
            metadata=_optional_mapping(
                data.get("metadata"),
                "scene.metadata",
            ),
        )


@dataclass
class FilmProject:
    project_id: str
    title: str
    brief: str
    target_duration_seconds: float
    visual_style: str
    profile: str
    language: str
    characters: list[Character]
    scenes: list[Scene]

    music_brief: str = ""
    sound_mix_target_lufs: float = -14.0
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    director_notes: str = ""
    character_bible: str = ""
    global_visual_rules: dict[str, Any] = field(default_factory=dict)
    global_lighting: dict[str, Any] = field(default_factory=dict)
    global_camera_rules: dict[str, Any] = field(default_factory=dict)
    global_sound_design: dict[str, Any] = field(default_factory=dict)
    style_guide: dict[str, Any] = field(default_factory=dict)
    production_settings: dict[str, Any] = field(default_factory=dict)
    negative_prompt: str = ""

    @property
    def render_profile(self) -> dict[str, int | str]:
        return dict(PROFILES[self.profile])

    @property
    def total_scene_duration(self) -> float:
        return sum(
            scene.duration_seconds
            for scene in self.scenes
        )

    @property
    def duration_difference(self) -> float:
        return (
            self.total_scene_duration
            - self.target_duration_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "FilmProject":
        if not isinstance(data, Mapping):
            raise ProjectConfigError(
                "The project configuration must be an object."
            )

        raw_characters = data.get("characters", [])
        raw_scenes = data.get("scenes", [])

        if not isinstance(raw_characters, list):
            raise ProjectConfigError(
                "characters must be a list."
            )

        if not isinstance(raw_scenes, list):
            raise ProjectConfigError(
                "scenes must be a list."
            )

        characters = [
            Character.from_dict(item)
            for item in raw_characters
        ]

        scenes = [
            Scene.from_dict(item)
            for item in raw_scenes
        ]

        project = cls(
            project_id=_text(
                data.get("project_id"),
                "project_id",
                required=True,
            ),
            title=_text(
                data.get("title"),
                "title",
                required=True,
            ),
            brief=_text(
                data.get("brief"),
                "brief",
                required=True,
            ),
            target_duration_seconds=_number(
                data.get("target_duration_seconds"),
                "target_duration_seconds",
                minimum=1,
                maximum=7200,
            ),
            visual_style=_text(
                data.get("visual_style"),
                "visual_style",
                required=True,
            ),
            profile=_text(
                data.get("profile"),
                "profile",
                default="landscape",
            ),
            language=_text(
                data.get("language"),
                "language",
                default="tr",
            ),
            characters=characters,
            scenes=scenes,
            music_brief=_text(
                data.get("music_brief"),
                "music_brief",
            ),
            sound_mix_target_lufs=_number(
                data.get("sound_mix_target_lufs"),
                "sound_mix_target_lufs",
                minimum=-30,
                maximum=-5,
                default=-14,
            ),
            schema_version=_text(
                data.get("schema_version"),
                "schema_version",
                default=SCHEMA_VERSION,
            ),
            metadata=_optional_mapping(
                data.get("metadata"),
                "metadata",
            ),
            director_notes=_text(
                data.get("director_notes"),
                "director_notes",
            ),
            character_bible=_text(
                data.get("character_bible"),
                "character_bible",
            ),
            global_visual_rules=_optional_mapping(
                data.get("global_visual_rules"),
                "global_visual_rules",
            ),
            global_lighting=_optional_mapping(
                data.get("global_lighting"),
                "global_lighting",
            ),
            global_camera_rules=_optional_mapping(
                data.get("global_camera_rules"),
                "global_camera_rules",
            ),
            global_sound_design=_optional_mapping(
                data.get("global_sound_design"),
                "global_sound_design",
            ),
            style_guide=_optional_mapping(
                data.get("style_guide"),
                "style_guide",
            ),
            production_settings=_optional_mapping(
                data.get("production_settings"),
                "production_settings",
            ),
            negative_prompt=_text(
                data.get("negative_prompt"),
                "negative_prompt",
            ),
        )

        validate_project(project)
        return project


def validate_project(project: FilmProject) -> None:
    if not isinstance(project, FilmProject):
        raise ProjectConfigError(
            "project must be a FilmProject instance."
        )

    if project.schema_version != SCHEMA_VERSION:
        raise ProjectConfigError(
            f"Unsupported schema version: {project.schema_version}"
        )

    if project.profile not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ProjectConfigError(
            f"Unknown profile '{project.profile}'. "
            f"Available profiles: {available}"
        )

    if not 0 <= len(project.characters) <= 12:
        raise ProjectConfigError(
            "A project can contain between 0 and 12 characters."
        )

    if not 1 <= len(project.scenes) <= 60:
        raise ProjectConfigError(
            "A project must contain between 1 and 60 scenes."
        )

    character_ids = [
        character.id
        for character in project.characters
    ]

    if len(character_ids) != len(set(character_ids)):
        raise ProjectConfigError(
            "Character IDs must be unique."
        )

    scene_ids = [
        scene.id
        for scene in project.scenes
    ]

    if len(scene_ids) != len(set(scene_ids)):
        raise ProjectConfigError(
            "Scene IDs must be unique."
        )

    expected_orders = list(
        range(1, len(project.scenes) + 1)
    )

    actual_orders = sorted(
        scene.order
        for scene in project.scenes
    )

    if actual_orders != expected_orders:
        raise ProjectConfigError(
            "Scene order values must be unique and consecutive, starting at 1."
        )

    known_character_ids = set(character_ids)

    for character in project.characters:
        if not isinstance(character, Character):
            raise ProjectConfigError(
                "Every character must be a Character instance."
            )

        if character.age is not None:
            if (
                isinstance(character.age, bool)
                or not isinstance(character.age, int)
                or not 0 <= character.age <= 120
            ):
                raise ProjectConfigError(
                    f"Character '{character.id}' has an invalid age."
                )

    for scene in project.scenes:
        if not isinstance(scene, Scene):
            raise ProjectConfigError(
                "Every scene must be a Scene instance."
            )

        unknown_ids = (
            set(scene.character_ids)
            - known_character_ids
        )

        if unknown_ids:
            raise ProjectConfigError(
                f"Scene '{scene.id}' references unknown characters: "
                f"{', '.join(sorted(unknown_ids))}"
            )

        for turn_index, turn in enumerate(
            scene.dialogue_turns,
            1,
        ):
            speaker = turn.get("speaker")

            if speaker is not None and speaker not in known_character_ids:
                raise ProjectConfigError(
                    f"Scene '{scene.id}', dialogue {turn_index} "
                    f"references unknown speaker '{speaker}'."
                )

            listeners = turn.get("listener_ids", [])

            if not isinstance(listeners, list):
                raise ProjectConfigError(
                    f"Scene '{scene.id}', dialogue {turn_index}: "
                    "listener_ids must be a list."
                )

            unknown_listeners = (
                set(listeners) - known_character_ids
            )

            if unknown_listeners:
                raise ProjectConfigError(
                    f"Scene '{scene.id}', dialogue {turn_index} "
                    "references unknown listeners."
                )

    if project.total_scene_duration <= 0:
        raise ProjectConfigError(
            "Total scene duration must be greater than zero."
        )


def load_project(
    source: str | Path | Mapping[str, Any],
) -> FilmProject:
    """Load a project from a JSON file path or mapping."""

    if isinstance(source, Mapping):
        data = source

    elif isinstance(source, (str, Path)):
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(
                f"Project file not found: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

        except json.JSONDecodeError as exc:
            raise ProjectConfigError(
                f"Invalid JSON in project file: {exc}"
            ) from exc

    else:
        raise ProjectConfigError(
            "Project source must be a file path or an object."
        )

    return FilmProject.from_dict(data)


def save_project(
    project: FilmProject,
    destination: str | Path,
) -> Path:
    """Validate and save a project as UTF-8 JSON."""

    validate_project(project)

    path = Path(destination)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            project.to_dict(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


def stable_scene_seed(
    project_id: str,
    scene_id: str,
    base_seed: int = 0,
) -> int:
    """Generate a deterministic seed for a scene."""

    if (
        not isinstance(project_id, str)
        or not project_id.strip()
    ):
        raise ProjectConfigError(
            "project_id is required to generate a scene seed."
        )

    if (
        not isinstance(scene_id, str)
        or not scene_id.strip()
    ):
        raise ProjectConfigError(
            "scene_id is required to generate a scene seed."
        )

    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or base_seed < 0
    ):
        raise ProjectConfigError(
            "base_seed must be a non-negative integer."
        )

    raw = (
        f"{project_id.strip()}:"
        f"{scene_id.strip()}:"
        f"{base_seed}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


__all__ = [
    "SCHEMA_VERSION",
    "PROFILES",
    "ProjectConfigError",
    "Character",
    "Scene",
    "FilmProject",
    "validate_project",
    "load_project",
    "save_project",
    "stable_scene_seed",
]
