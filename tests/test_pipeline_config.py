
import json
import tempfile
import unittest
from pathlib import Path

from pipeline_config import (
    SCHEMA_VERSION,
    PROFILES,
    Character,
    FilmProject,
    ProjectConfigError,
    Scene,
    load_project,
    save_project,
    stable_scene_seed,
    validate_project,
)


def make_valid_project():
    return {
        "project_id": "test-film-001",
        "title": "Test Film",
        "brief": "A cinematic test project.",
        "target_duration_seconds": 30,
        "visual_style": "cinematic realistic",
        "profile": "landscape",
        "language": "tr",
        "characters": [
            {
                "id": "character_1",
                "name": "Ali",
                "visual_description": "Dark hair, navy coat.",
                "voice_description": "Calm male voice.",
            }
        ],
        "scenes": [
            {
                "id": "scene_1",
                "order": 1,
                "duration_seconds": 15,
                "visual_prompt": "A cinematic city street at night.",
                "dialogue": "Burada ne oluyor?",
                "character_ids": ["character_1"],
                "location": "Istanbul",
                "camera": {
                    "movement": "slow tracking",
                },
                "sound_design": {
                    "ambience": "city night",
                },
                "transition_out": "cut",
            },
            {
                "id": "scene_2",
                "order": 2,
                "duration_seconds": 15,
                "visual_prompt": "A wide cinematic view of the city.",
                "character_ids": [],
            },
        ],
        "music_brief": "Atmospheric cinematic music.",
        "sound_mix_target_lufs": -14,
    }


class PipelineConfigTests(unittest.TestCase):

    def setUp(self):
        self.data = make_valid_project()

    def test_valid_project_loads(self):
        project = load_project(self.data)

        self.assertIsInstance(project, FilmProject)
        self.assertEqual(project.title, "Test Film")
        self.assertEqual(len(project.scenes), 2)

    def test_schema_version_is_current(self):
        project = load_project(self.data)

        self.assertEqual(project.schema_version, SCHEMA_VERSION)

    def test_all_render_profiles_exist(self):
        expected = {
            "landscape",
            "landscape_30",
            "vertical",
            "square",
        }

        self.assertEqual(set(PROFILES.keys()), expected)

    def test_render_profile_dimensions(self):
        project = load_project(self.data)

        self.assertEqual(
            project.render_profile["width"],
            1920,
        )
        self.assertEqual(
            project.render_profile["height"],
            1080,
        )
        self.assertEqual(
            project.render_profile["fps"],
            24,
        )

    def test_vertical_profile_is_supported(self):
        self.data["profile"] = "vertical"
        project = load_project(self.data)

        self.assertEqual(
            project.render_profile["aspect_ratio"],
            "9:16",
        )

    def test_square_profile_is_supported(self):
        self.data["profile"] = "square"
        project = load_project(self.data)

        self.assertEqual(
            project.render_profile["aspect_ratio"],
            "1:1",
        )

    def test_scene_duration_is_calculated(self):
        project = load_project(self.data)

        self.assertEqual(project.total_scene_duration, 30)

    def test_dialogue_is_optional(self):
        self.data["scenes"][0]["dialogue"] = ""
        project = load_project(self.data)

        self.assertEqual(project.scenes[0].dialogue, "")

    def test_characters_are_optional(self):
        self.data["characters"] = []

        for scene in self.data["scenes"]:
            scene["character_ids"] = []

        project = load_project(self.data)

        self.assertEqual(project.characters, [])

    def test_duplicate_character_ids_are_rejected(self):
        self.data["characters"].append({
            "id": "character_1",
            "name": "Duplicate",
            "visual_description": "Another character.",
        })

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_duplicate_scene_ids_are_rejected(self):
        self.data["scenes"][1]["id"] = "scene_1"

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_invalid_character_reference_is_rejected(self):
        self.data["scenes"][0]["character_ids"] = ["unknown"]

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_invalid_scene_order_is_rejected(self):
        self.data["scenes"][1]["order"] = 1

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_unknown_profile_is_rejected(self):
        self.data["profile"] = "unknown_profile"

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_empty_project_title_is_rejected(self):
        self.data["title"] = ""

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_missing_visual_prompt_is_rejected(self):
        self.data["scenes"][0]["visual_prompt"] = ""

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_scene_duration_below_minimum_is_rejected(self):
        self.data["scenes"][0]["duration_seconds"] = 0

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_scene_duration_above_maximum_is_rejected(self):
        self.data["scenes"][0]["duration_seconds"] = 301

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_too_many_characters_are_rejected(self):
        self.data["characters"] = [
            {
                "id": f"character_{index}",
                "name": f"Character {index}",
                "visual_description": "Test character.",
            }
            for index in range(13)
        ]

        for scene in self.data["scenes"]:
            scene["character_ids"] = []

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_too_many_scenes_are_rejected(self):
        self.data["scenes"] = [
            {
                "id": f"scene_{index}",
                "order": index,
                "duration_seconds": 1,
                "visual_prompt": "Test scene.",
                "character_ids": [],
            }
            for index in range(1, 62)
        ]

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_boolean_is_not_accepted_as_duration(self):
        self.data["scenes"][0]["duration_seconds"] = True

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_invalid_camera_configuration_is_rejected(self):
        self.data["scenes"][0]["camera"] = "not-an-object"

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_invalid_seed_is_rejected(self):
        self.data["scenes"][0]["seed"] = -1

        with self.assertRaises(ProjectConfigError):
            load_project(self.data)

    def test_stable_scene_seed_is_deterministic(self):
        first = stable_scene_seed(
            "project-1",
            "scene-1",
            42,
        )
        second = stable_scene_seed(
            "project-1",
            "scene-1",
            42,
        )

        self.assertEqual(first, second)

    def test_different_scenes_get_different_seeds(self):
        first = stable_scene_seed(
            "project-1",
            "scene-1",
            42,
        )
        second = stable_scene_seed(
            "project-1",
            "scene-2",
            42,
        )

        self.assertNotEqual(first, second)

    def test_negative_base_seed_is_rejected(self):
        with self.assertRaises(ProjectConfigError):
            stable_scene_seed(
                "project-1",
                "scene-1",
                -1,
            )

    def test_project_can_be_saved_and_loaded(self):
        project = load_project(self.data)

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "project.json"

            saved_path = save_project(
                project,
                destination,
            )

            loaded_project = load_project(saved_path)

            self.assertTrue(saved_path.exists())
            self.assertEqual(
                loaded_project.title,
                project.title,
            )
            self.assertEqual(
                len(loaded_project.scenes),
                len(project.scenes),
            )

    def test_saved_json_preserves_unicode(self):
        self.data["title"] = "İstanbul'da Son Kaçış"
        project = load_project(self.data)

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "project.json"
            save_project(project, destination)

            content = destination.read_text(encoding="utf-8")

            self.assertIn("İstanbul'da Son Kaçış", content)

    def test_invalid_json_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text("{invalid json", encoding="utf-8")

            with self.assertRaises(ProjectConfigError):
                load_project(path)

    def test_nonexistent_file_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.json"

            with self.assertRaises(FileNotFoundError):
                load_project(path)

    def test_non_object_project_is_rejected(self):
        with self.assertRaises(ProjectConfigError):
            load_project([])

    def test_to_dict_preserves_project_data(self):
        project = load_project(self.data)
        exported = project.to_dict()

        self.assertEqual(
            exported["project_id"],
            "test-film-001",
        )
        self.assertEqual(
            exported["scenes"][0]["visual_prompt"],
            self.data["scenes"][0]["visual_prompt"],
        )


if __name__ == "__main__":
    unittest.main()
