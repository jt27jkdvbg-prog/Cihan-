
import copy
import unittest

from film_project import (
    PRODUCTION_DEFAULTS,
    build_film_project,
    to_legacy_script,
    validate_film_project,
)


def make_valid_script():
    characters = [
        {
            "id": "ali",
            "name": "Ali",
            "description": "Dark-haired detective, navy coat",
            "personality": "Calm and observant",
            "voice": "voice_ali",
            "voice_gender": "male",
            "voice_profile": {
                "edge": "tr-TR-AhmetNeural",
                "gemini": "Fenrir",
            },
        },
        {
            "id": "ayse",
            "name": "Ayşe",
            "description": "Journalist with short brown hair",
            "personality": "Curious and brave",
            "voice": "voice_ayse",
            "voice_gender": "female",
            "voice_profile": {
                "edge": "tr-TR-EmelNeural",
                "gemini": "Kore",
            },
        },
    ]

    scenes = []

    for index in range(1, 9):
        scenes.append({
            "scene_id": f"scene_{index:02d}",
            "narration": f"Scene {index} narration.",
            "visual_subject": "Ali and Ayşe",
            "action": "They walk through the old city.",
            "location": "Old city street",
            "props": "Street lamps",
            "camera": "Slow cinematic tracking shot",
            "image_prompt": "Realistic cinematic city scene",
            "motion_prompt": "Both characters walk naturally",
            "sound": "Wind and distant city ambience",
            "dialogue_turns": [],
            "shot_plan": [],
            "transition": "Soft cut",
        })

    scenes[0]["dialogue_turns"] = [
        {
            "speaker": "Ali",
            "text": "Burada bir şey var.",
            "emotion": "curious",
            "delivery": "quiet",
            "listener_ids": ["Ayşe"],
        },
        {
            "speaker": "Ayşe",
            "text": "Neyi fark ettin?",
            "emotion": "curious",
            "delivery": "natural",
            "listener_ids": ["Ali"],
        },
    ]

    return {
        "title": "The Hidden Signal",
        "description": "A cinematic mystery.",
        "style": "cinematic realistic",
        "character_bible": "Keep both characters visually consistent.",
        "tags": ["mystery", "cinematic"],
        "characters": characters,
        "scenes": scenes,
    }


class FilmProjectTests(unittest.TestCase):

    def setUp(self):
        self.script = make_valid_script()

    def build_project(self, **kwargs):
        options = {
            "script": self.script,
            "topic": "The Hidden Signal",
            "target_seconds": 60,
            "language": "Turkish",
            "visual_style": "cinematic realistic",
        }
        options.update(kwargs)
        return build_film_project(**options)

    def test_valid_script_builds_valid_project(self):
        project = self.build_project()
        report = validate_film_project(project)

        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["scene_count"], 8)
        self.assertEqual(report["character_count"], 2)
        self.assertEqual(report["dialogue_turn_count"], 2)

    def test_production_defaults_are_applied(self):
        project = self.build_project()

        self.assertEqual(
            project["production"],
            PRODUCTION_DEFAULTS,
        )

    def test_character_names_are_resolved_to_ids(self):
        project = self.build_project()

        first_turn = project["scenes"][0]["dialogue_turns"][0]

        self.assertEqual(first_turn["speaker"], "ali")
        self.assertEqual(first_turn["listener_ids"], ["ayse"])

    def test_voice_profiles_are_preserved(self):
        project = self.build_project()

        self.assertEqual(
            project["characters"][0]["voice_profile"]["gemini"],
            "Fenrir",
        )
        self.assertEqual(
            project["characters"][1]["voice_profile"]["edge"],
            "tr-TR-EmelNeural",
        )

    def test_character_ids_must_be_unique(self):
        script = copy.deepcopy(self.script)
        script["characters"][1]["id"] = "ali"

        with self.assertRaises(ValueError):
            self.build_project(script=script)

    def test_character_names_must_be_unique(self):
        script = copy.deepcopy(self.script)
        script["characters"][1]["name"] = "Ali"

        with self.assertRaises(ValueError):
            self.build_project(script=script)

    def test_unknown_dialogue_speaker_is_rejected(self):
        script = copy.deepcopy(self.script)
        script["scenes"][0]["dialogue_turns"][0]["speaker"] = "Unknown"

        with self.assertRaises(ValueError):
            self.build_project(script=script)

    def test_duplicate_scene_ids_are_rejected(self):
        script = copy.deepcopy(self.script)
        script["scenes"][1]["scene_id"] = "scene_01"

        with self.assertRaises(ValueError):
            self.build_project(script=script)

    def test_missing_required_scene_field_is_rejected(self):
        script = copy.deepcopy(self.script)
        script["scenes"][0]["sound"] = ""

        with self.assertRaises(ValueError):
            self.build_project(script=script)

    def test_scene_count_below_minimum_is_rejected(self):
        script = copy.deepcopy(self.script)
        script["scenes"] = script["scenes"][:7]

        with self.assertRaises(ValueError):
            self.build_project(script=script)

    def test_character_count_above_maximum_is_rejected(self):
        script = copy.deepcopy(self.script)

        for index in range(7):
            script["characters"].append({
                "id": f"extra_{index}",
                "name": f"Extra {index}",
            })

        with self.assertRaises(ValueError):
            self.build_project(script=script)

    def test_duration_is_clamped_to_supported_range(self):
        short_project = self.build_project(target_seconds=1)
        long_project = self.build_project(target_seconds=999)

        self.assertEqual(
            short_project["film"]["target_duration_seconds"],
            15,
        )
        self.assertEqual(
            long_project["film"]["target_duration_seconds"],
            180,
        )

    def test_invalid_production_settings_are_rejected(self):
        project = self.build_project()
        project["production"]["fps"] = 30

        report = validate_film_project(project)

        self.assertFalse(report["valid"])
        self.assertTrue(
            any("fps" in error for error in report["errors"])
        )

    def test_project_without_character_dialogue_is_rejected(self):
        script = copy.deepcopy(self.script)

        for scene in script["scenes"]:
            scene["dialogue_turns"] = []

        with self.assertRaises(ValueError):
            self.build_project(script=script)

    def test_legacy_adapter_preserves_project_content(self):
        project = self.build_project()
        legacy = to_legacy_script(project)

        self.assertEqual(legacy["title"], project["film"]["title"])
        self.assertEqual(legacy["characters"], project["characters"])
        self.assertEqual(legacy["scenes"], project["scenes"])
        self.assertFalse(legacy["fallback_mode"])

    def test_legacy_adapter_returns_independent_data(self):
        project = self.build_project()
        legacy = to_legacy_script(project)

        legacy["scenes"][0]["narration"] = "Changed"

        self.assertNotEqual(
            project["scenes"][0]["narration"],
            legacy["scenes"][0]["narration"],
        )

    def test_invalid_project_cannot_be_converted(self):
        project = self.build_project()
        project["characters"] = []

        with self.assertRaises(ValueError):
            to_legacy_script(project)

    def test_non_object_project_is_rejected(self):
        report = validate_film_project(None)

        self.assertFalse(report["valid"])
        self.assertGreater(len(report["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
