
import copy
import unittest

from film_project import build_film_project
from pipeline_config import ProjectConfigError
from project_adapter import (
    LEGACY_PRODUCTION,
    legacy_to_pipeline_project,
    pipeline_project_to_legacy,
)


def make_legacy_project():
    characters = [
        {
            "id": "ali",
            "name": "Ali",
            "description": "Dark hair, navy coat",
            "personality": "Calm and observant",
            "voice": "ali_voice",
            "voice_gender": "male",
            "voice_profile": {"edge": "tr-TR-AhmetNeural"},
        },
        {
            "id": "ayse",
            "name": "Ayşe",
            "description": "Short brown hair, beige jacket",
            "personality": "Curious and brave",
            "voice": "ayse_voice",
            "voice_gender": "female",
            "voice_profile": {"edge": "tr-TR-EmelNeural"},
        },
    ]

    scenes = []

    for index in range(1, 9):
        scenes.append({
            "scene_id": f"scene_{index:02d}",
            "narration": f"Scene {index} narration.",
            "visual_subject": "Ali and Ayşe",
            "action": "They walk through the city.",
            "location": "Istanbul",
            "props": "Street lamps",
            "camera": "Slow tracking shot",
            "image_prompt": "Cinematic realistic Istanbul street.",
            "motion_prompt": "Characters walk naturally.",
            "sound": "City ambience and footsteps.",
            "dialogue_turns": [],
            "shot_plan": [],
            "transition": "cut",
        })

    scenes[0]["dialogue_turns"] = [
        {
            "speaker": "ali",
            "text": "Burada bir şey var.",
            "emotion": "curious",
            "delivery": "quiet",
            "listener_ids": ["ayse"],
        },
        {
            "speaker": "ayse",
            "text": "Neyi fark ettin?",
            "emotion": "curious",
            "delivery": "natural",
            "listener_ids": ["ali"],
        },
    ]

    return build_film_project(
        script={
            "title": "The Hidden Signal",
            "description": "A cinematic mystery.",
            "style": "cinematic realistic",
            "character_bible": "Keep character appearances consistent.",
            "tags": ["mystery", "cinematic"],
            "characters": characters,
            "scenes": scenes,
        },
        topic="The Hidden Signal",
        target_seconds=60,
        language="Turkish",
    )


class ProjectAdapterTests(unittest.TestCase):

    def setUp(self):
        self.legacy = make_legacy_project()

    def test_legacy_project_converts_to_pipeline_schema(self):
        project = legacy_to_pipeline_project(self.legacy)

        self.assertEqual(project.title, "The Hidden Signal")
        self.assertEqual(len(project.characters), 2)
        self.assertEqual(len(project.scenes), 8)
        self.assertEqual(project.characters[0].id, "ali")
        self.assertEqual(project.scenes[0].dialogue_turns[0]["speaker"], "ali")

    def test_legacy_production_settings_are_preserved(self):
        project = legacy_to_pipeline_project(self.legacy)

        for key, value in LEGACY_PRODUCTION.items():
            self.assertEqual(project.production_settings[key], value)

        self.assertEqual(project.production_settings["fps"], 16)
        self.assertEqual(project.production_settings["resolution"], "832x480")

    def test_legacy_scene_content_is_preserved(self):
        project = legacy_to_pipeline_project(self.legacy)
        scene = project.scenes[0]

        self.assertEqual(
            scene.metadata["legacy_image_prompt"],
            "Cinematic realistic Istanbul street.",
        )
        self.assertEqual(
            scene.metadata["legacy_motion_prompt"],
            "Characters walk naturally.",
        )
        self.assertEqual(
            scene.metadata["legacy_sound"],
            "City ambience and footsteps.",
        )
        self.assertEqual(
            scene.metadata["legacy_narration"],
            "Scene 1 narration.",
        )
        self.assertEqual(len(scene.dialogue_turns), 2)

    def test_pipeline_project_converts_back_to_legacy(self):
        project = legacy_to_pipeline_project(self.legacy)
        converted = pipeline_project_to_legacy(project)

        self.assertEqual(
            converted["film"]["title"],
            self.legacy["film"]["title"],
        )
        self.assertEqual(
            converted["characters"][0]["id"],
            self.legacy["characters"][0]["id"],
        )
        self.assertEqual(
            converted["scenes"][0]["image_prompt"],
            self.legacy["scenes"][0]["image_prompt"],
        )
        self.assertEqual(
            converted["scenes"][0]["motion_prompt"],
            self.legacy["scenes"][0]["motion_prompt"],
        )
        self.assertEqual(
            converted["scenes"][0]["sound"],
            self.legacy["scenes"][0]["sound"],
        )
        self.assertEqual(
            converted["scenes"][0]["dialogue_turns"],
            self.legacy["scenes"][0]["dialogue_turns"],
        )

    def test_round_trip_preserves_wan_production_format(self):
        project = legacy_to_pipeline_project(self.legacy)
        converted = pipeline_project_to_legacy(project)

        self.assertEqual(
            converted["production"],
            {
                "aspect_ratio": "26:15",
                "resolution": "832x480",
                "fps": 16,
            },
        )

    def test_invalid_legacy_project_is_rejected(self):
        invalid = copy.deepcopy(self.legacy)
        invalid["scenes"] = []

        with self.assertRaises(ProjectConfigError):
            legacy_to_pipeline_project(invalid)

    def test_pipeline_project_with_wrong_production_settings_is_rejected(self):
        project = legacy_to_pipeline_project(self.legacy)
        project.production_settings["fps"] = 30

        with self.assertRaises(ProjectConfigError):
            pipeline_project_to_legacy(project)

    def test_pipeline_project_with_too_few_characters_is_rejected(self):
        project = legacy_to_pipeline_project(self.legacy)
        project.characters = project.characters[:1]

        with self.assertRaises(ProjectConfigError):
            pipeline_project_to_legacy(project)


if __name__ == "__main__":
    unittest.main()
