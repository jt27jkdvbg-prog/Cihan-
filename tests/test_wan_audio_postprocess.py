import tempfile
import unittest
from pathlib import Path

from wan_audio_postprocess import (
    build_profiles,
    build_utterances,
    clean_text,
    estimate_word_times,
    timestamp,
    write_srt,
)

from film_project import build_film_project
from project_adapter import legacy_to_pipeline_project
from wan_project_input import normalize_wan_project


class WanAudioPostprocessTests(unittest.TestCase):

    def setUp(self):
        self.characters = [
            {
                "id": "ali",
                "name": "Ali",
                "voice_gender": "male",
            },
            {
                "id": "ayse",
                "name": "Ayşe",
                "voice_gender": "female",
            },
        ]

    def test_timestamp_format(self):
        self.assertEqual(
            timestamp(65.25),
            "00:01:05,250",
        )

    def test_clean_text(self):
        self.assertEqual(
            clean_text("  Merhaba dünya  "),
            "Merhaba dünya",
        )
        self.assertEqual(clean_text(None), "")

    def test_word_timing_is_ordered(self):
        result = estimate_word_times(
            "Merhaba dünya. Nasılsın?",
            4.0,
        )

        self.assertEqual(len(result), 3)

        for index, item in enumerate(result):
            self.assertLess(item[1], item[2])

            if index > 0:
                self.assertGreaterEqual(
                    item[1],
                    result[index - 1][1],
                )

    def test_character_voice_profiles(self):
        profiles = build_profiles(self.characters)

        self.assertEqual(len(profiles), 2)
        self.assertIn("ali", profiles)
        self.assertIn("ayse", profiles)

        self.assertNotEqual(
            profiles["ali"]["gemini"],
            profiles["ayse"]["gemini"],
        )

    def test_narration_and_dialogue_are_resolved(self):
        profiles = build_profiles(self.characters)

        scene = {
            "narration": "Gece sessizdi.",
            "dialogue_turns": [
                {
                    "speaker": "ali",
                    "text": "Burada ne oldu?",
                },
                {
                    "speaker": "ayse",
                    "text": "Henüz bilmiyorum.",
                },
            ],
        }

        result = build_utterances(scene, profiles)

        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0]["speaker_id"])
        self.assertEqual(result[1]["speaker_id"], "ali")
        self.assertEqual(result[2]["speaker_id"], "ayse")

    def test_unknown_speaker_is_rejected(self):
        profiles = build_profiles(self.characters)

        with self.assertRaises(ValueError):
            build_utterances(
                {
                    "dialogue_turns": [
                        {
                            "speaker": "unknown",
                            "text": "Merhaba",
                        }
                    ]
                },
                profiles,
            )

    def test_srt_file_is_created(self):
        entries = [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "Merhaba dünya.",
            }
        ]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.srt"

            write_srt(entries, path)

            content = path.read_text(encoding="utf-8")

            self.assertIn("00:00:00,000", content)
            self.assertIn("00:00:01,500", content)
            self.assertIn("Merhaba dünya.", content)

    def test_audio_settings_survive_project_normalization(self):
        legacy_project = build_film_project(
            script={
                "characters": [
                    {
                        "id": "ali",
                        "name": "Ali",
                        "description": "Ana karakter",
                        "personality": "Cesur",
                        "voice_gender": "male",
                    },
                    {
                        "id": "ayse",
                        "name": "Ayşe",
                        "description": "İkinci karakter",
                        "personality": "Kararlı",
                        "voice_gender": "female",
                    },
                ],
                "scenes": [
                    {
                        "scene_number": 1,
                        "description": "Gece vakti şehir",
                        "narration": "Şehir sessizliğe gömülmüştü.",
                        "dialogue_turns": [
                            {
                                "speaker": "ali",
                                "text": "Burada ne oldu?",
                            }
                        ],
                    }
                ],
            },
            topic="Son Işık",
            target_seconds=60,
        )

        if hasattr(legacy_project, "to_dict"):
            legacy_project = legacy_project.to_dict()

        modern_project = legacy_to_pipeline_project(
            legacy_project
        )

        if hasattr(modern_project, "to_dict"):
            modern_project = modern_project.to_dict()

        expected_audio = {
            "enabled": True,
            "narration": True,
            "subtitles": True,
            "music": False,
            "ambience": False,
            "sound_effects": False,
        }

        expected_cost_policy = {
            "mode": "free_only",
            "allow_paid_services": False,
            "budget": 0,
        }

        modern_project["audio"] = expected_audio
        modern_project["cost_policy"] = expected_cost_policy

        normalized = normalize_wan_project(modern_project)

        self.assertEqual(
            normalized["audio"],
            expected_audio,
        )

        self.assertEqual(
            normalized["cost_policy"],
            expected_cost_policy,
        )


if __name__ == "__main__":
    unittest.main()
