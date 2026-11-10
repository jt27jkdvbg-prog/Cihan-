import unittest

from speech_director import (
    build_voice_profiles,
    build_speech_schedule,
)


class SpeechDirectorTests(unittest.TestCase):

    def setUp(self):
        self.characters = [
            {
                "id": "ali",
                "name": "Ali",
                "voice_gender": "male",
                "personality": "Calm",
            },
            {
                "id": "ayse",
                "name": "Ayşe",
                "voice_gender": "female",
                "personality": "Curious",
            },
            {
                "id": "mert",
                "name": "Mert",
                "voice_gender": "male",
                "personality": "Direct",
            },
        ]

    def test_three_characters_get_unique_gemini_voices(self):
        profiles = build_voice_profiles(self.characters)

        voices = [p["gemini"] for p in profiles.values()]

        self.assertEqual(len(profiles), 3)
        self.assertEqual(len(set(voices)), 3)

    def test_character_voice_profile_is_preserved(self):
        characters = [
            {
                "id": "ali",
                "name": "Ali",
                "voice_profile": {
                    "gemini": "Fenrir",
                    "edge": "tr-TR-AhmetNeural",
                },
            },
            {
                "id": "ayse",
                "name": "Ayşe",
                "voice_profile": {
                    "gemini": "Kore",
                    "edge": "tr-TR-EmelNeural",
                },
            },
        ]

        profiles = build_voice_profiles(characters)

        self.assertEqual(profiles["ali"]["gemini"], "Fenrir")
        self.assertEqual(profiles["ayse"]["gemini"], "Kore")

    def test_dialogue_schedule_preserves_speaker_order(self):
        scenes = [{
            "dialogue_turns": [
                {"speaker": "ali", "text": "Ne oldu?"},
                {"speaker": "ayse", "text": "Bir şey buldum."},
                {"speaker": "mert", "text": "Göster bakalım."},
            ]
        }]

        schedule = build_speech_schedule(scenes, self.characters)

        self.assertEqual(len(schedule), 3)
        self.assertEqual(
            [turn["speaker_id"] for turn in schedule],
            ["ali", "ayse", "mert"],
        )

    def test_unknown_speaker_is_rejected(self):
        scenes = [{
            "dialogue_turns": [
                {"speaker": "unknown", "text": "Merhaba"}
            ]
        }]

        with self.assertRaises(ValueError):
            build_speech_schedule(scenes, self.characters)

    def test_empty_dialogue_is_rejected(self):
        scenes = [{
            "dialogue_turns": [
                {"speaker": "ali", "text": ""}
            ]
        }]

        with self.assertRaises(ValueError):
            build_speech_schedule(scenes, self.characters)

    def test_listener_ids_are_preserved(self):
        scenes = [{
            "dialogue_turns": [
                {
                    "speaker": "ali",
                    "text": "Beni dinleyin.",
                    "listener_ids": ["ayse", "mert", "ali"],
                }
            ]
        }]

        schedule = build_speech_schedule(scenes, self.characters)

        self.assertEqual(
            schedule[0]["listener_ids"],
            ["ayse", "mert"],
        )

    def test_cast_size_is_validated(self):
        with self.assertRaises(ValueError):
            build_voice_profiles(self.characters[:1])


if __name__ == "__main__":
    unittest.main()
