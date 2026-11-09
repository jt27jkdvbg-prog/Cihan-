
import unittest

from dialogue_director import (
    normalize_cast,
    normalize_dialogue_turns,
    validate_cast_and_dialogue,
)


class DialogueDirectorTests(unittest.TestCase):

    def setUp(self):
        self.cast = [
            {
                "id": "ali",
                "name": "Ali",
                "description": "Young detective",
                "voice": "voice_a",
                "personality": "Calm",
            },
            {
                "id": "ayse",
                "name": "Ayşe",
                "description": "Journalist",
                "voice": "voice_b",
                "personality": "Curious",
            },
            {
                "id": "mert",
                "name": "Mert",
                "description": "Engineer",
                "voice": "voice_c",
                "personality": "Direct",
            },
        ]

    def test_three_character_cast(self):
        result = normalize_cast(self.cast)
        self.assertEqual(len(result), 3)

    def test_dialogue_speakers_are_resolved(self):
        scene = {
            "dialogue_turns": [
                {"speaker": "Ali", "text": "Burada ne oldu?"},
                {"speaker": "Ayşe", "text": "Henüz bilmiyorum."},
                {"speaker": "Mert", "text": "Bir ipucu buldum."},
            ]
        }
        turns = normalize_dialogue_turns(scene, self.cast)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[2]["speaker_id"], "mert")

    def test_unknown_speaker_is_rejected(self):
        scene = {
            "dialogue_turns": [
                {"speaker": "Unknown", "text": "Merhaba"}
            ]
        }
        with self.assertRaises(ValueError):
            normalize_dialogue_turns(scene, self.cast)

    def test_legacy_dialogue_without_speaker_is_rejected(self):
        scene = {"dialogue": "Merhaba"}
        with self.assertRaises(ValueError):
            normalize_dialogue_turns(scene, self.cast)

    def test_cast_must_have_two_to_eight_characters(self):
        with self.assertRaises(ValueError):
            normalize_cast(self.cast[:1])

    def test_validation_accepts_multi_character_dialogue(self):
        scenes = [{
            "dialogue_turns": [
                {"speaker": "Ali", "text": "Ne yapacağız?"},
                {"speaker": "Ayşe", "text": "Birlikte çözeceğiz."},
            ]
        }]
        result = validate_cast_and_dialogue(scenes, self.cast)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["cast_count"], 3)
        self.assertEqual(result["dialogue_turn_count"], 2)


if __name__ == "__main__":
    unittest.main()
