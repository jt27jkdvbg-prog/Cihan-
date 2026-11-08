
import unittest

from cinematic_director import (
    build_script_prompt,
    enrich_image_prompt,
    validate_script,
)
from quality_gate import require_real_motion


class CinematicDirectorTests(unittest.TestCase):

    def test_prompt_includes_topic_and_vertical_format(self):
        prompt = build_script_prompt("Kayıp Tanrıların Mührü", 60)
        self.assertIn("Kayıp Tanrıların Mührü", prompt)
        self.assertIn("9:16", prompt)

    def test_character_continuity(self):
        result = enrich_image_prompt(
            "cinematic",
            "warrior with red cloak",
            {"image_prompt": "standing in an ancient temple"},
        )
        self.assertIn("red cloak", result)
        self.assertIn("ancient temple", result)

    def test_valid_script(self):
        scenes = [
            {
                "narration": "A short narration",
                "image_prompt": "A cinematic scene",
                "motion_prompt": "The hero walks forward",
                "sound": "Wind and distant thunder",
            }
            for _ in range(8)
        ]

        result = validate_script(
            {"title": "Test Film", "style": "cinematic", "scenes": scenes},
            60,
        )
        self.assertTrue(result["valid"], result["errors"])

    def test_invalid_script(self):
        result = validate_script(
            {"title": "Test Film", "scenes": [{"narration": "Hello"}]},
            30,
        )
        self.assertFalse(result["valid"])

    def test_real_motion_requires_provider(self):
        with self.assertRaises(RuntimeError):
            require_real_motion("real", False)

    def test_preview_mode(self):
        self.assertFalse(require_real_motion("preview", False))


if __name__ == "__main__":
    unittest.main()
