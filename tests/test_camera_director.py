import unittest

from camera_director import (
    all_camera_preset_names,
    build_camera_plan,
    build_zoompan_filter,
    camera_category,
)


class CameraDirectorTests(unittest.TestCase):

    def test_exposes_at_least_30_combinations(self):
        self.assertGreaterEqual(
            len(all_camera_preset_names()), 30
        )

    def test_scene_hints_select_expected_category(self):
        self.assertEqual(
            camera_category({"camera": "Extreme close-up of face"}),
            "closeup",
        )
        self.assertEqual(
            camera_category({"action": "Aerial reveal of the valley"}),
            "wide",
        )
        self.assertEqual(
            camera_category({"action": "The hero runs through the street"}),
            "tracking",
        )
        self.assertEqual(
            camera_category({"narration": "A dramatic battle begins"}),
            "dramatic",
        )

    def test_plan_is_deterministic(self):
        scene = {
            "camera": "tracking shot",
            "action": "follow the character",
        }
        self.assertEqual(
            build_camera_plan(scene, 3),
            build_camera_plan(scene, 3),
        )

    def test_zoompan_filter_contains_requested_output(self):
        result = build_zoompan_filter(
            {"camera": "wide shot"},
            2,
            135,
            1080,
            1920,
            30,
        )

        self.assertIn("d=135", result)
        self.assertIn("s=1080x1920", result)
        self.assertIn("fps=30", result)
        self.assertIn("zoompan=", result)

    def test_invalid_scene_falls_back_to_general(self):
        self.assertEqual(camera_category(None), "general")


if __name__ == "__main__":
    unittest.main()
