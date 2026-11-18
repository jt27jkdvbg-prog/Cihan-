
import copy
import unittest

from pipeline_config import ProjectConfigError
from project_adapter import legacy_to_pipeline_project
from test_project_adapter import make_legacy_project
from wan_project_input import normalize_wan_project


class WanProjectInputTests(unittest.TestCase):

    def setUp(self):
        self.legacy = make_legacy_project()

    def test_legacy_project_is_accepted_unchanged(self):
        original = copy.deepcopy(self.legacy)

        result = normalize_wan_project(self.legacy)

        self.assertEqual(result, original)

    def test_modern_project_is_converted_to_legacy_wan_format(self):
        modern = legacy_to_pipeline_project(self.legacy)

        result = normalize_wan_project(modern.to_dict())

        self.assertIn("film", result)
        self.assertIn("production", result)
        self.assertEqual(result["production"]["resolution"], "832x480")
        self.assertEqual(result["production"]["fps"], 16)
        self.assertEqual(len(result["characters"]), 2)
        self.assertEqual(len(result["scenes"]), 8)

    def test_modern_project_with_wrong_fps_is_rejected(self):
        modern = legacy_to_pipeline_project(self.legacy)
        modern.production_settings["fps"] = 30

        with self.assertRaises(ProjectConfigError):
            normalize_wan_project(modern.to_dict())

    def test_invalid_legacy_project_is_rejected(self):
        invalid = copy.deepcopy(self.legacy)
        invalid["scenes"] = []

        with self.assertRaises(ProjectConfigError):
            normalize_wan_project(invalid)

    def test_non_object_input_is_rejected(self):
        with self.assertRaises(ProjectConfigError):
            normalize_wan_project([])


if __name__ == "__main__":
    unittest.main()
