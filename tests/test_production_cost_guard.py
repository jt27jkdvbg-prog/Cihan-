
import tempfile
import unittest
from pathlib import Path

import production_cost_guard as guard


class ProductionCostGuardTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.music = self.root / "music.wav"
        self.ambience = self.root / "ambience.wav"
        self.effect = self.root / "effect.wav"

        for path in (self.music, self.ambience, self.effect):
            path.write_bytes(b"test-audio")

    def tearDown(self):
        self.temp.cleanup()

    def base_project(self):
        return {
            "cost_policy": {
                "mode": "free_only",
                "paid_services_allowed": False,
                "max_project_budget": 0,
            },
            "audio": {
                "tts_provider_order": ["edge"],
                "music": {"enabled": False},
                "ambience": {"enabled": False},
                "sound_effects": {"enabled": False},
            },
            "scenes": [],
        }

    def test_verified_local_music_is_allowed(self):
        project = self.base_project()
        project["audio"]["music"] = {
            "enabled": True,
            "provider": "licensed_library",
            "asset_path": str(self.music),
            "license_verified": True,
        }

        report = guard.build_report(
            project,
            project_dir=self.root,
        )

        self.assertTrue(report["production_allowed"])
        self.assertEqual(report["blocked_count"], 0)
        self.assertTrue(report["library_assets"][0]["valid"])

    def test_unverified_music_is_blocked(self):
        project = self.base_project()
        project["audio"]["music"] = {
            "enabled": True,
            "provider": "licensed_library",
            "asset_path": str(self.music),
            "license_verified": False,
        }

        report = guard.build_report(
            project,
            project_dir=self.root,
        )

        self.assertFalse(report["production_allowed"])
        self.assertEqual(report["blocked_count"], 1)

    def test_missing_music_file_is_blocked(self):
        project = self.base_project()
        project["audio"]["music"] = {
            "enabled": True,
            "provider": "licensed_library",
            "asset_path": str(self.root / "missing.wav"),
            "license_verified": True,
        }

        report = guard.build_report(
            project,
            project_dir=self.root,
        )

        self.assertFalse(report["production_allowed"])
        self.assertFalse(report["library_assets"][0]["file_exists"])

    def test_verified_ambience_is_allowed(self):
        project = self.base_project()
        project["audio"]["ambience"] = {
            "enabled": True,
            "provider": "library",
            "asset_path": str(self.ambience),
            "license_verified": True,
        }

        report = guard.build_report(
            project,
            project_dir=self.root,
        )

        self.assertTrue(report["production_allowed"])
        self.assertTrue(report["library_assets"][0]["valid"])

    def test_scene_effect_assets_are_checked(self):
        project = self.base_project()
        project["audio"]["sound_effects"] = {
            "enabled": True,
            "provider": "licensed_library",
        }
        project["scenes"] = [
            {
                "sound_effects": [
                    {
                        "asset_path": str(self.effect),
                        "license_verified": True,
                    }
                ]
            }
        ]

        report = guard.build_report(
            project,
            project_dir=self.root,
        )

        self.assertTrue(report["production_allowed"])
        self.assertEqual(len(report["library_assets"]), 1)
        self.assertTrue(report["library_assets"][0]["valid"])

    def test_missing_scene_effect_is_blocked(self):
        project = self.base_project()
        project["audio"]["sound_effects"] = {
            "enabled": True,
            "provider": "licensed_library",
        }
        project["scenes"] = [
            {
                "sound_effects": [
                    {
                        "asset_path": str(self.root / "missing.wav"),
                        "license_verified": True,
                    }
                ]
            }
        ]

        report = guard.build_report(
            project,
            project_dir=self.root,
        )

        self.assertFalse(report["production_allowed"])

    def test_paid_fal_service_remains_blocked(self):
        project = self.base_project()
        project["audio"]["music"] = {
            "enabled": True,
            "provider": "fal",
        }
        project["cost_policy"] = {
            "mode": "paid_allowed",
            "paid_services_allowed": True,
            "max_project_budget": 100,
        }

        report = guard.build_report(
            project,
            approval=True,
        )

        self.assertFalse(report["production_allowed"])

    def test_unknown_provider_is_blocked(self):
        project = self.base_project()
        project["audio"]["tts_provider_order"] = ["unknown_provider"]

        report = guard.build_report(project)

        self.assertFalse(report["production_allowed"])

    def test_gemini_requires_quota_confirmation(self):
        project = self.base_project()
        project["audio"]["tts_provider_order"] = ["gemini"]

        blocked = guard.build_report(
            project,
            quota_confirmed=False,
        )
        allowed = guard.build_report(
            project,
            quota_confirmed=True,
        )

        self.assertFalse(blocked["production_allowed"])
        self.assertTrue(allowed["production_allowed"])


if __name__ == "__main__":
    unittest.main()
