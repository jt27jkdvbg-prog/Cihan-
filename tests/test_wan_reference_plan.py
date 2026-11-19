
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wan_reference_plan import (
    WanReferencePlanError,
    build_vace_reference_plan,
    write_vace_reference_plan,
)


def make_project():
    return {
        "project_id": "reference-plan-test",
        "title": "Reference Plan Test",
        "brief": "Testing per-scene character references.",
        "target_duration_seconds": 20,
        "visual_style": "cinematic realistic",
        "profile": "landscape",
        "language": "tr",
        "characters": [
            {
                "id": "hero",
                "name": "Hero",
                "visual_description": "Fictional adult character.",
                "appearance": "Short dark hair.",
                "clothing": "Dark jacket.",
            },
            {
                "id": "friend",
                "name": "Friend",
                "visual_description": "Fictional adult character.",
                "appearance": "Curly brown hair.",
                "clothing": "Light coat.",
            },
        ],
        "scenes": [
            {
                "id": "scene-2",
                "order": 2,
                "duration_seconds": 10,
                "visual_prompt": "The hero walks through a city.",
                "character_ids": ["hero"],
            },
            {
                "id": "scene-1",
                "order": 1,
                "duration_seconds": 10,
                "visual_prompt": "The hero meets a friend.",
                "character_ids": ["hero", "friend"],
            },
        ],
    }


class WanReferencePlanTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.assets_root = (
            self.root
            / "artifacts"
            / "character_assets"
        )

        self.assets_root.mkdir(parents=True)

        self.project = make_project()

        self.records = []

        for character in self.project["characters"]:
            character_dir = (
                self.assets_root / character["id"]
            )
            character_dir.mkdir()

            image = character_dir / "anchor.jpg"
            image.write_bytes(
                b"\xff\xd8\xff"
                + (character["id"].encode() * 1500)
            )

            self.records.append({
                "character_id": character["id"],
                "character_name": character["name"],
                "asset_type": "canonical_character_anchor",
                "path": (
                    f"artifacts/character_assets/"
                    f"{character['id']}/anchor.jpg"
                ),
                "engine": "fal",
                "bytes": image.stat().st_size,
                "sha256": hashlib.sha256(
                    image.read_bytes()
                ).hexdigest(),
                "validated": True,
            })

        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, **overrides):
        manifest = {
            "schema_version": 1,
            "project_id": "reference-plan-test",
            "project_title": "Reference Plan Test",
            "asset_count": 2,
            "all_assets_validated": True,
            "registered_in_production_job": True,
            "assets": self.records,
        }

        manifest.update(overrides)

        path = self.assets_root / "character_assets.json"
        path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        return path

    def build(self):
        return build_vace_reference_plan(
            self.project,
            self.root,
        )

    def test_builds_reference_plan_for_each_scene(self):
        plan = self.build()

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["engine"], "wan-vace-r2v")
        self.assertEqual(
            plan["conditioning_mode"],
            "reference_to_video",
        )
        self.assertEqual(plan["scene_count"], 2)
        self.assertEqual(plan["character_count"], 2)
        self.assertEqual(plan["unique_reference_count"], 2)
        self.assertEqual(plan["scene_reference_count"], 3)

    def test_scenes_are_sorted_by_order(self):
        plan = self.build()

        self.assertEqual(
            [scene["scene_id"] for scene in plan["scenes"]],
            ["scene-1", "scene-2"],
        )

    def test_only_scene_characters_are_included(self):
        plan = self.build()

        first = plan["scenes"][0]
        second = plan["scenes"][1]

        self.assertEqual(
            first["character_ids"],
            ["hero", "friend"],
        )

        self.assertEqual(
            len(first["reference_images"]),
            2,
        )

        self.assertEqual(
            second["character_ids"],
            ["hero"],
        )

        self.assertEqual(
            len(second["reference_images"]),
            1,
        )

        self.assertEqual(
            second["reference_images"][0]["character_id"],
            "hero",
        )

    def test_reference_paths_are_workspace_relative(self):
        plan = self.build()

        for scene in plan["scenes"]:
            for reference in scene["reference_images"]:
                path = self.root / reference["path"]
                self.assertTrue(path.is_file())
                self.assertFalse(Path(reference["path"]).is_absolute())

    def test_scene_without_characters_has_no_references(self):
        project = make_project()
        project["scenes"][0]["character_ids"] = []
        project["scenes"][1]["character_ids"] = []

        plan = build_vace_reference_plan(
            project,
            self.root,
        )

        self.assertEqual(plan["engine"], "wan-t2v")
        self.assertEqual(
            plan["conditioning_mode"],
            "text_to_video",
        )
        self.assertEqual(plan["scene_reference_count"], 0)

        for scene in plan["scenes"]:
            self.assertEqual(scene["reference_images"], [])
            self.assertFalse(
                scene["reference_conditioning_required"]
            )

    def test_missing_manifest_is_rejected(self):
        (self.assets_root / "character_assets.json").unlink()

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_invalid_manifest_json_is_rejected(self):
        (self.assets_root / "character_assets.json").write_text(
            "{invalid",
            encoding="utf-8",
        )

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_wrong_project_id_is_rejected(self):
        self.write_manifest(project_id="another-project")

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_unregistered_assets_are_rejected(self):
        self.write_manifest(
            registered_in_production_job=False
        )

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_unvalidated_manifest_is_rejected(self):
        self.write_manifest(all_assets_validated=False)

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_wrong_asset_count_is_rejected(self):
        self.write_manifest(asset_count=1)

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_duplicate_character_record_is_rejected(self):
        records = list(self.records)
        records[1] = dict(records[0])

        self.write_manifest(assets=records)

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_unknown_character_record_is_rejected(self):
        records = [dict(item) for item in self.records]
        records[0]["character_id"] = "unknown"

        self.write_manifest(assets=records)

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_wrong_character_name_is_rejected(self):
        records = [dict(item) for item in self.records]
        records[0]["character_name"] = "Wrong Name"

        self.write_manifest(assets=records)

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_invalid_sha256_is_rejected(self):
        records = [dict(item) for item in self.records]
        records[0]["sha256"] = "0" * 64

        self.write_manifest(assets=records)

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_wrong_image_size_is_rejected(self):
        records = [dict(item) for item in self.records]
        records[0]["bytes"] += 1

        self.write_manifest(assets=records)

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_missing_image_is_rejected(self):
        image = (
            self.root
            / self.records[0]["path"]
        )
        image.unlink()

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_unsupported_image_format_is_rejected(self):
        image = (
            self.root
            / self.records[0]["path"]
        )
        image.write_bytes(b"NOT-AN-IMAGE" * 1000)

        records = [dict(item) for item in self.records]
        records[0]["bytes"] = image.stat().st_size
        records[0]["sha256"] = hashlib.sha256(
            image.read_bytes()
        ).hexdigest()

        self.write_manifest(assets=records)

        with self.assertRaises(WanReferencePlanError):
            self.build()

    def test_path_traversal_is_rejected(self):
        records = [dict(item) for item in self.records]
        records[0]["path"] = "../../outside.jpg"

        self.write_manifest(assets=records)

        with self.assertRaises(WanReferencePlanError):
            self.build()

        def test_unknown_scene_character_is_rejected(self):
        project = make_project()
        project["scenes"][0]["character_ids"] = [
            "not-in-project",
        ]

        from pipeline_config import ProjectConfigError

        with self.assertRaises(ProjectConfigError):
            build_vace_reference_plan(
                project,
                self.root,
            )

    def test_unknown_scene_character_is_rejected(self):
        project = make_project()
        project["scenes"][0]["character_ids"] = [
            "not-in-project",
        ]

        with self.assertRaises(WanReferencePlanError):
            build_vace_reference_plan(
                project,
                self.root,
            )

    def test_missing_workspace_is_rejected(self):
        with self.assertRaises(WanReferencePlanError):
            build_vace_reference_plan(
                self.project,
                self.root / "missing",
            )

    def test_write_creates_valid_json_plan(self):
        output = write_vace_reference_plan(
            self.project,
            self.root,
        )

        self.assertTrue(output.is_file())

        saved = json.loads(
            output.read_text(encoding="utf-8")
        )

        self.assertEqual(saved["status"], "ready")
        self.assertEqual(saved["scene_count"], 2)

    def test_output_path_cannot_escape_workspace(self):
        with self.assertRaises(WanReferencePlanError):
            write_vace_reference_plan(
                self.project,
                self.root,
                self.root.parent / "outside.json",
            )


if __name__ == "__main__":
    unittest.main()
