
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from character_asset_generator import (
    CharacterAssetError,
    generate_character_assets,
)


def make_project():
    return {
        "project_id": "character-test",
        "title": "Character Reference Test",
        "brief": "A cinematic character continuity test.",
        "target_duration_seconds": 10,
        "visual_style": "cinematic realistic",
        "profile": "landscape",
        "language": "tr",
        "character_bible": "Keep the same face and clothing in every scene.",
        "characters": [
            {
                "id": "main-hero",
                "name": "Main Hero",
                "visual_description": (
                    "A fictional adult with short dark hair and brown eyes."
                ),
                "appearance": "Distinctive facial features.",
                "clothing": "Dark blue jacket and neutral trousers.",
                "personality": "Calm and determined.",
                "continuity_prompt": (
                    "Never change hairstyle, face shape or jacket."
                ),
            }
        ],
        "scenes": [
            {
                "id": "scene-1",
                "order": 1,
                "duration_seconds": 10,
                "visual_prompt": "The character enters a quiet room.",
                "character_ids": ["main-hero"],
            }
        ],
    }


class CharacterAssetGeneratorTests(unittest.TestCase):

    def fake_image_generation(self, prompt, path, engines):
        Path(path).write_bytes(
            b"\xff\xd8\xff" + (b"image-data" * 700)
        )
        return "fal"

    def test_generates_and_registers_character_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "character_asset_generator.image_pipeline.generate_image",
                side_effect=self.fake_image_generation,
            ):
                manifest = generate_character_assets(
                    make_project(),
                    temporary,
                    engines=["fal"],
                )

            self.assertTrue(manifest["all_assets_validated"])
            self.assertEqual(manifest["asset_count"], 1)

            asset = manifest["assets"][0]
            image_path = Path(temporary) / asset["path"]

            self.assertTrue(image_path.is_file())
            self.assertGreater(image_path.stat().st_size, 5000)
            self.assertEqual(asset["character_id"], "main-hero")
            self.assertEqual(asset["engine"], "fal")
            self.assertTrue(asset["validated"])
            self.assertEqual(len(asset["sha256"]), 64)

            manifest_path = Path(temporary) / "character_assets.json"
            self.assertTrue(manifest_path.is_file())

            saved_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["asset_count"], 1)

    def test_updated_project_contains_reference_image_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "character_asset_generator.image_pipeline.generate_image",
                side_effect=self.fake_image_generation,
            ):
                generate_character_assets(
                    make_project(),
                    temporary,
                    engines=["fal"],
                )

            updated_path = (
                Path(temporary) / "project_with_character_assets.json"
            )
            updated = json.loads(
                updated_path.read_text(encoding="utf-8")
            )

            character = updated["characters"][0]
            self.assertTrue(character["reference_image"])
            self.assertEqual(
                character["reference_images"],
                [character["reference_image"]],
            )

            referenced_file = (
                Path(temporary) / character["reference_image"]
            )
            self.assertTrue(referenced_file.is_file())

    def test_empty_character_list_is_rejected(self):
        project = make_project()
        project["characters"] = []

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(CharacterAssetError):
                generate_character_assets(
                    project,
                    temporary,
                    engines=["fal"],
                )

    def test_invalid_image_content_is_rejected(self):
        def fake_invalid_generation(prompt, path, engines):
            Path(path).write_bytes(b"not-an-image" * 1000)
            return "fal"

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "character_asset_generator.image_pipeline.generate_image",
                side_effect=fake_invalid_generation,
            ):
                with self.assertRaises(CharacterAssetError):
                    generate_character_assets(
                        make_project(),
                        temporary,
                        engines=["fal"],
                    )

    def test_provider_failure_does_not_create_manifest(self):
        def fake_failure(prompt, path, engines):
            raise RuntimeError("Provider unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "character_asset_generator.image_pipeline.generate_image",
                side_effect=fake_failure,
            ):
                with self.assertRaises(CharacterAssetError):
                    generate_character_assets(
                        make_project(),
                        temporary,
                        engines=["fal"],
                    )

            self.assertFalse(
                (Path(temporary) / "character_assets.json").exists()
            )

    def test_prompt_contains_identity_and_continuity_rules(self):
        prompts = []

        def fake_capture(prompt, path, engines):
            prompts.append(prompt)
            Path(path).write_bytes(
                b"\xff\xd8\xff" + (b"image-data" * 700)
            )
            return "fal"

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "character_asset_generator.image_pipeline.generate_image",
                side_effect=fake_capture,
            ):
                generate_character_assets(
                    make_project(),
                    temporary,
                    engines=["fal"],
                )

        self.assertEqual(len(prompts), 1)
        self.assertIn("Main Hero", prompts[0])
        self.assertIn("Never change hairstyle", prompts[0])
        self.assertIn("one person only", prompts[0])


if __name__ == "__main__":
    unittest.main()
