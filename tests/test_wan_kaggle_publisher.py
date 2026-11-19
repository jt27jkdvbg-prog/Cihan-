
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wan_kaggle_publisher import (
    WanKagglePublishError,
    _attach_dataset_to_kernel,
    publish_reference_dataset,
)


class WanKagglePublisherTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.metadata = self.root / "kernel-metadata.json"
        self.metadata.write_text(
            json.dumps({
                "id": "test-owner/test-kernel",
                "is_private": True,
                "dataset_sources": ["test-owner/existing-data"],
            }),
            encoding="utf-8",
        )

        self.payload = self.root / "payload"
        self.payload.mkdir()

        self.output = self.root / "dataset-output"

    def tearDown(self):
        self.temp.cleanup()

    def read_metadata(self):
        return json.loads(
            self.metadata.read_text(encoding="utf-8")
        )

    def test_attaches_dataset_without_removing_existing_sources(self):
        _attach_dataset_to_kernel(
            self.metadata,
            "test-owner/new-data",
        )

        metadata = self.read_metadata()

        self.assertEqual(
            metadata["dataset_sources"],
            [
                "test-owner/existing-data",
                "test-owner/new-data",
            ],
        )

    def test_does_not_duplicate_dataset_source(self):
        _attach_dataset_to_kernel(
            self.metadata,
            "test-owner/existing-data",
        )

        metadata = self.read_metadata()

        self.assertEqual(
            metadata["dataset_sources"],
            ["test-owner/existing-data"],
        )

    def test_rejects_public_kernel(self):
        self.metadata.write_text(
            json.dumps({
                "is_private": False,
                "dataset_sources": [],
            }),
            encoding="utf-8",
        )

        with self.assertRaises(WanKagglePublishError):
            _attach_dataset_to_kernel(
                self.metadata,
                "test-owner/new-data",
            )

    def test_rejects_missing_metadata_file(self):
        with self.assertRaises(WanKagglePublishError):
            _attach_dataset_to_kernel(
                self.root / "missing.json",
                "test-owner/new-data",
            )

    def test_rejects_invalid_dataset_sources(self):
        self.metadata.write_text(
            json.dumps({
                "is_private": True,
                "dataset_sources": "not-a-list",
            }),
            encoding="utf-8",
        )

        with self.assertRaises(WanKagglePublishError):
            _attach_dataset_to_kernel(
                self.metadata,
                "test-owner/new-data",
            )

    def test_rejects_invalid_json(self):
        self.metadata.write_text(
            "{invalid-json",
            encoding="utf-8",
        )

        with self.assertRaises(WanKagglePublishError):
            _attach_dataset_to_kernel(
                self.metadata,
                "test-owner/new-data",
            )

    @patch("wan_kaggle_publisher.subprocess.run")
    @patch("wan_kaggle_publisher.prepare_kaggle_dataset")
    @patch("wan_kaggle_publisher.shutil.which")
    def test_publishes_without_public_flag(
        self,
        mock_which,
        mock_prepare,
        mock_run,
    ):
        mock_which.return_value = "/usr/bin/kaggle"
        mock_prepare.return_value = self.output / "dataset-metadata.json"

        with patch.dict(
            os.environ,
            {"KAGGLE_API_TOKEN": "test-token"},
        ):
            result = publish_reference_dataset(
                self.payload,
                self.output,
                owner="test-owner",
                slug="wan-test-data",
                kernel_metadata_path=self.metadata,
            )

        command = mock_run.call_args.args[0]

        self.assertIn("datasets", command)
        self.assertIn("create", command)
        self.assertIn("-r", command)
        self.assertIn("zip", command)
        self.assertNotIn("--public", command)

        self.assertEqual(
            result["dataset_id"],
            "test-owner/wan-test-data",
        )
        self.assertEqual(result["privacy"], "private")
        self.assertTrue(result["published"])
        self.assertTrue(result["kernel_metadata_updated"])

        metadata = self.read_metadata()

        self.assertIn(
            "test-owner/wan-test-data",
            metadata["dataset_sources"],
        )

        mock_run.assert_called_once()

    @patch("wan_kaggle_publisher.shutil.which")
    def test_rejects_missing_kaggle_cli(self, mock_which):
        mock_which.return_value = None

        with patch.dict(
            os.environ,
            {"KAGGLE_API_TOKEN": "test-token"},
        ):
            with self.assertRaises(WanKagglePublishError):
                publish_reference_dataset(
                    self.payload,
                    self.output,
                    owner="test-owner",
                    slug="wan-test-data",
                )

    @patch("wan_kaggle_publisher.shutil.which")
    def test_rejects_missing_authentication(self, mock_which):
        mock_which.return_value = "/usr/bin/kaggle"

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "wan_kaggle_publisher.Path.home",
                return_value=self.root / "no-home",
            ):
                with self.assertRaises(WanKagglePublishError):
                    publish_reference_dataset(
                        self.payload,
                        self.output,
                        owner="test-owner",
                        slug="wan-test-data",
                    )


if __name__ == "__main__":
    unittest.main()
