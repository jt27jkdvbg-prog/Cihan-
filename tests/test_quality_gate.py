
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quality_gate import (
    _parse_frame_rate,
    inspect_media,
    require_real_motion,
    validate_final_video,
    validate_motion_outputs,
)


def valid_metadata():
    return {
        "path": "/tmp/final-film.mp4",
        "file_size_bytes": 1024,
        "duration_seconds": 60.0,
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "video": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "fps": 24.0,
        },
        "audio": {
            "present": True,
            "codec": "aac",
            "sample_rate": "48000",
            "channels": 2,
        },
    }


class QualityGateTests(unittest.TestCase):

    def test_real_motion_requires_provider(self):
        with self.assertRaises(RuntimeError):
            require_real_motion("real", False)

    def test_real_motion_accepts_configured_provider(self):
        self.assertTrue(require_real_motion("real", True))

    def test_preview_mode_does_not_require_provider(self):
        self.assertFalse(require_real_motion("preview", False))

    def test_invalid_motion_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            require_real_motion("unknown", True)

    def test_validate_motion_outputs_finds_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "scene-1.mp4"
            second = Path(temporary) / "scene-2.mp4"
            first.write_bytes(b"video-data")
            second.write_bytes(b"video-data")

            result = validate_motion_outputs(
                [first, second],
                mode="real",
            )

            self.assertEqual(len(result), 2)

    def test_real_motion_rejects_missing_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "scene-1.mp4"
            missing = Path(temporary) / "scene-2.mp4"
            existing.write_bytes(b"video-data")

            with self.assertRaises(RuntimeError):
                validate_motion_outputs(
                    [existing, missing],
                    mode="real",
                )

    def test_preview_motion_allows_missing_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "scene-1.mp4"
            missing = Path(temporary) / "scene-2.mp4"
            existing.write_bytes(b"video-data")

            result = validate_motion_outputs(
                [existing, missing],
                mode="preview",
            )

            self.assertEqual(len(result), 1)

    def test_invalid_motion_validation_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_motion_outputs([], mode="invalid")

    def test_frame_rate_parser_supports_fraction(self):
        self.assertEqual(_parse_frame_rate("24/1"), 24.0)
        self.assertAlmostEqual(
            _parse_frame_rate("30000/1001"),
            29.97002997,
            places=5,
        )

    def test_frame_rate_parser_handles_invalid_values(self):
        for value in (None, "", "0/0", "N/A", "invalid"):
            self.assertEqual(_parse_frame_rate(value), 0.0)

    @patch("quality_gate.subprocess.run")
    @patch("quality_gate.shutil.which", return_value="/usr/bin/ffprobe")
    def test_inspect_media_reads_ffprobe_metadata(
        self,
        mock_which,
        mock_run,
    ):
        import json
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            media = Path(temporary) / "film.mp4"
            media.write_bytes(b"video-data")

            ffprobe_output = {
                "format": {
                    "duration": "60.0",
                    "format_name": "mp4",
                },
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1280,
                        "height": 720,
                        "avg_frame_rate": "24/1",
                    },
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                    },
                ],
            }

            mock_run.return_value = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(ffprobe_output),
                stderr="",
            )

            result = inspect_media(media)

            self.assertEqual(result["video"]["codec"], "h264")
            self.assertEqual(result["video"]["width"], 1280)
            self.assertEqual(result["video"]["height"], 720)
            self.assertEqual(result["video"]["fps"], 24.0)
            self.assertTrue(result["audio"]["present"])
            self.assertEqual(result["audio"]["codec"], "aac")
            mock_which.assert_called_once_with("ffprobe")

    def test_inspect_media_rejects_missing_file(self):
        with self.assertRaises(RuntimeError):
            inspect_media("/tmp/this-file-does-not-exist.mp4")

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_valid_final_video_passes(self, mock_inspect):
        report = validate_final_video(
            "final-film.mp4",
            require_audio=True,
            expected_duration=60,
        )

        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["errors"], [])
        mock_inspect.assert_called_once()

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_final_video_report_contains_limitations(self, mock_inspect):
        report = validate_final_video("final-film.mp4")

        self.assertIn("limitations", report)
        self.assertTrue(report["limitations"])
        self.assertIn("warnings", report)

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_video_without_audio_fails_when_audio_required(
        self,
        mock_inspect,
    ):
        metadata = valid_metadata()
        metadata["audio"]["present"] = False
        metadata["audio"]["codec"] = None
        mock_inspect.return_value = metadata

        with self.assertRaises(RuntimeError):
            validate_final_video(
                "final-film.mp4",
                require_audio=True,
            )

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_video_without_audio_can_pass_when_not_required(
        self,
        mock_inspect,
    ):
        metadata = valid_metadata()
        metadata["audio"]["present"] = False
        metadata["audio"]["codec"] = None
        mock_inspect.return_value = metadata

        report = validate_final_video(
            "silent-film.mp4",
            require_audio=False,
        )

        self.assertTrue(report["valid"])

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_low_resolution_fails(self, mock_inspect):
        metadata = valid_metadata()
        metadata["video"]["width"] = 320
        metadata["video"]["height"] = 240
        mock_inspect.return_value = metadata

        with self.assertRaises(RuntimeError):
            validate_final_video("low-resolution.mp4")

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_low_frame_rate_fails(self, mock_inspect):
        metadata = valid_metadata()
        metadata["video"]["fps"] = 5.0
        mock_inspect.return_value = metadata

        with self.assertRaises(RuntimeError):
            validate_final_video("low-fps.mp4")

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_duration_mismatch_fails(self, mock_inspect):
        with self.assertRaises(RuntimeError):
            validate_final_video(
                "wrong-duration.mp4",
                expected_duration=120,
                duration_tolerance=2,
            )

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_non_positive_expected_duration_is_rejected(
        self,
        mock_inspect,
    ):
        with self.assertRaises(ValueError):
            validate_final_video(
                "final-film.mp4",
                expected_duration=0,
            )

    @patch("quality_gate.inspect_media", return_value=valid_metadata())
    def test_custom_minimum_quality_thresholds(self, mock_inspect):
        report = validate_final_video(
            "final-film.mp4",
            min_width=1920,
            min_height=1080,
        ) if False else None

        metadata = valid_metadata()
        metadata["video"]["width"] = 1920
        metadata["video"]["height"] = 1080
        mock_inspect.return_value = metadata

        report = validate_final_video(
            "final-film.mp4",
            min_width=1920,
            min_height=1080,
        )

        self.assertTrue(report["valid"])


if __name__ == "__main__":
    unittest.main()
