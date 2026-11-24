
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audio_asset_mixer


class AudioAssetMixerTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.voice = self.root / "voice.wav"
        self.music = self.root / "music.wav"
        self.ambience = self.root / "ambience.wav"
        self.effect = self.root / "effect.wav"

        for path in (
            self.voice,
            self.music,
            self.ambience,
            self.effect,
        ):
            path.write_bytes(b"test-audio")

    def tearDown(self):
        self.temp.cleanup()

    def asset(self, path, **extra):
        return {
            "path": str(path),
            "license_verified": True,
            "volume": 0.2,
            **extra,
        }

    def test_duration_must_be_positive(self):
        for value in (0, -1, "invalid", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    audio_asset_mixer._duration(value)

    def test_unverified_asset_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "lisansı doğrulanmamış"
        ):
            audio_asset_mixer._asset(
                {
                    "path": str(self.music),
                    "license_verified": False,
                },
                "music",
            )

    def test_missing_asset_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            audio_asset_mixer._asset(
                {
                    "path": str(self.root / "missing.wav"),
                    "license_verified": True,
                },
                "music",
            )

    def test_invalid_volume_is_rejected(self):
        for volume in (-0.1, 1.1, "loud"):
            with self.subTest(volume=volume):
                with self.assertRaises(ValueError):
                    audio_asset_mixer._asset(
                        {
                            "path": str(self.music),
                            "license_verified": True,
                            "volume": volume,
                        },
                        "music",
                    )

    @patch.object(
        audio_asset_mixer.shutil,
        "which",
        return_value="/usr/bin/ffmpeg",
    )
    @patch.object(audio_asset_mixer.subprocess, "run")
    def test_builds_complete_audio_mix(
        self, run_mock, _which_mock
    ):
        output = self.root / "mixed.m4a"

        def create_output(*args, **kwargs):
            output.write_bytes(b"mixed-audio")

        run_mock.side_effect = create_output

        report = audio_asset_mixer.mix_audio_layers(
            voice_path=self.voice,
            output_path=output,
            duration=30,
            music=self.asset(self.music, volume=0.15),
            ambience=self.asset(self.ambience, volume=0.12),
            effects=[
                self.asset(
                    self.effect,
                    volume=0.4,
                    offset=7.5,
                )
            ],
        )

        self.assertTrue(output.is_file())
        self.assertEqual(report["duration_seconds"], 30.0)
        self.assertTrue(report["music_ducking"])

        self.assertEqual(
            report["layers"],
            ["dialogue", "music", "ambience", "effect_0"],
        )

        command = run_mock.call_args.args[0]

        self.assertIn("-filter_complex", command)

        filter_graph = command[
            command.index("-filter_complex") + 1
        ]

        self.assertIn("sidechaincompress", filter_graph)
        self.assertIn("adelay=7500|7500", filter_graph)
        self.assertIn("loudnorm", filter_graph)

    @patch.object(
        audio_asset_mixer.shutil,
        "which",
        return_value="/usr/bin/ffmpeg",
    )
    def test_effect_outside_film_duration_is_rejected(
        self, _which_mock
    ):
        with self.assertRaisesRegex(
            ValueError, "film süresinin dışında"
        ):
            audio_asset_mixer.mix_audio_layers(
                voice_path=self.voice,
                output_path=self.root / "mixed.m4a",
                duration=10,
                effects=[
                    self.asset(self.effect, offset=10)
                ],
            )

    @patch.object(
        audio_asset_mixer.shutil,
        "which",
        return_value=None,
    )
    def test_missing_ffmpeg_is_reported(self, _which_mock):
        with self.assertRaisesRegex(
            RuntimeError, "FFmpeg bulunamadı"
        ):
            audio_asset_mixer.mix_audio_layers(
                voice_path=self.voice,
                output_path=self.root / "mixed.m4a",
                duration=10,
            )


if __name__ == "__main__":
    unittest.main()
