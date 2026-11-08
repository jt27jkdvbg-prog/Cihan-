
import unittest

import generate


class TestVideoPipeline(unittest.TestCase):

    def test_parse_json_with_surrounding_text(self):
        result = generate.parse_json(
            'Sonuç:\n{"title": "Deneme", "scenes": []}'
        )
        self.assertEqual(result["title"], "Deneme")

    def test_parse_json_rejects_invalid_content(self):
        with self.assertRaises(ValueError):
            generate.parse_json("JSON bulunamadı")

    def test_estimate_words_returns_timed_words(self):
        words = generate.estimate_words(
            "Merhaba dünya. Bu bir test!",
            3.0
        )
        self.assertEqual(len(words), 6)
        self.assertGreater(words[0][2], words[0][1])
        self.assertLessEqual(words[-1][2], 3.0)

    def test_caption_events_preserve_scene_offset(self):
        words = [
            ("Merhaba", 0.0, 0.2),
            ("dünya.", 0.25, 0.5),
            ("Bu", 0.6, 0.7),
            ("bir", 0.75, 0.8),
            ("test.", 0.9, 1.2),
        ]

        events = generate.caption_events(words, 5.0)

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0][2], "Merhaba dünya.")
        self.assertGreaterEqual(events[0][0], 5.0)

    def test_ass_time_format(self):
        self.assertEqual(
            generate.ass_time(0),
            "0:00:00.00"
        )
        self.assertEqual(
            generate.ass_time(3661.23),
            "1:01:01.23"
        )


if __name__ == "__main__":
    unittest.main()
