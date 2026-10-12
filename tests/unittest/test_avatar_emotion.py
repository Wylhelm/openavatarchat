import os
import unittest

import numpy as np

from handlers.avatar.emotion.emotion_arkit import (
    EMOTION_MORPHS, LIP_SYNC_DAMPING, load_arkit_channels, build_channel_index, normalize_emotion,
    emotion_to_arkit, blend_emotion, ramp_levels,
)
from handlers.avatar.emotion.emotion_tag import EmotionTagStreamParser, strip_emotion_tags

CHANNELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "src", "handlers", "avatar", "lam",
                             "assets", "arkit_face_channels.txt")


def feed_all(parser, chunks):
    emitted = "".join(parser.feed(chunk)[0] for chunk in chunks)
    return emitted + parser.flush()[0]


class TestEmotionArkit(unittest.TestCase):
    def setUp(self):
        self.channels = load_arkit_channels(CHANNELS_PATH)
        self.index = build_channel_index(self.channels)

    def test_channels_cover_emotion_table(self):
        self.assertEqual(len(self.channels), 52)
        for morphs in EMOTION_MORPHS.values():
            for name in morphs:
                self.assertIn(name, self.index)

    def test_normalize_emotion(self):
        self.assertEqual(normalize_emotion("Joie"), "joie")
        self.assertEqual(normalize_emotion("colère"), "colere")
        self.assertEqual(normalize_emotion(" Réflexion "), "reflexion")
        self.assertEqual(normalize_emotion("happy"), "neutre")
        self.assertEqual(normalize_emotion(None), "neutre")

    def test_emotion_to_arkit_damps_mouth(self):
        vec = emotion_to_arkit("joie", self.index)
        self.assertEqual(vec.shape, (52,))
        self.assertAlmostEqual(vec[self.index["mouthSmileLeft"]], 0.7 * LIP_SYNC_DAMPING, places=5)
        self.assertAlmostEqual(vec[self.index["cheekSquintLeft"]], 0.5, places=5)
        self.assertFalse(np.any(emotion_to_arkit("neutre", self.index)))

    def test_blend_shapes_and_clamp(self):
        vec = emotion_to_arkit("surprise", self.index)
        base = np.full((10, 52), 0.9, dtype=np.float32)
        blended = blend_emotion(base, vec, 1.0)
        self.assertEqual(blended.shape, (10, 52))
        self.assertEqual(blended.dtype, np.float32)
        self.assertLessEqual(blended.max(), 1.0)
        self.assertEqual(blend_emotion(np.zeros((1, 52), np.float32), vec, 0.5).shape, (1, 52))

    def test_blend_per_frame_weight(self):
        vec = emotion_to_arkit("joie", self.index)
        weights = np.array([[0.0], [1.0]], dtype=np.float32)
        blended = blend_emotion(np.zeros((2, 52), np.float32), vec, weights)
        self.assertFalse(np.any(blended[0]))
        self.assertTrue(np.allclose(blended[1], vec))

    def test_ramp_up_carries_level(self):
        levels, level = ramp_levels(0.0, 5, ramp_frames=10)
        self.assertEqual(levels.shape, (5, 1))
        self.assertAlmostEqual(level, 0.5, places=5)
        levels, level = ramp_levels(level, 10, ramp_frames=10)
        self.assertAlmostEqual(level, 1.0, places=5)
        self.assertAlmostEqual(float(levels[-1, 0]), 1.0, places=5)

    def test_ramp_fade_out(self):
        levels, level = ramp_levels(1.0, 30, ramp_frames=9, fade_out=True)
        self.assertEqual(level, 0.0)
        self.assertAlmostEqual(float(levels[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(levels[-1, 0]), 0.0, places=5)


class TestEmotionTagParser(unittest.TestCase):
    def test_char_by_char_tag(self):
        parser = EmotionTagStreamParser()
        text = feed_all(parser, list("[joie] Bonjour Guillaume !"))
        self.assertEqual(text, "Bonjour Guillaume !")
        self.assertEqual(parser.emotion, "joie")

    def test_tag_in_one_chunk_with_accent(self):
        parser = EmotionTagStreamParser()
        text = feed_all(parser, ["  [Colère]", " ", "Non", " merci."])
        self.assertEqual(text, "Non merci.")
        self.assertEqual(parser.emotion, "colere")

    def test_no_tag_passes_through(self):
        parser = EmotionTagStreamParser()
        chunks = ["Il ", "est ", "14h."]
        self.assertEqual(parser.feed(chunks[0]), ("Il ", True))
        self.assertEqual(parser.emotion, "neutre")
        self.assertEqual(feed_all(parser, chunks[1:]), "est 14h.")

    def test_unknown_bracket_is_kept(self):
        parser = EmotionTagStreamParser()
        text = feed_all(parser, list("[note] texte"))
        self.assertEqual(text, "[note] texte")
        self.assertEqual(parser.emotion, "neutre")

    def test_unterminated_bracket_flushes(self):
        parser = EmotionTagStreamParser()
        text = feed_all(parser, list("[ceci est un long crochet sans fin"))
        self.assertEqual(text, "[ceci est un long crochet sans fin")

    def test_short_unterminated_released_on_flush(self):
        parser = EmotionTagStreamParser()
        self.assertEqual(feed_all(parser, ["[jo"]), "[jo")

    def test_strip_emotion_tags(self):
        self.assertEqual(strip_emotion_tags("[joie] Salut [surprise] oh [note] ok"), "Salut oh [note] ok")


if __name__ == "__main__":
    unittest.main()
