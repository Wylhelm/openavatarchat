import os
import unittest

import numpy as np

from handlers.avatar.emotion.emotion_arkit import (
    EMOTION_MORPHS, LIP_SYNC_DAMPING, load_arkit_channels, build_channel_index, normalize_emotion,
    emotion_to_arkit, blend_emotion, ramp_levels,
)
from handlers.avatar.emotion.emotion_tag import EmotionTagStreamParser, strip_emotion_tags
from handlers.avatar.emotion.micro_expression import MicroExpressionState, micro_expression_offsets

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

    def test_intensity(self):
        self.assertEqual(normalize_emotion("joie:0.4"), "joie:0.40")
        self.assertEqual(normalize_emotion("Amusé:0,6"), "amuse:0.60")
        self.assertEqual(normalize_emotion("joie:2"), "joie")
        self.assertEqual(normalize_emotion("joie:abc"), "neutre")
        full = emotion_to_arkit("joie", self.index)
        self.assertTrue(np.allclose(emotion_to_arkit("joie:0.5", self.index), full * 0.5))

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

    def test_blend_suppresses_eye_wide_during_blink(self):
        vec = emotion_to_arkit("surprise", self.index)
        base = np.zeros((2, 52), dtype=np.float32)
        base[1, self.index["eyeBlinkLeft"]] = 0.95
        base[1, self.index["eyeBlinkRight"]] = 0.95
        blended = blend_emotion(base, vec, 1.0)
        wide_idx = self.index["eyeWideLeft"]
        self.assertGreater(blended[0, wide_idx], 0.5)
        self.assertLess(blended[1, wide_idx], 0.05)

    def test_soft_knee_compression_preserves_headroom(self):
        vec = emotion_to_arkit("surprise", self.index)
        base = np.full((2, 52), 0.7, dtype=np.float32)
        blended_low = blend_emotion(base, vec, 0.5)
        blended_high = blend_emotion(base, vec, 1.2)
        brow_idx = self.index["browInnerUp"]
        # Soft-knee keeps values strictly below 1.0 while preserving ordering
        self.assertLess(blended_high[0, brow_idx], 1.0)
        self.assertGreater(blended_high[0, brow_idx], blended_low[0, brow_idx])

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


class TestMicroExpression(unittest.TestCase):
    def setUp(self):
        self.index = build_channel_index(load_arkit_channels(CHANNELS_PATH))

    def test_offsets_shape_and_state(self):
        state = MicroExpressionState()
        frames = np.zeros((30, 52), dtype=np.float32)
        offsets = micro_expression_offsets(frames, self.index, state, 30)
        self.assertEqual(offsets.shape, (30, 52))
        self.assertEqual(state.frame, 30)
        self.assertTrue(np.all(offsets >= 0.0))
        self.assertLess(offsets.max(), 0.1)

    def test_emphasis_lifts_brows_on_open_jaw(self):
        frames = np.zeros((30, 52), dtype=np.float32)
        quiet = micro_expression_offsets(frames, self.index, MicroExpressionState(), 30)
        frames[:, self.index["jawOpen"]] = 0.7
        loud = micro_expression_offsets(frames, self.index, MicroExpressionState(), 30)
        brow = self.index["browInnerUp"]
        self.assertGreater(loud[-1, brow], quiet[-1, brow] + 0.1)

    def test_drift_continues_across_segments(self):
        frames = np.zeros((60, 52), dtype=np.float32)
        whole = micro_expression_offsets(frames, self.index, MicroExpressionState(), 30)
        state = MicroExpressionState()
        split = np.concatenate([micro_expression_offsets(frames[:25], self.index, state, 30),
                                micro_expression_offsets(frames[25:], self.index, state, 30)])
        self.assertTrue(np.allclose(whole, split))


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

    def test_tag_with_intensity(self):
        parser = EmotionTagStreamParser()
        text = feed_all(parser, list("[réflexion:0.75] Bonne question."))
        self.assertEqual(text, "Bonne question.")
        self.assertEqual(parser.emotion, "reflexion:0.75")

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
