import io
import sys
import types
import unittest
import wave

import numpy as np

# Provide lightweight stubs when running outside the container venv
if "loguru" not in sys.modules:
    try:
        import loguru  # noqa: F401
    except ImportError:
        loguru_stub = types.ModuleType("loguru")

        class _DummyLogger:
            def info(self, *a, **kw): pass
            def warning(self, *a, **kw): pass
            def error(self, *a, **kw): pass
            def debug(self, *a, **kw): pass
            def opt(self, *a, **kw): return self

        loguru_stub.logger = _DummyLogger()
        sys.modules["loguru"] = loguru_stub

if "librosa" not in sys.modules:
    try:
        import librosa  # noqa: F401
    except ImportError:
        librosa_stub = types.ModuleType("librosa")
        librosa_stub.resample = lambda y, orig_sr, target_sr: y
        sys.modules["librosa"] = librosa_stub

from chat_engine.contexts.session_context import SharedStates, TurnLatencyMetrics
from handlers.asr.whisper_local.asr_handler_whisper_local import (
    ASRConfig,
    ASRContext,
    HandlerASR,
    _clean_whisper_hallucinations,
    _merge_transcript_parts,
    _normalize_audio_chunk,
)
from handlers.avatar.emotion.emotion_tag import EmotionTagStreamParser
from handlers.tts.coqui_local.tts_handler_coqui_local import (
    _decode_wav_bytes,
    extract_tts_segments,
)
from handlers.vad.silerovad.vad_handler_silero import (
    HumanAudioVADContext,
    SileroVADConfigModel,
    SpeakingStatus,
)


class TestTurnLatencyTracker(unittest.TestCase):
    def test_latency_summary_formatting(self):
        states = SharedStates()
        sid = "speech-test-1"
        states.mark_speech_start(sid, now=10.0)
        states.mark_speech_prewarm(sid, now=10.5)
        states.mark_vad_end(sid, vad_end_delay_ms=480.0, now=12.0)
        states.mark_asr_done(sid, pre_chunks=1, t_start=12.0, now=12.42)
        states.mark_llm_first_token(sid, t_start=12.42, now=12.70)
        states.mark_llm_first_segment(sid, now=12.83)
        states.mark_tts_first_audio(sid, t_start=12.83, now=13.19)
        states.mark_lam_first_motion(sid, now=13.21)
        summary = states.mark_ws_first_send(sid, now=13.22)

        self.assertIsNotNone(summary)
        self.assertIn("[LATENCY speech-test-1]", summary)
        self.assertIn("Post-VAD: 1220ms", summary)
        self.assertIn("Total syllable->audio: 1850ms", summary)
        self.assertIn("ASR=420ms (pre=1)", summary)
        self.assertIn("LLM_1st_tok=280ms", summary)
        self.assertIn("TTS_1st=360ms", summary)
        # Calling mark_ws_first_send a second time for the same turn should not re-log
        self.assertIsNone(states.mark_ws_first_send(sid, now=13.50))

    def test_barge_in_aborts_active_turn(self):
        states = SharedStates()
        states.active_speech_id = "speech-old-1"
        states.avatar_speaking = True
        states.pipeline_generating = True
        gen_before = states.turn_generation_id
        callback_called = []
        states.interrupt_callback = lambda: callback_called.append(True)

        states.trigger_barge_in(new_speech_id="speech-new-2")
        self.assertTrue(states.is_turn_aborted(speech_id="speech-old-1"))
        self.assertTrue(states.is_turn_aborted(generation_id=gen_before))
        self.assertFalse(states.is_turn_aborted(speech_id="speech-new-2", generation_id=states.turn_generation_id))
        self.assertFalse(states.avatar_speaking)
        self.assertFalse(states.pipeline_generating)
        self.assertEqual(len(callback_called), 1)


class TestTTSFirstClauseSegmentation(unittest.TestCase):
    def test_first_clause_splits_on_comma(self):
        text = "Bien sûr Guillaume, je vérifie tout de suite dans tes notes."
        segments, rem = extract_tts_segments(text, first_chunk_sent=False)
        self.assertEqual(segments, ["Bien sûr Guillaume,"])
        self.assertEqual(rem, "je vérifie tout de suite dans tes notes.")

    def test_first_clause_splits_on_short_sentence(self):
        text = "Avec plaisir ! Je lance la recherche maintenant"
        segments, rem = extract_tts_segments(text, first_chunk_sent=False)
        self.assertEqual(segments, ["Avec plaisir !"])
        self.assertEqual(rem, "Je lance la recherche maintenant")

    def test_first_clause_skips_4char_interjection_to_cut_on_next_punctuation(self):
        text = "Ah ! Pi Dog, sur Raspberry Pi. C'est ça. Une base vraiment accessible pour commencer"
        segments, rem = extract_tts_segments(text, first_chunk_sent=False)
        self.assertEqual(segments[0], "Ah ! Pi Dog, sur Raspberry Pi.")
        self.assertEqual(segments[1], "C'est ça.")
        self.assertEqual(rem, "Une base vraiment accessible pour commencer")

    def test_first_clause_splits_on_french_hinge_when_unpunctuated(self):
        text = "Je viens de vérifier l'état du serveur principal mais la base de données"
        segments, rem = extract_tts_segments(text, first_chunk_sent=False)
        self.assertEqual(segments, ["Je viens de vérifier l'état du serveur principal"])
        self.assertEqual(rem, "mais la base de données")

    def test_subsequent_segments_keep_sentences_together(self):
        text = "Oui, c'est prêt. Et voici la suite."
        segments, rem = extract_tts_segments(text, first_chunk_sent=True)
        self.assertEqual(segments, ["Oui, c'est prêt."])
        self.assertEqual(rem, "Et voici la suite.")

    def test_long_subsequent_sentence_splits_into_sub_clauses(self):
        text = (
            "à 6h55, le robot quitte sa base dans le salon, suit son plan de navigation "
            "jusqu'à ta chambre, s'arrête au pied du lit et commence à te parler en douceur. "
            "Il ajuste ensuite son volume."
        )
        segments, rem = extract_tts_segments(text, first_chunk_sent=True)
        self.assertGreaterEqual(len(segments), 2)
        for seg in segments:
            self.assertLessEqual(len(seg), 150)
        self.assertEqual(rem, "Il ajuste ensuite son volume.")

    def test_short_introductory_colon_does_not_create_tiny_segment(self):
        text = "Enfin, la discrétion : un robot qui roule la nuit doit être silencieux. Suite."
        segments, rem = extract_tts_segments(text, first_chunk_sent=True)
        self.assertTrue(segments[0].startswith("Enfin, la discrétion : un robot"))
        self.assertEqual(rem, "Suite.")

    def test_fast_decode_wav_bytes(self):
        sr = 24000
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False, dtype=np.float32)
        sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sine.tobytes())
        decoded = _decode_wav_bytes(buf.getvalue(), sr)
        self.assertEqual(decoded.shape, (1, len(sine)))
        self.assertEqual(decoded.dtype, np.float32)
        self.assertAlmostEqual(float(np.abs(decoded).max()), 16000 / 32768.0, places=3)


class TestStreamingASRHelpers(unittest.TestCase):
    def test_merge_transcript_parts_deduplicates_boundary_overlap(self):
        parts = [
            "Bonjour Guillaume comment ça",
            "comment ça va aujourd'hui ?",
        ]
        merged = _merge_transcript_parts(parts)
        self.assertEqual(merged, "Bonjour Guillaume comment ça va aujourd'hui ?")

    def test_should_flush_streaming_chunk_on_low_energy_tail(self):
        ctx = ASRContext("sess1", dump_audio=False)
        ctx.config = ASRConfig(
            streaming_asr_enabled=True,
            streaming_chunk_seconds=2.5,
            streaming_max_chunk_seconds=3.5,
            low_energy_window_ms=160,
            low_energy_threshold=0.035,
        )
        # 2.6 seconds of audio with a quiet tail
        loud = np.full(int(16000 * 2.4), 0.25, dtype=np.float32)
        quiet_tail = np.full(int(16000 * 0.2), 0.005, dtype=np.float32)
        ctx.output_audios = [loud, quiet_tail]
        ctx.pending_sample_count = len(loud) + len(quiet_tail)
        self.assertTrue(HandlerASR._should_flush_streaming_chunk(ctx, 16000))
        ctx.executor.shutdown(wait=False)

    def test_normalize_audio_chunk(self):
        pcm = np.array([0, 16384, -16384], dtype=np.int16)
        norm = _normalize_audio_chunk(pcm)
        self.assertAlmostEqual(float(norm[1]), 0.5, places=4)

    def test_clean_whisper_hallucinations(self):
        self.assertEqual(
            _clean_whisper_hallucinations("Sous-titrage Société Radio-Canada", is_short_tail=True),
            "",
        )
        self.assertEqual(
            _clean_whisper_hallucinations("Merci.", is_short_tail=True),
            "",
        )
        self.assertEqual(
            _clean_whisper_hallucinations("Merci.", is_short_tail=False),
            "Merci.",
        )
        self.assertEqual(
            _clean_whisper_hallucinations("...", is_short_tail=False),
            "",
        )


class TestVADBargeInAndPrewarm(unittest.TestCase):
    def test_prewarm_and_barge_in_thresholds(self):
        ctx = HumanAudioVADContext("sess1")
        ctx.config = SileroVADConfigModel(
            speaking_threshold=0.3,
            start_delay=1024,
            end_delay=2048,
            barge_in_enabled=True,
            barge_in_threshold=0.60,
            barge_in_start_delay=1536,
            barge_in_min_rms=0.02,
            prewarm_delay=2048,
        )
        states = SharedStates()
        states.avatar_speaking = True
        ctx.shared_states = states

        clip_quiet = np.full(512, 0.005, dtype=np.float32)
        # Quiet echo with high prob should be ignored while avatar is speaking
        ctx.update_status(0.75, clip_quiet, timestamp=0)
        self.assertEqual(ctx.speaking_status, SpeakingStatus.END)

        # Confident user voice with sufficient RMS should transition PRE_START -> START -> prewarm
        clip_loud = np.full(512, 0.15, dtype=np.float32)
        ctx.update_status(0.75, clip_loud, timestamp=512)
        self.assertEqual(ctx.speaking_status, SpeakingStatus.PRE_START)
        ctx.update_status(0.75, clip_loud, timestamp=1024)
        audio_out, extra = ctx.update_status(0.75, clip_loud, timestamp=1536)
        self.assertIsNotNone(audio_out)
        self.assertTrue(extra.get("human_speech_start"))

        # Next clip reaches prewarm_delay (2048 samples)
        _, extra2 = ctx.update_status(0.75, clip_loud, timestamp=2048)
        self.assertTrue(extra2.get("speech_prewarm"))

    def test_barge_in_disabled_uses_normal_threshold(self):
        ctx = HumanAudioVADContext("sess2")
        ctx.config = SileroVADConfigModel(
            speaking_threshold=0.3,
            start_delay=1024,
            end_delay=2048,
            barge_in_enabled=False,
            barge_in_threshold=0.72,
            barge_in_start_delay=3584,
            barge_in_min_rms=0.035,
        )
        states = SharedStates()
        states.avatar_speaking = True
        ctx.shared_states = states

        clip = np.full(512, 0.01, dtype=np.float32)
        ctx.update_status(0.40, clip, timestamp=0)
        self.assertEqual(ctx.speaking_status, SpeakingStatus.PRE_START)
        audio_out, extra = ctx.update_status(0.40, clip, timestamp=512)
        self.assertIsNotNone(audio_out)
        self.assertTrue(extra.get("human_speech_start"))


class TestEmotionTagStreamParserRecovery(unittest.TestCase):
    def test_restarted_tag_after_tool_call(self):
        parser = EmotionTagStreamParser()
        out1, res1 = parser.feed("[sur")
        self.assertEqual(out1, "")
        self.assertFalse(res1)
        out2, res2 = parser.feed("[sceptique] Ah, je vois !")
        self.assertTrue(res2)
        self.assertEqual(parser.emotion, "sceptique")
        self.assertEqual(out2, "Ah, je vois !")


if __name__ == "__main__":
    unittest.main()
