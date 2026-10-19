import io
import re
import threading
import time
import wave
from abc import ABC
from typing import Dict, List, Optional, Tuple, cast

import librosa
import numpy as np
import requests
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


_FRENCH_HINGE_PATTERN = re.compile(
    r"\s+(?=(?:mais|donc|car|parce\s+que|puisque|quand|lorsque|comme|pour|avec|sans|et|ou|afin\s+de)\b)",
    re.IGNORECASE | re.UNICODE,
)


def extract_tts_segments(buffer: str, first_chunk_sent: bool) -> Tuple[List[str], str]:
    """
    Adaptive two-speed segmentation for streaming LLM -> TTS:
    - First segment of a turn (first_chunk_sent=False): cuts on the earliest natural
      clause punctuation (>=3 words & >=18 chars, or terminal .!? with >=2 words),
      or on a French hinge conjunction once 8+ words have accumulated.
    - Subsequent segments (first_chunk_sent=True): cuts on sentence endings (.!?;:)
      or on commas after >=65 chars to preserve natural prosody.
    """
    segments: List[str] = []
    remaining = buffer
    is_first = not first_chunk_sent

    while remaining:
        if is_first:
            # 1) Look for first terminal punctuation (. ! ? 。 ！ ？) whose prefix is >= 6 chars
            term_match = None
            for m in re.finditer(r"([.!?。！？]+)\s+", remaining):
                if len(remaining[:m.end(1)].strip()) >= 6:
                    term_match = m
                    break
            # 2) Look for first clause punctuation (, ; : — …) whose prefix has >= 3 words and >= 18 chars
            clause_match = None
            for m in re.finditer(r"([,;:—…]+)\s+", remaining):
                cand_clause = remaining[:m.end(1)].strip()
                if len(cand_clause) >= 18 and len(cand_clause.split()) >= 3:
                    clause_match = m
                    break

            chosen_end = None
            if term_match and clause_match:
                if clause_match.start() < term_match.start():
                    chosen_end = (clause_match.end(1), clause_match.end())
                else:
                    chosen_end = (term_match.end(1), term_match.end())
            elif term_match:
                chosen_end = (term_match.end(1), term_match.end())
            elif clause_match:
                chosen_end = (clause_match.end(1), clause_match.end())

            if chosen_end is not None:
                cut_idx, next_idx = chosen_end
                seg = remaining[:cut_idx].strip()
                if seg:
                    segments.append(seg)
                    remaining = remaining[next_idx:]
                    is_first = False
                    continue

            # 3) Fallback for long unpunctuated opening clause: cut before a French hinge word after 6+ words
            words = remaining.split()
            if len(words) >= 9 and len(remaining) >= 42:
                hinge_cut = None
                for m in _FRENCH_HINGE_PATTERN.finditer(remaining):
                    prefix = remaining[:m.start()].strip()
                    suffix = remaining[m.end():].strip()
                    if len(prefix.split()) >= 6 and len(prefix) >= 32 and len(suffix.split()) >= 2:
                        hinge_cut = (m.start(), m.end())
                        break
                if hinge_cut is not None:
                    cut_idx, next_idx = hinge_cut
                    seg = remaining[:cut_idx].strip()
                    if seg:
                        segments.append(seg)
                        remaining = remaining[next_idx:]
                        is_first = False
                        continue
            break
        else:
            # Subsequent segments: prefer full sentences (. ! ? ; :), or comma if >= 65 chars
            sent_match = None
            for m in re.finditer(r"([.!?;:。！？]+)\s+", remaining):
                if len(remaining[:m.end(1)].strip()) >= 6:
                    sent_match = m
                    break
            if sent_match:
                cand = remaining[:sent_match.end(1)].strip()
                segments.append(cand)
                remaining = remaining[sent_match.end():]
                continue
            if len(remaining) >= 65:
                comma_match = None
                for m in re.finditer(r"(,)\s+", remaining):
                    if m.end(1) >= 55:
                        comma_match = m
                        break
                if comma_match is not None:
                    cand = remaining[:comma_match.end(1)].strip()
                    if cand:
                        segments.append(cand)
                        remaining = remaining[comma_match.end():]
                        continue
            break

    return segments, remaining


def _decode_wav_bytes(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Fast in-memory PCM WAV decoding via standard wave + numpy, avoiding librosa overhead when sr matches."""
    try:
        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            orig_sr = wf.getframerate()
            n_frames = wf.getnframes()
            pcm_bytes = wf.readframes(n_frames)
        if sampwidth == 2:
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            audio = (np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        if orig_sr != target_sr and audio.size > 0:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return audio[np.newaxis, ...]
    except Exception:
        audio_bytes = io.BytesIO(raw_bytes)
        output_audio, orig_sr = librosa.load(audio_bytes, sr=None)
        if orig_sr != target_sr:
            output_audio = librosa.resample(output_audio, orig_sr=orig_sr, target_sr=target_sr)
        return output_audio[np.newaxis, ...]


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    api_url: str = Field(default="http://10.0.0.175:8200")
    voice: str = Field(default="Chandra MacFarland")
    language: str = Field(default="fr")
    sample_rate: int = Field(default=24000)
    warmup_enabled: bool = Field(default=False)
    warmup_timeout: float = Field(default=15.0)


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        self.processed_sentences = set()
        self.first_chunk_sent = False
        self.current_speech_id: Optional[str] = None
        self.shared_states = None


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

        self.api_url = None
        self.voice = None
        self.language = None
        self.sample_rate = None
        self._thread_local = threading.local()

    def _get_http_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        config = cast(TTSConfig, handler_config)
        self.api_url = config.api_url
        self.voice = config.voice
        self.language = config.language
        self.sample_rate = config.sample_rate

        logger.info(f"CoquiLocal TTS handler loaded")
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Voice: {self.voice}")
        logger.info(f"Language: {self.language}")
        if config.warmup_enabled:
            started_at = time.monotonic()
            warmup_audio = self._synthesize(
                "Bonjour.",
                timeout=config.warmup_timeout,
            )
            elapsed = time.monotonic() - started_at
            if warmup_audio.shape[-1] > 240:
                logger.info(f"CoquiLocal warmup completed in {elapsed:.2f}s")
            else:
                logger.warning("CoquiLocal warmup did not return usable audio")

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.config = handler_config
        context.input_text = ''
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context, context: HandlerContext):
        pass

    def _remove_duplicate_endings(self, text):
        """Remove duplicate sentences, phrases, and words (aggressive cleanup for LLM looping)"""
        if not text or len(text.strip()) < 3:
            return text

        sentences = re.split(r'([.!?。！？])', text)
        cleaned_sentences = []
        seen_sentences = set()

        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
                sentence = (sentences[i] + sentences[i+1]).strip()
                i += 2
            else:
                sentence = sentences[i].strip()
                i += 1

            if not sentence:
                continue

            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())

            if normalized in seen_sentences:
                continue

            is_duplicate = False
            for seen in seen_sentences:
                if len(normalized) > 10 and len(seen) > 10:
                    if normalized in seen or seen in normalized:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_sentences.add(normalized)
                cleaned_sentences.append(sentence)

        if not cleaned_sentences:
            first_sentence = re.split(r'([.!?。！？])', text)[:2]
            return ''.join(first_sentence) if len(first_sentence) >= 2 else text.strip()[:100]

        result = ' '.join(cleaned_sentences)

        words = result.split()
        if len(words) > 1:
            cleaned_words = []
            for i, word in enumerate(words):
                if i == 0 or word != words[i-1]:
                    cleaned_words.append(word)

            if len(cleaned_words) >= 6:
                last_three = ' '.join(cleaned_words[-3:])
                if len(cleaned_words) >= 6:
                    prev_three = ' '.join(cleaned_words[-6:-3])
                    if last_three == prev_three:
                        cleaned_words = cleaned_words[:-3]

            result = ' '.join(cleaned_words)

        if text.strip() and text.strip()[-1] in '.!?。！？':
            if not result.endswith(text.strip()[-1]):
                result = result.rstrip('.!?。！？') + text.strip()[-1]

        return result.strip()

    def _synthesize(self, text: str, timeout: float = 60) -> np.ndarray:
        """Call the Coqui proxy and return float32 audio array shaped (1, samples)."""
        endpoint = f"{self.api_url}/v1/audio/speech"
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": self.voice,
            "speaker": self.voice,
            "language": self.language,
            "response_format": "wav",
        }
        try:
            response = self._get_http_session().post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"CoquiLocal TTS: connection error to {endpoint}: {e}")
            return np.zeros((1, 240), dtype=np.float32)
        except requests.exceptions.Timeout:
            logger.error(f"CoquiLocal TTS: request timed out to {endpoint}")
            return np.zeros((1, 240), dtype=np.float32)
        except requests.exceptions.HTTPError as e:
            logger.error(f"CoquiLocal TTS: HTTP error {e.response.status_code}: {e}")
            return np.zeros((1, 240), dtype=np.float32)
        except Exception as e:
            logger.error(f"CoquiLocal TTS: unexpected error: {e}", exc_info=True)
            return np.zeros((1, 240), dtype=np.float32)

        try:
            return _decode_wav_bytes(response.content, self.sample_rate)
        except Exception as e:
            logger.error(f"CoquiLocal TTS: failed to decode audio response: {e}", exc_info=True)
            return np.zeros((1, 240), dtype=np.float32)

    @staticmethod
    def _is_aborted(context: TTSContext, speech_id: str) -> bool:
        if context.shared_states is not None and hasattr(context.shared_states, "is_turn_aborted"):
            return context.shared_states.is_turn_aborted(speech_id=speech_id)
        return False

    def _process_sentence_chunk(self, clean_sentence, speech_id, output_definition, context: TTSContext):
        """Process a single sentence chunk for TTS"""
        if not clean_sentence or len(clean_sentence.strip()) < 1:
            return
        if self._is_aborted(context, speech_id):
            logger.info(f"Skipping TTS chunk for aborted speech_id={speech_id}")
            return

        normalized = re.sub(r'\s+', ' ', clean_sentence.lower().strip())

        if normalized in context.processed_sentences:
            logger.warning(f"Skipping duplicate sentence: '{clean_sentence[:50]}...'")
            return

        is_duplicate = False
        for processed in context.processed_sentences:
            if len(normalized) > 15 and len(processed) > 15:
                if normalized in processed or processed in normalized:
                    words_normalized = set(normalized.split())
                    words_processed = set(processed.split())
                    if len(words_normalized) > 0 and len(words_processed) > 0:
                        overlap = len(words_normalized & words_processed) / max(len(words_normalized), len(words_processed))
                        if overlap > 0.8:
                            is_duplicate = True
                            logger.warning(f"Skipping similar sentence (overlap: {overlap:.2f}): '{clean_sentence[:50]}...'")
                            break

        if is_duplicate:
            return

        context.processed_sentences.add(normalized)
        if len(context.processed_sentences) > 20:
            context.processed_sentences = set(list(context.processed_sentences)[-20:])

        is_first_chunk = not context.first_chunk_sent
        context.first_chunk_sent = True
        if is_first_chunk and context.shared_states is not None and hasattr(context.shared_states, "mark_llm_first_segment"):
            context.shared_states.mark_llm_first_segment(speech_id)

        logger.info(f'Processing sentence chunk ({len(clean_sentence)} chars, first={is_first_chunk}): {clean_sentence[:100]}...')
        try:
            t_tts_start = time.monotonic()
            logger.info(f"Calling CoquiLocal TTS - Language: {self.language}, Voice: {self.voice}, Text: '{clean_sentence[:100]}...'")

            output_audio = self._synthesize(clean_sentence)

            if self._is_aborted(context, speech_id):
                logger.info(f"Discarding synthesized TTS audio for aborted speech_id={speech_id}")
                return

            if is_first_chunk and context.shared_states is not None and hasattr(context.shared_states, "mark_tts_first_audio"):
                context.shared_states.mark_tts_first_audio(speech_id, t_start=t_tts_start)

            output = DataBundle(output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", False)
            output.add_meta("speech_id", speech_id)
            logger.info(f"CoquiLocal TTS generated audio: shape={output_audio.shape}, sample_rate={self.sample_rate}")
            context.submit_data(output)
        except Exception as e:
            logger.error(f"Error in CoquiLocal TTS generation for sentence '{clean_sentence}': {e}", exc_info=True)

    def filter_text(self, text):
        if not text:
            return text
        filtered_text = re.sub(r'[^\w\s,.\-!?;:\'"()àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ，。！？]', '', text, flags=re.UNICODE)
        return filtered_text

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if context.current_speech_id != speech_id:
            context.current_speech_id = speech_id
            context.input_text = ''
            context.first_chunk_sent = False
            context.processed_sentences.clear()

        if self._is_aborted(context, speech_id):
            context.input_text = ''
            return

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            filtered = self.filter_text(text)
            context.input_text += filtered

        text_end = inputs.data.get_meta("avatar_text_end", False)
        if not text_end:
            ready_segments, remaining = extract_tts_segments(context.input_text, context.first_chunk_sent)
            context.input_text = remaining
            for sentence in ready_segments:
                if self._is_aborted(context, speech_id):
                    context.input_text = ''
                    return
                if len(sentence.strip()) < 1:
                    continue
                sentence = re.sub(r'\s*([,.])\s*', r'\1 ', sentence)
                sentence = re.sub(r'\s*([:;!?])\s*', r' \1 ', sentence)
                sentence = re.sub(r'\s+', ' ', sentence)
                sentence = sentence.strip()
                sentence = self._remove_duplicate_endings(sentence)
                self._process_sentence_chunk(sentence, speech_id, output_definition, context)
        else:
            logger.info('last sentence: ' + context.input_text)
            if not self._is_aborted(context, speech_id) and context.input_text is not None and len(context.input_text.strip()) > 0:
                try:
                    clean_text = context.input_text.strip()
                    clean_text = re.sub(r'\s*([,.])\s*', r'\1 ', clean_text)
                    clean_text = re.sub(r'\s*([:;!?])\s*', r' \1 ', clean_text)
                    clean_text = re.sub(r'\s+', ' ', clean_text)
                    clean_text = clean_text.strip()
                    clean_text = self._remove_duplicate_endings(clean_text)

                    if clean_text and clean_text[-1] not in '.!?。！？':
                        clean_text += '.'

                    max_chunk_length = 4000
                    if len(clean_text) > max_chunk_length:
                        chunks = re.split(r'([,.;:!?])\s+', clean_text)
                        current_chunk = ""
                        for i in range(0, len(chunks), 2):
                            if i < len(chunks):
                                part = chunks[i] + (chunks[i+1] if i+1 < len(chunks) else "")
                                if len(current_chunk) + len(part) > max_chunk_length and current_chunk:
                                    self._process_sentence_chunk(current_chunk.strip(), speech_id, output_definition, context)
                                    current_chunk = part
                                else:
                                    current_chunk += part
                        if current_chunk.strip():
                            self._process_sentence_chunk(current_chunk.strip(), speech_id, output_definition, context)
                    else:
                        self._process_sentence_chunk(clean_text, speech_id, output_definition, context)
                except Exception as e:
                    logger.error(f"Error in CoquiLocal TTS generation for final sentence '{context.input_text}': {e}", exc_info=True)
            context.input_text = ''
            context.first_chunk_sent = False
            context.processed_sentences.clear()
            if self._is_aborted(context, speech_id):
                return
            output = DataBundle(output_definition)
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
            output.add_meta("avatar_speech_end", True)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
            logger.info(f"speech end")

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        logger.info('destroy context')
