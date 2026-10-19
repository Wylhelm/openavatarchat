
import io
import os
import threading
import time
import wave
from abc import ABC
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, cast

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
from engine_utils.directory_info import DirectoryInfo


def _normalize_audio_chunk(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return np.array([], dtype=np.float32)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.float32:
        max_val = float(np.abs(audio).max())
        return audio / 32768.0 if max_val > 1.0 else audio
    out = audio.astype(np.float32)
    max_val = float(np.abs(out).max())
    return out / max_val if max_val > 1.0 else out


def _merge_transcript_parts(parts: List[str]) -> str:
    """Join streaming ASR segment transcripts while removing boundary word duplication."""
    cleaned: List[str] = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    merged_words: List[str] = cleaned[0].split()
    for part in cleaned[1:]:
        words = part.split()
        if not words:
            continue
        # Remove 1-2 word overlap at chunk boundary
        overlap = 0
        for k in (2, 1):
            if len(merged_words) >= k and len(words) >= k:
                tail = [w.lower().strip(".,!?;:") for w in merged_words[-k:]]
                head = [w.lower().strip(".,!?;:") for w in words[:k]]
                if tail == head and any(tail):
                    overlap = k
                    break
        if overlap > 0 and merged_words[-1].endswith("."):
            merged_words[-1] = merged_words[-1].rstrip(".")
        merged_words.extend(words[overlap:])
    return " ".join(merged_words).strip()


class ASRConfig(HandlerBaseConfigModel, BaseModel):
    api_url: str = Field(default="http://10.0.0.175:9400")
    language: str = Field(default="fr")
    task: str = Field(default="transcribe")
    warmup_enabled: bool = Field(default=False)
    warmup_timeout: float = Field(default=15.0)
    dump_audio: bool = Field(default=False)
    streaming_asr_enabled: bool = Field(default=True)
    streaming_chunk_seconds: float = Field(default=2.5)
    streaming_max_chunk_seconds: float = Field(default=3.5)
    low_energy_window_ms: int = Field(default=160)
    low_energy_threshold: float = Field(default=0.035)


class ASRContext(HandlerContext):
    def __init__(self, session_id: str, dump_audio: bool = False):
        super().__init__(session_id)
        self.config: Optional[ASRConfig] = None
        self.local_session_id = 0
        self.output_audios: List[np.ndarray] = []
        self.pending_sample_count: int = 0
        self.current_speech_id: Optional[str] = None
        self.pre_transcription_futures: List[Future] = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"asr-{session_id[:6]}")

        self.dump_audio = dump_audio
        self.audio_dump_file = None
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(),
                                          f"dump_talk_audio_{session_id}.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")
        self.shared_states = None

    def reset_turn(self, speech_id: Optional[str] = None):
        self.current_speech_id = speech_id
        self.output_audios.clear()
        self.pending_sample_count = 0
        for fut in self.pre_transcription_futures:
            fut.cancel()
        self.pre_transcription_futures.clear()


class HandlerASR(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.api_url = "http://10.0.0.175:9400"
        self.language = "fr"
        self.task = "transcribe"
        self._thread_local = threading.local()

    def _get_http_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="ASR_WhisperLocal",
            config_model=ASRConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, ASRConfig):
            self.api_url = handler_config.api_url
            self.language = handler_config.language
            self.task = handler_config.task

        logger.info(f"WhisperLocal ASR handler ready: api_url='{self.api_url}', language='{self.language}', task='{self.task}'")
        if isinstance(handler_config, ASRConfig) and handler_config.warmup_enabled:
            self._warmup(handler_config)

    def _warmup(self, config: ASRConfig):
        """Run one silent inference so the first user request does not load the model."""
        audio_int16 = np.zeros(16000, dtype=np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        url = (
            f"{config.api_url}/asr?language={config.language}"
            f"&task={config.task}&output=txt"
        )
        started_at = time.monotonic()
        try:
            response = self._get_http_session().post(
                url,
                files={"audio_file": ("warmup.wav", wav_buffer.getvalue(), "audio/wav")},
                timeout=config.warmup_timeout,
            )
            response.raise_for_status()
            elapsed = time.monotonic() - started_at
            logger.info(f"WhisperLocal warmup completed in {elapsed:.2f}s")
        except Exception as e:
            logger.warning(f"WhisperLocal warmup failed; startup will continue: {e}")

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, ASRConfig):
            handler_config = ASRConfig()
        context = ASRContext(
            session_context.session_info.session_id,
            dump_audio=handler_config.dump_audio,
        )
        context.config = handler_config
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context, handler_context):
        pass

    @staticmethod
    def _should_flush_streaming_chunk(context: ASRContext, sample_rate: int) -> bool:
        cfg = context.config
        if cfg is None or not cfg.streaming_asr_enabled:
            return False
        duration = context.pending_sample_count / max(1, sample_rate)
        if duration < cfg.streaming_chunk_seconds:
            return False
        if duration >= cfg.streaming_max_chunk_seconds:
            return True
        if not context.output_audios:
            return False
        window_samples = max(256, int(sample_rate * (cfg.low_energy_window_ms / 1000.0)))
        last_chunk = context.output_audios[-1]
        tail = last_chunk[-window_samples:] if last_chunk.shape[0] >= window_samples else last_chunk
        if tail.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(tail))))
        return rms <= cfg.low_energy_threshold

    def _submit_streaming_pre_chunk(self, context: ASRContext, sample_rate: int):
        if not context.output_audios:
            return
        chunk_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()
        context.pending_sample_count = 0
        if chunk_audio.size == 0 or float(np.abs(chunk_audio).max()) < 0.01:
            return
        dur = len(chunk_audio) / max(1, sample_rate)
        logger.info(f"Streaming ASR: pre-transcribing {dur:.2f}s chunk in background for {context.current_speech_id}")
        fut = context.executor.submit(self._transcribe_via_api, chunk_audio, sample_rate, context.config)
        context.pre_transcription_futures.append(fut)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):

        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(ASRContext, context)
        if inputs.type == ChatDataType.HUMAN_AUDIO:
            audio = inputs.data.get_main_data()
            audio_entry = inputs.data.get_main_definition_entry()
            if audio_entry and audio_entry.sample_rate > 0:
                input_sample_rate = audio_entry.sample_rate
            else:
                input_sample_rate = 16000
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        speech_start = inputs.data.get_meta("human_speech_start", False)
        if speech_start or (context.current_speech_id is not None and speech_id != context.current_speech_id):
            context.reset_turn(speech_id)
        elif context.current_speech_id is None:
            context.current_speech_id = speech_id

        target_sample_rate = 16000
        if audio is not None:
            audio = audio.squeeze()
            if audio.ndim == 0:
                audio = audio.reshape(1)
            audio_norm = _normalize_audio_chunk(audio)
            if input_sample_rate != target_sample_rate and audio_norm.size > 0:
                try:
                    import librosa
                    audio_norm = librosa.resample(
                        audio_norm,
                        orig_sr=input_sample_rate,
                        target_sr=target_sample_rate,
                    )
                except Exception as e:
                    logger.error(f"Error resampling audio: {e}")

            if audio_norm.size > 0:
                context.output_audios.append(audio_norm)
                context.pending_sample_count += int(audio_norm.shape[0])

        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            if self._should_flush_streaming_chunk(context, target_sample_rate):
                self._submit_streaming_pre_chunk(context, target_sample_rate)
            return

        t_asr_start = time.monotonic()
        remainder_audio = (
            np.concatenate(context.output_audios)
            if context.output_audios
            else np.array([], dtype=np.float32)
        )
        context.output_audios.clear()
        context.pending_sample_count = 0

        trim_trailing = int(inputs.data.get_meta("trim_trailing_samples", 0) or 0)
        if trim_trailing > 0 and remainder_audio.size > trim_trailing + 4800:
            remainder_audio = remainder_audio[:-trim_trailing]

        if context.audio_dump_file is not None and remainder_audio.size > 0:
            context.audio_dump_file.write(remainder_audio.tobytes())

        pre_texts: List[str] = []
        num_pre_chunks = len(context.pre_transcription_futures)
        for fut in context.pre_transcription_futures:
            try:
                part_text = fut.result(timeout=15.0)
                if part_text:
                    pre_texts.append(part_text)
            except Exception as e:
                logger.warning(f"Streaming ASR pre-chunk failed: {e}")
        context.pre_transcription_futures.clear()

        tail_text = ""
        if remainder_audio.size > 0:
            # If we already have pre-transcribed chunks, allow shorter tail (0.20s); otherwise 0.35s
            min_duration_seconds = 0.20 if pre_texts else 0.35
            min_samples = int(target_sample_rate * min_duration_seconds)
            if len(remainder_audio) < min_samples:
                duration = len(remainder_audio) / target_sample_rate
                if not pre_texts:
                    logger.warning(
                        f"Audio too short ({len(remainder_audio)} samples = {duration:.2f}s, "
                        f"need {min_duration_seconds}s), skipping ASR"
                    )
            else:
                audio_level = float(np.abs(remainder_audio).max())
                if audio_level < 0.01:
                    if not pre_texts:
                        logger.warning(f"Audio too quiet (max level: {audio_level:.4f}), skipping ASR")
                else:
                    duration = len(remainder_audio) / target_sample_rate
                    logger.info(
                        f"Sending {len(remainder_audio)} samples ({duration:.2f}s, level: {audio_level:.4f}, "
                        f"pre_chunks={num_pre_chunks}) to WhisperLocal API"
                    )
                    tail_text = self._transcribe_via_api(remainder_audio, target_sample_rate, context.config)

        all_parts = pre_texts + ([tail_text] if tail_text else [])
        output_text = _merge_transcript_parts(all_parts)

        if context.shared_states and hasattr(context.shared_states, "mark_asr_done"):
            context.shared_states.mark_asr_done(
                speech_id,
                pre_chunks=num_pre_chunks,
                t_start=t_asr_start,
            )

        if len(output_text) == 0:
            if context.shared_states:
                context.shared_states.enable_vad = True
            return

        output = DataBundle(output_definition)
        output.set_main_data(output_text)
        output.add_meta('human_text_end', False)
        output.add_meta('speech_id', speech_id)
        yield output

        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("human_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def _transcribe_via_api(self, audio_normalized: np.ndarray, sample_rate: int,
                            config: Optional[ASRConfig]) -> str:
        cfg = config if isinstance(config, ASRConfig) else ASRConfig()
        api_url = cfg.api_url
        language = cfg.language
        task = cfg.task

        # Convert float32 [-1, 1] to int16 PCM for WAV encoding
        audio_int16 = np.clip(audio_normalized * 32767.0, -32768, 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_bytes = wav_buffer.getvalue()

        url = f"{api_url}/asr?language={language}&task={task}&output=txt"
        try:
            response = self._get_http_session().post(
                url,
                files={"audio_file": ("audio.wav", wav_bytes, "audio/wav")},
                timeout=30,
            )
            response.raise_for_status()
            text = response.text.strip()
            logger.info(f"WhisperLocal ASR result: '{text}'")
            return text
        except Exception as e:
            logger.error(f"WhisperLocal ASR connection error: {e}")
            return ""

    def destroy_context(self, context: HandlerContext):
        if hasattr(context, 'executor') and context.executor:
            context.executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(context, 'audio_dump_file') and context.audio_dump_file:
            context.audio_dump_file.close()
