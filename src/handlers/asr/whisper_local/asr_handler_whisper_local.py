
import io
import wave
from typing import Dict, Optional, cast
from loguru import logger
import numpy as np
import requests
from pydantic import BaseModel, Field
from abc import ABC
import os
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.contexts.session_context import SessionContext

from engine_utils.directory_info import DirectoryInfo
from engine_utils.general_slicer import SliceContext, slice_data


class ASRConfig(HandlerBaseConfigModel, BaseModel):
    api_url: str = Field(default="http://10.0.0.175:9400")
    language: str = Field(default="fr")
    task: str = Field(default="transcribe")


class ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )

        self.dump_audio = True
        self.audio_dump_file = None
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(),
                                          f"dump_talk_audio_{session_id}.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")
        self.shared_states = None


class HandlerASR(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.api_url = "http://10.0.0.175:9400"
        self.language = "fr"
        self.task = "transcribe"

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

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, ASRConfig):
            handler_config = ASRConfig()
        context = ASRContext(session_context.session_info.session_id)
        context.config = handler_config
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context, handler_context):
        pass

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
                logger.debug(f"No audio entry or invalid sample rate, defaulting to {input_sample_rate} Hz")
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if audio is not None:
            audio = audio.squeeze()

            if not hasattr(context, 'input_sample_rate'):
                context.input_sample_rate = input_sample_rate
                logger.info(f"Detected input audio sample rate: {input_sample_rate} Hz (audio shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'})")

            logger.info(f'audio in: shape={audio.shape if hasattr(audio, "shape") else "unknown"}, dtype={audio.dtype if hasattr(audio, "dtype") else "unknown"}')
            for audio_segment in slice_data(context.audio_slice_context, audio):
                if audio_segment is None or audio_segment.shape[0] == 0:
                    continue
                context.output_audios.append(audio_segment)

        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        # Flush remainder from slice context
        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate(
                    [remainder_audio,
                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))])
                context.output_audios.append(remainder_audio)
        output_audio = np.concatenate(context.output_audios)
        logger.info(f'Processing speech end: total audio length={len(output_audio)} samples, dtype={output_audio.dtype}')
        if context.audio_dump_file is not None:
            logger.info('dump audio')
            context.audio_dump_file.write(output_audio.tobytes())

        input_sample_rate = getattr(context, 'input_sample_rate', 16000)
        target_sample_rate = 16000
        logger.info(f'Using input_sample_rate={input_sample_rate} Hz, target_sample_rate={target_sample_rate} Hz')

        if len(output_audio) > 0:
            if output_audio.dtype == np.int16:
                audio_normalized = output_audio.astype(np.float32) / 32768.0
            elif output_audio.dtype == np.float32:
                if np.abs(output_audio).max() > 1.0:
                    audio_normalized = output_audio / 32768.0
                else:
                    audio_normalized = output_audio
            else:
                audio_normalized = output_audio.astype(np.float32)
                if np.abs(audio_normalized).max() > 1.0:
                    audio_normalized = audio_normalized / np.abs(audio_normalized).max()

            if input_sample_rate != target_sample_rate:
                try:
                    import librosa
                    logger.info(f"Resampling audio from {input_sample_rate} Hz to {target_sample_rate} Hz (before: {len(audio_normalized)} samples)")
                    audio_normalized = librosa.resample(
                        audio_normalized,
                        orig_sr=input_sample_rate,
                        target_sr=target_sample_rate
                    )
                    logger.info(f"Resampled audio: {len(audio_normalized)} samples")
                except ImportError:
                    logger.error("librosa not available for resampling! Install it with: pip install librosa")
                    logger.warning(f"Continuing without resampling (input={input_sample_rate}Hz, target={target_sample_rate}Hz)")
                except Exception as e:
                    logger.error(f"Error resampling audio: {e}", exc_info=True)
                    logger.warning(f"Continuing without resampling due to error")
        else:
            audio_normalized = np.array([], dtype=np.float32)

        output_text = ""
        if len(audio_normalized) > 0:
            min_duration_seconds = 0.5
            min_samples = int(target_sample_rate * min_duration_seconds)
            if len(audio_normalized) < min_samples:
                duration = len(audio_normalized) / target_sample_rate
                logger.warning(f"Audio too short ({len(audio_normalized)} samples = {duration:.2f}s at {target_sample_rate}Hz, need {min_duration_seconds}s), skipping ASR")
            else:
                audio_level = np.abs(audio_normalized).max()
                if audio_level < 0.01:
                    logger.warning(f"Audio too quiet (max level: {audio_level:.4f}), skipping ASR")
                else:
                    duration = len(audio_normalized) / target_sample_rate
                    logger.info(f"Sending {len(audio_normalized)} samples ({duration:.2f}s, level: {audio_level:.4f}) to WhisperLocal API")
                    output_text = self._transcribe_via_api(audio_normalized, target_sample_rate, context.config)
        else:
            logger.warning("Empty audio, skipping ASR")

        context.output_audios.clear()

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
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_bytes = wav_buffer.getvalue()

        url = f"{api_url}/asr?language={language}&task={task}&output=txt"
        try:
            response = requests.post(
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
        if hasattr(context, 'audio_dump_file') and context.audio_dump_file:
            context.audio_dump_file.close()
