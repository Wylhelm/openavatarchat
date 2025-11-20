
import re
from typing import Dict, Optional, cast
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field
from abc import ABC
import os
import torch
import whisper
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
    model_name: str = Field(default="medium")  # Whisper model: tiny, base, small, medium, large
    language: str = Field(default="fr")  # Force French language
    device: Optional[str] = Field(default=None)  # Auto-detect if None


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
        self.model_name = 'medium'
        self.language = 'fr'
        self.device = None
        self.model = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="ASR_Whisper",
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
            self.model_name = handler_config.model_name
            self.language = handler_config.language
            self.device = handler_config.device

        # Auto-detect device if not specified
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        logger.info(f"Loading Whisper model '{self.model_name}' on device '{self.device}' for language '{self.language}'")
        self.model = whisper.load_model(self.model_name, device=self.device)
        logger.info("Whisper model loaded successfully")

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
            # Get the actual sample rate from the audio entry
            # VAD typically outputs at 16kHz, but check the entry to be sure
            audio_entry = inputs.data.get_main_definition_entry()
            if audio_entry and audio_entry.sample_rate > 0:
                input_sample_rate = audio_entry.sample_rate
            else:
                # Default to 16kHz (VAD standard)
                input_sample_rate = 16000
                logger.debug(f"No audio entry or invalid sample rate, defaulting to {input_sample_rate} Hz")
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        if audio is not None:
            audio = audio.squeeze()
            
            # Store sample rate in context if not already set
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

        # prefill remainder audio in slice context
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

        # Get input sample rate (default to 16kHz if not detected)
        input_sample_rate = getattr(context, 'input_sample_rate', 16000)
        target_sample_rate = 16000  # Whisper expects 16kHz
        logger.info(f'Using input_sample_rate={input_sample_rate} Hz, target_sample_rate={target_sample_rate} Hz')

        # Whisper accepts audio in float32 format, normalized to [-1, 1]
        # First normalize, then resample, then check
        if len(output_audio) > 0:
            # Normalize to [-1, 1] range (assuming input is int16 or float32)
            if output_audio.dtype == np.int16:
                audio_normalized = output_audio.astype(np.float32) / 32768.0
            elif output_audio.dtype == np.float32:
                # Check if already normalized
                if np.abs(output_audio).max() > 1.0:
                    audio_normalized = output_audio / 32768.0
                else:
                    audio_normalized = output_audio
            else:
                audio_normalized = output_audio.astype(np.float32)
                if np.abs(audio_normalized).max() > 1.0:
                    audio_normalized = audio_normalized / np.abs(audio_normalized).max()
            
            # Resample to 16kHz if needed (do this BEFORE checking minimum length)
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
                    # Try to continue without resampling (might work if sample rates are compatible)
                    logger.warning(f"Continuing without resampling (input={input_sample_rate}Hz, target={target_sample_rate}Hz)")
                except Exception as e:
                    logger.error(f"Error resampling audio: {e}", exc_info=True)
                    # Try to continue without resampling
                    logger.warning(f"Continuing without resampling due to error")
        else:
            audio_normalized = np.array([], dtype=np.float32)

        # Generate ASR result with Whisper
        if len(audio_normalized) > 0:
            # Check minimum audio length (at least 0.5 seconds at target sample rate)
            min_duration_seconds = 0.5
            min_samples = int(target_sample_rate * min_duration_seconds)
            if len(audio_normalized) < min_samples:
                duration = len(audio_normalized) / target_sample_rate
                logger.warning(f"Audio too short ({len(audio_normalized)} samples = {duration:.2f}s at {target_sample_rate}Hz, need {min_duration_seconds}s), skipping Whisper ASR")
                output_text = ""
            else:
                # Check audio level (should have some signal)
                audio_level = np.abs(audio_normalized).max()
                if audio_level < 0.01:  # Very quiet audio
                    logger.warning(f"Audio too quiet (max level: {audio_level:.4f}), skipping Whisper ASR")
                    output_text = ""
                else:
                    duration = len(audio_normalized) / target_sample_rate
                    logger.info(f"Running Whisper ASR on {len(audio_normalized)} samples (duration: {duration:.2f}s, level: {audio_level:.4f})")
                    try:
                        result = self.model.transcribe(
                            audio_normalized,
                            language=context.config.language if context.config else self.language,
                            task="transcribe",
                            fp16=False if self.device == "cpu" else True
                        )
                        output_text = result["text"].strip()
                        detected_lang = result.get('language', 'unknown')
                        logger.info(f"Whisper ASR result: '{output_text}' (detected language: {detected_lang})")
                    except Exception as e:
                        logger.error(f"Error in Whisper transcribe: {e}", exc_info=True)
                        output_text = ""
        else:
            logger.warning("Empty audio, skipping Whisper ASR")
            output_text = ""

        context.output_audios.clear()
        
        if len(output_text) == 0:
            # If ASR result is empty, re-enable VAD
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

    def destroy_context(self, context: HandlerContext):
        if hasattr(context, 'audio_dump_file') and context.audio_dump_file:
            context.audio_dump_file.close()

