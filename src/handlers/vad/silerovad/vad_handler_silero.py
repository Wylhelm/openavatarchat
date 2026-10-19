import enum
import math
import os
from abc import ABC
from typing import cast, Dict, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerDetail, HandlerDataInfo, HandlerBaseInfo
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.general_slicer import SliceContext, slice_data


class SileroVADConfigModel(HandlerBaseConfigModel, BaseModel):
    speaking_threshold: float = Field(default=0.5)
    start_delay: int = Field(default=2048)
    end_delay: int = Field(default=5000)
    buffer_look_back: int = Field(default=1024)
    speech_padding: int = Field(default=512)
    barge_in_enabled: bool = Field(default=False)
    barge_in_threshold: float = Field(default=0.40)
    barge_in_start_delay: int = Field(default=1536)
    barge_in_min_rms: float = Field(default=0.005)
    prewarm_delay: int = Field(default=8000)


class SpeakingStatus(enum.Enum):
    PRE_START = enum.auto()
    START = enum.auto()
    END = enum.auto()


class HumanAudioVADContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: SileroVADConfigModel = SileroVADConfigModel()
        self.speaking_status = SpeakingStatus.END

        self.clip_size = 512

        self.audio_history = []
        self.history_length_limit = 0

        self.speech_length: int = 0
        self.silence_length: int = 0
        self.total_speech_samples: int = 0
        self.prewarm_sent: bool = False

        self.shared_states = None

        self.model_state: Optional[np.ndarray] = None
        self.slice_context: Optional[SliceContext] = None

        self.speech_id: int = 0

    def reset(self):
        self.audio_history.clear()
        self.speech_length = 0
        self.silence_length = 0
        self.total_speech_samples = 0
        self.prewarm_sent = False
        if self.slice_context is not None:
            self.slice_context.flush()

    def _is_avatar_active(self) -> bool:
        if self.shared_states is None:
            return False
        return bool(
            getattr(self.shared_states, "avatar_speaking", False)
            or getattr(self.shared_states, "pipeline_generating", False)
        )

    def _active_start_params(self) -> Tuple[float, int]:
        if self.config.barge_in_enabled and self._is_avatar_active() and self.speaking_status != SpeakingStatus.START:
            return (
                max(self.config.speaking_threshold, self.config.barge_in_threshold),
                self.config.barge_in_start_delay,
            )
        return self.config.speaking_threshold, self.config.start_delay

    def _update_status_on_pre_start(self, clip: np.ndarray, _timestamp: Optional[int] = None):
        _, required_start_delay = self._active_start_params()
        if self.speech_length >= required_start_delay:
            head_sample_id = None
            self.speaking_status = SpeakingStatus.START
            self.total_speech_samples = self.speech_length
            self.prewarm_sent = False
            sample_num_to_fetch = self.config.buffer_look_back + required_start_delay + self.silence_length
            slice_num_to_fetch = math.ceil(sample_num_to_fetch / self.clip_size)
            audio_clips = []
            for history_entry in self.audio_history[-slice_num_to_fetch:]:
                history_clip, history_timestamp = history_entry
                if head_sample_id is None:
                    head_sample_id = history_timestamp
                audio_clips.append(history_clip)
            output_audio = np.concatenate(audio_clips, axis=0)
            output_audio = np.concatenate(
                [np.zeros(self.config.speech_padding, dtype=clip.dtype), output_audio], axis=0)
            self.speech_id += 1
            logger.info("Start of human speech")
            extra_args = {
                "human_speech_start": True,
                "pre_padding": self.config.speech_padding,
            }
            if head_sample_id is not None:
                extra_args["head_sample_id"] = head_sample_id
                logger.info(f"VAD pre_start to start got timestamp {head_sample_id}")
            return output_audio, extra_args
        else:
            # Allow up to 3 frames (~96ms) of consonant/diphthong dip during PRE_START
            # so natural speech does not immediately abort after the first 32ms frame.
            if self.silence_length > self.clip_size * 3:
                logger.info("Back to not started status")
                self.speaking_status = SpeakingStatus.END
                self.speech_length = 0
            return None, {}

    def _update_status_on_start(self, clip: np.ndarray, timestamp: Optional[int] = None):
        if self.silence_length >= self.config.end_delay:
            self.speaking_status = SpeakingStatus.END
            output_audio = np.concatenate(
                [clip, np.zeros(self.config.speech_padding, dtype=clip.dtype)], axis=0)
            logger.info("End of human speech")
            extra_args = {
                "human_speech_end": True,
                "post_padding": self.config.speech_padding,
            }
            if timestamp is not None:
                extra_args["head_sample_id"] = timestamp
                logger.info(f"VAD start to start got timestamp {timestamp}")
            return output_audio, extra_args
        else:
            extra_args = {"head_sample_id": timestamp}
            if not self.prewarm_sent and self.total_speech_samples >= self.config.prewarm_delay:
                self.prewarm_sent = True
                extra_args["speech_prewarm"] = True
            return clip, extra_args

    def _update_status_on_end(self, _clip: np.ndarray, _timestamp: Optional[int] = None):
        if self.speech_length > 0:
            logger.info("Pre start of new human speech")
            self.speaking_status = SpeakingStatus.PRE_START
        return None, {}

    def _append_to_history(self, clip: np.ndarray, timestamp: Optional[int] = None):
        self.audio_history.append((clip, timestamp))
        while 0 < self.history_length_limit < len(self.audio_history):
            self.audio_history.pop(0)

    def update_status(self, speech_prob: float, clip: np.ndarray,
                      timestamp: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict]:
        self._append_to_history(clip, timestamp)
        threshold, _ = self._active_start_params()
        is_voice = speech_prob > threshold
        if (
            self.config.barge_in_enabled
            and is_voice
            and self.speaking_status != SpeakingStatus.START
            and self._is_avatar_active()
        ):
            rms = float(np.sqrt(np.mean(np.square(clip)))) if clip.size > 0 else 0.0
            if rms < self.config.barge_in_min_rms:
                is_voice = False

        if is_voice:
            self.speech_length += self.clip_size
            self.total_speech_samples += self.clip_size
            self.silence_length = 0
        else:
            self.silence_length += self.clip_size
            if self.speaking_status != SpeakingStatus.PRE_START:
                self.speech_length = 0
        if self.speaking_status == SpeakingStatus.PRE_START:
            return self._update_status_on_pre_start(clip, timestamp)
        elif self.speaking_status == SpeakingStatus.START:
            return self._update_status_on_start(clip, timestamp)
        elif self.speaking_status == SpeakingStatus.END:
            return self._update_status_on_end(clip, timestamp)


class HandlerAudioVAD(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.model = None

    def get_handler_info(self):
        return HandlerBaseInfo(
            config_model=SileroVADConfigModel
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config = None):
        import onnxruntime
        model_name = "silero_vad.onnx"
        model_path = os.path.join(self.handler_root, "silero_vad",
                                  "src", "silero_vad", "data",
                                  model_name)
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self.model = onnxruntime.InferenceSession(model_path,
                                                  providers=["CPUExecutionProvider"],
                                                  sess_options=options)

    def create_context(self, session_context: SessionContext, handler_config = None) -> HandlerContext:
        context = HumanAudioVADContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if isinstance(handler_config, SileroVADConfigModel):
            context.config = handler_config
        context.model_state = np.zeros((2, 1, 128), dtype=np.float32)
        context.slice_context = SliceContext.create_numpy_slice_context(
            slice_size=context.clip_size,
            slice_axis=0,
        )
        max_start = max(context.config.start_delay, context.config.barge_in_start_delay)
        context.history_length_limit = math.ceil((max_start + context.config.buffer_look_back)
                                                 / context.clip_size)
        return context

    def start_context(self, session_context, handler_context):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("human_audio", 1, 16000))

        inputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO
            )
        }
        outputs = {ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
                definition=definition
            )
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )

    def _inference(self, context: HumanAudioVADContext, clip: np.ndarray, sr: int=16000):
        clip = clip.squeeze()
        if clip.ndim != 1:
            logger.warning("Input audio should be 1-dim array")
            return 0
        clip = np.expand_dims(clip, axis=0)
        inputs = {
            "input": clip,
            "sr": np.array([sr], dtype=np.int64),
            "state": context.model_state
        }
        prob, state = self.model.run(None, inputs)
        context.model_state = state
        return prob[0][0]

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(HumanAudioVADContext, context)
        output_definition = output_definitions.get(ChatDataType.HUMAN_AUDIO).definition
        if not context.shared_states.enable_vad:
            return
        if inputs.type != ChatDataType.MIC_AUDIO:
            return

        audio = inputs.data.get_main_data()
        if audio is None:
            return
        audio_entry = inputs.data.get_main_definition_entry()
        sample_rate = audio_entry.sample_rate
        audio = audio.squeeze()

        timestamp = None
        if inputs.is_timestamp_valid():
            timestamp = inputs.timestamp

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32767

        context.slice_context.update_start_id(timestamp[0], force_update=False)

        for clip in slice_data(context.slice_context, audio):
            head_sample_id = context.slice_context.get_last_slice_start_index()
            speech_prob = self._inference(context, clip)
            audio_clip, extra_args = context.update_status(speech_prob, clip, timestamp=head_sample_id)
            human_speech_start = extra_args.get("human_speech_start", False)
            human_speech_end = extra_args.get("human_speech_end", False)
            speech_prewarm = extra_args.get("speech_prewarm", False)
            timestamp = extra_args.get("head_sample_id", head_sample_id)
            speech_id = f"speech-{context.session_id}-{context.speech_id}"
            if human_speech_start and context.shared_states is not None:
                if context.config.barge_in_enabled and context._is_avatar_active():
                    logger.info(f"Barge-in detected on {speech_id}: interrupting avatar speech/generation")
                    context.shared_states.trigger_barge_in(new_speech_id=speech_id)
                    extra_args["barge_in"] = True
                context.shared_states.human_speaking = True
                if hasattr(context.shared_states, "mark_speech_start"):
                    context.shared_states.mark_speech_start(speech_id)
            if speech_prewarm and context.shared_states is not None:
                if hasattr(context.shared_states, "mark_speech_prewarm"):
                    context.shared_states.mark_speech_prewarm(speech_id)
            if human_speech_end:
                if context.shared_states is not None:
                    context.shared_states.human_speaking = False
                    if hasattr(context.shared_states, "mark_vad_end"):
                        vad_tail_ms = (context.config.end_delay / max(1, sample_rate)) * 1000.0
                        context.shared_states.mark_vad_end(speech_id, vad_end_delay_ms=vad_tail_ms)
                    if not context.config.barge_in_enabled:
                        context.shared_states.enable_vad = False
                context.reset()
            if audio_clip is not None:
                output = DataBundle(output_definition)
                output.set_main_data(np.expand_dims(audio_clip, axis=0))
                for flag_name, flag_value in extra_args.items():
                    output.add_meta(flag_name, flag_value)
                output.add_meta("speech_id", speech_id)
                output_chat_data = ChatData(
                    type=ChatDataType.HUMAN_AUDIO,
                    data=output
                )
                if timestamp >= 0:
                    output_chat_data.timestamp = timestamp, sample_rate
                yield output_chat_data

    def destroy_context(self, context: HandlerContext):
        pass
