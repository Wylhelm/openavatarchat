import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from chat_engine.data_models.chat_engine_config_data import EngineChannelType
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry
from chat_engine.data_models.session_info_data import SessionInfoData, IOQueueType


MAX_TRACKED_AVATAR_EMOTIONS = 8
MAX_TRACKED_LATENCY_TURNS = 12


@dataclass
class TurnLatencyMetrics:
    speech_id: str
    t_speech_start: float = 0.0
    t_speech_prewarm: float = 0.0
    t_vad_end: float = 0.0
    vad_end_delay_ms: float = 480.0
    t_asr_start: float = 0.0
    t_asr_done: float = 0.0
    asr_pre_chunks: int = 0
    t_llm_start: float = 0.0
    t_llm_first_token: float = 0.0
    t_llm_first_segment: float = 0.0
    t_tts_first_start: float = 0.0
    t_tts_first_audio: float = 0.0
    t_lam_first_motion: float = 0.0
    t_ws_first_send: float = 0.0
    player_prebuffer_ms: float = 150.0
    logged: bool = False

    def format_summary(self) -> Optional[str]:
        if self.t_ws_first_send <= 0.0:
            return None
        ref_vad = self.t_vad_end if self.t_vad_end > 0.0 else (
            self.t_asr_start if self.t_asr_start > 0.0 else self.t_llm_start
        )
        if ref_vad <= 0.0:
            return None

        post_vad_ms = max(0.0, (self.t_ws_first_send - ref_vad) * 1000.0)
        vad_tail_ms = self.vad_end_delay_ms if self.t_vad_end > 0.0 else 0.0
        total_syllable_to_audio_ms = vad_tail_ms + post_vad_ms + self.player_prebuffer_ms

        asr_ms = (
            max(0.0, (self.t_asr_done - self.t_vad_end) * 1000.0)
            if self.t_asr_done > 0.0 and self.t_vad_end > 0.0
            else (
                max(0.0, (self.t_asr_done - self.t_asr_start) * 1000.0)
                if self.t_asr_done > 0.0 and self.t_asr_start > 0.0
                else 0.0
            )
        )
        llm_ref = self.t_llm_start if self.t_llm_start > 0.0 else (
            self.t_asr_done if self.t_asr_done > 0.0 else ref_vad
        )
        llm_ttft_ms = (
            max(0.0, (self.t_llm_first_token - llm_ref) * 1000.0)
            if self.t_llm_first_token > 0.0 and llm_ref > 0.0
            else 0.0
        )
        llm_seg_ms = (
            max(0.0, (self.t_llm_first_segment - llm_ref) * 1000.0)
            if self.t_llm_first_segment > 0.0 and llm_ref > 0.0
            else 0.0
        )
        tts_ref = self.t_tts_first_start if self.t_tts_first_start > 0.0 else self.t_llm_first_segment
        tts_ms = (
            max(0.0, (self.t_tts_first_audio - tts_ref) * 1000.0)
            if self.t_tts_first_audio > 0.0 and tts_ref > 0.0
            else 0.0
        )
        lam_ref = self.t_tts_first_audio if self.t_tts_first_audio > 0.0 else self.t_lam_first_motion
        lam_ms = (
            max(0.0, (self.t_lam_first_motion - lam_ref) * 1000.0)
            if self.t_lam_first_motion > 0.0 and lam_ref > 0.0
            else 0.0
        )

        return (
            f"[LATENCY {self.speech_id}] "
            f"Total syllable->audio: {total_syllable_to_audio_ms:.0f}ms "
            f"(Post-VAD: {post_vad_ms:.0f}ms) | "
            f"VAD_tail={vad_tail_ms:.0f}ms | "
            f"ASR={asr_ms:.0f}ms (pre={self.asr_pre_chunks}) | "
            f"LLM_1st_tok={llm_ttft_ms:.0f}ms | "
            f"LLM_1st_seg={llm_seg_ms:.0f}ms | "
            f"TTS_1st={tts_ms:.0f}ms | "
            f"LAM={lam_ms:.0f}ms | "
            f"Player_buf={self.player_prebuffer_ms:.0f}ms"
        )


@dataclass
class SharedStates:
    active: bool = False
    enable_vad: bool = True
    human_speaking: bool = False
    avatar_speaking: bool = False
    pipeline_generating: bool = False
    client_no_aec: bool = False
    turn_generation_id: int = 0
    barge_in_seq: int = 0
    prewarm_speech_id: Optional[str] = None
    prewarm_seq: int = 0
    active_speech_id: Optional[str] = None
    aborted_speech_ids: List[str] = field(default_factory=list)
    interrupt_callback: Optional[Callable[[], None]] = None

    latest_avatar_emotion: str = "neutral"
    latest_emotion_seq: int = 0
    # speech_id -> emotion, written by the LLM handler and read by the avatar handler
    avatar_emotions: Dict[str, str] = field(default_factory=dict)
    latency_metrics: Dict[str, TurnLatencyMetrics] = field(default_factory=dict)

    def set_avatar_emotion(self, speech_id: str, emotion: str):
        self.avatar_emotions.pop(speech_id, None)
        self.avatar_emotions[speech_id] = emotion
        self.latest_avatar_emotion = emotion
        self.latest_emotion_seq += 1
        while len(self.avatar_emotions) > MAX_TRACKED_AVATAR_EMOTIONS:
            self.avatar_emotions.pop(next(iter(self.avatar_emotions)))

    def get_avatar_emotion(self, speech_id: Optional[str]) -> Optional[str]:
        if speech_id is None:
            return None
        return self.avatar_emotions.get(speech_id)

    def pop_avatar_emotion(self, speech_id: Optional[str]):
        if speech_id is not None:
            self.avatar_emotions.pop(speech_id, None)

    def trigger_barge_in(self, new_speech_id: Optional[str] = None):
        """Abort current avatar generation/playback immediately when user barges in."""
        self.turn_generation_id += 1
        self.barge_in_seq += 1
        if self.active_speech_id and self.active_speech_id != new_speech_id:
            if self.active_speech_id not in self.aborted_speech_ids:
                self.aborted_speech_ids.append(self.active_speech_id)
                while len(self.aborted_speech_ids) > MAX_TRACKED_LATENCY_TURNS:
                    self.aborted_speech_ids.pop(0)
        self.avatar_speaking = False
        self.pipeline_generating = False
        if self.interrupt_callback is not None:
            try:
                self.interrupt_callback()
            except Exception as e:
                logger.debug(f"Interrupt callback error: {e}")

    def is_turn_aborted(self, speech_id: Optional[str] = None, generation_id: Optional[int] = None) -> bool:
        if generation_id is not None and generation_id != self.turn_generation_id:
            return True
        if speech_id is not None and speech_id in self.aborted_speech_ids:
            return True
        return False

    def get_or_create_latency(self, speech_id: str) -> TurnLatencyMetrics:
        metrics = self.latency_metrics.get(speech_id)
        if metrics is None:
            metrics = TurnLatencyMetrics(speech_id=speech_id)
            self.latency_metrics[speech_id] = metrics
            while len(self.latency_metrics) > MAX_TRACKED_LATENCY_TURNS:
                self.latency_metrics.pop(next(iter(self.latency_metrics)))
        return metrics

    def mark_speech_start(self, speech_id: str, now: Optional[float] = None):
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        if m.t_speech_start <= 0.0:
            m.t_speech_start = t

    def mark_speech_prewarm(self, speech_id: str, now: Optional[float] = None):
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        if m.t_speech_prewarm <= 0.0:
            m.t_speech_prewarm = t
        if self.prewarm_speech_id != speech_id:
            self.prewarm_speech_id = speech_id
            self.prewarm_seq += 1

    def mark_vad_end(self, speech_id: str, vad_end_delay_ms: float = 480.0, now: Optional[float] = None):
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        m.t_vad_end = t
        m.vad_end_delay_ms = vad_end_delay_ms

    def mark_asr_done(self, speech_id: str, pre_chunks: int = 0,
                      t_start: Optional[float] = None, now: Optional[float] = None):
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        if t_start is not None and m.t_asr_start <= 0.0:
            m.t_asr_start = t_start
        m.t_asr_done = t
        m.asr_pre_chunks = pre_chunks

    def mark_llm_first_token(self, speech_id: str, t_start: Optional[float] = None, now: Optional[float] = None):
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        if t_start is not None and m.t_llm_start <= 0.0:
            m.t_llm_start = t_start
        if m.t_llm_first_token <= 0.0:
            m.t_llm_first_token = t

    def mark_llm_first_segment(self, speech_id: str, now: Optional[float] = None):
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        if m.t_llm_first_segment <= 0.0:
            m.t_llm_first_segment = t

    def mark_tts_first_audio(self, speech_id: str, t_start: Optional[float] = None, now: Optional[float] = None):
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        if t_start is not None and m.t_tts_first_start <= 0.0:
            m.t_tts_first_start = t_start
        if m.t_tts_first_audio <= 0.0:
            m.t_tts_first_audio = t

    def mark_lam_first_motion(self, speech_id: str, now: Optional[float] = None):
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        if m.t_lam_first_motion <= 0.0:
            m.t_lam_first_motion = t

    def mark_ws_first_send(self, speech_id: str, now: Optional[float] = None) -> Optional[str]:
        if not speech_id:
            return None
        t = time.monotonic() if now is None else now
        m = self.get_or_create_latency(speech_id)
        if m.t_ws_first_send <= 0.0:
            m.t_ws_first_send = t
        if not m.logged:
            summary = m.format_summary()
            if summary:
                m.logged = True
                logger.info(summary)
                return summary
        return None


class SessionContext(object):
    def __init__(self, session_info: SessionInfoData,
                 input_queues: Dict[EngineChannelType, IOQueueType],
                 output_queues: Dict[EngineChannelType, IOQueueType]):
        self.session_info = session_info
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.shared_states = SharedStates()
        self.input_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.input_start_time: float = -1.0

    def get_input_audio_definition(self, sample_rate: int, channel_num: int = 1, entry_name: str = "mic_audio"):
        definition = self.input_definitions.get(EngineChannelType.AUDIO, None)
        if definition is None:
            definition = DataBundleDefinition()
            definition.add_entry(DataBundleEntry.create_audio_entry(entry_name, channel_num, sample_rate))
            definition.lockdown()
            self.input_definitions[EngineChannelType.AUDIO] = definition
        return definition

    def get_input_video_definition(self, frame_shape: List[int], frame_rate: int, entry_name: str = "camera_video",
                                   allow_shape_change: bool = False):
        data_shape = [1] + frame_shape
        definition = self.input_definitions.get(EngineChannelType.VIDEO, None)
        if allow_shape_change and definition is not None:
            defined_shape = list(definition.get_main_entry().shape)
            if defined_shape != data_shape:
                logger.warning(f"Video definition updated, orign shape={defined_shape}, new shape={data_shape}")
                definition = None
        if definition is None:
            definition = DataBundleDefinition()
            definition.add_entry(DataBundleEntry.create_framed_entry(entry_name, data_shape, 0, frame_rate))
            definition.lockdown()
            self.input_definitions[EngineChannelType.VIDEO] = definition
        return definition
    
    def get_input_text_definition(self, entry_name: str = "human_text"):
        definition = self.input_definitions.get(EngineChannelType.TEXT, None)
        if definition is None:
            definition = DataBundleDefinition()
            definition.add_entry(DataBundleEntry.create_text_entry(entry_name))
            definition.lockdown()
            self.input_definitions[EngineChannelType.TEXT] = definition
        return definition

    def cleanup(self):
        for data_queue in self.input_queues.values():
            while not data_queue.empty():
                data_queue.get_nowait()
        for data_queue in self.output_queues.values():
            while not data_queue.empty():
                data_queue.get_nowait()

    def set_input_start(self):
        if self.input_start_time < 0:
            now = time.monotonic()
            self.input_start_time = now

    def get_timestamp(self) -> Tuple[int, int]:
        now = time.monotonic()
        if self.input_start_time < 0:
            return -1, 0
        else:
            return (
                round((now - self.input_start_time) * self.session_info.timestamp_base),
                self.session_info.timestamp_base
            )
