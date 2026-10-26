"""
RealtimeBridge handler — full-duplex speech-to-speech via OpenAI Realtime API.

Replaces SileroVad + WhisperLocal + CoquiLocal with a single expressive voice
bridge. Raw mic audio (MIC_AUDIO) is streamed to the avatar-bridge over HTTP;
the bridge relays it to the OpenAI Realtime API (semantic VAD + STT + expressive
TTS), routes the transcript through the OpenClaw gateway (avatar agent brain +
tools), and streams the expressive audio back. This handler emits that audio as
AVATAR_AUDIO for the WebRTC player.

Transport: HTTP streaming (requests), NOT a WebSocket client. This matches the
existing Whisper/Coqui handlers (both use `requests`) and avoids adding a
`websockets` dependency to the engine image — so no image rebuild is needed,
only a bind-mount of this package + a container restart.

Wire protocol (engine <-> bridge), HTTP POST with newline-delimited JSON body:
  engine -> bridge (request body, one JSON per line):
    {"type":"audio","audio_b64":"<base64 16-bit PCM 24kHz>"}
  bridge -> engine (streamed response body, one JSON per line):
    {"type":"audio","audio_b64":"..."}   # 24kHz PCM, playable
    {"type":"text","text":"..."}         # transcript (debug)
    {"type":"done"}

Reversible: this is an ADDITION. SileroVad/Whisper/Coqui stay intact; enabling
this handler and disabling the other three in the yaml switches modes.
"""
import base64
import json
import threading
from abc import ABC
from typing import Dict, Optional, cast

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

# OpenAI Realtime input/output is 24kHz mono 16-bit PCM.
REALTIME_SAMPLE_RATE = 24000


class RealtimeBridgeConfig(HandlerBaseConfigModel, BaseModel):
    bridge_url: str = Field(default="http://10.0.0.253:9000/realtime/voice")
    # Mic arrives at 16kHz (WebRTC input); we upsample to 24kHz for Realtime.
    input_sample_rate: int = Field(default=16000)
    output_sample_rate: int = Field(default=24000)
    # Emit output audio in frames of this many samples (480 == 20ms @ 24kHz).
    output_frame_size: int = Field(default=480)
    # Max audio samples buffered before flushing a chunk to the bridge (2400 == 100ms @ 24kHz).
    chunk_samples: int = Field(default=2400)
    # HTTP request timeout for the streaming POST (seconds). Long-lived connection.
    request_timeout: float = Field(default=3600.0)


class RealtimeBridgeContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[RealtimeBridgeConfig] = None
        self.shared_states = None
        # Worker thread that owns the streaming HTTP connection.
        self.thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Outbound audio chunks (bytes) pushed from handle(), drained by the worker.
        self._outbound: list = []          # guarded by _outbound_lock
        self._outbound_lock = threading.Lock()
        self._outbound_cv = threading.Condition(self._outbound_lock)
        self._connected = threading.Event()


class HandlerRealtimeBridge(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.config: Optional[RealtimeBridgeConfig] = None
        self._thread_local = threading.local()

    def _get_http_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="RealtimeBridge",
            config_model=RealtimeBridgeConfig,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        self.config = cast(RealtimeBridgeConfig, handler_config)
        logger.info(f"RealtimeBridge handler loaded: bridge_url='{self.config.bridge_url}'")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        context = RealtimeBridgeContext(session_context.session_info.session_id)
        context.config = self.config
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        context = cast(RealtimeBridgeContext, handler_context)
        context.thread = threading.Thread(target=self._worker, args=(context,), daemon=True)
        context.thread.start()
        logger.info("RealtimeBridge worker thread started")

    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        output_def = DataBundleDefinition()
        output_def.add_entry(DataBundleEntry.create_audio_entry(
            "avatar_audio", 1, self.config.output_sample_rate))

        inputs = {ChatDataType.MIC_AUDIO: HandlerDataInfo(type=ChatDataType.MIC_AUDIO)}
        outputs = {ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
            type=ChatDataType.AVATAR_AUDIO, definition=output_def)}
        return HandlerDetail(inputs=inputs, outputs=outputs)

    # ---- worker: long-lived streaming HTTP POST, drain outbound + emit inbound ----

    def _worker(self, context: RealtimeBridgeContext):
        """Maintain a streaming POST to the bridge; send mic audio, read avatar audio."""
        while not context._stop.is_set():
            try:
                self._stream_loop(context)
            except Exception as e:
                logger.warning(f"RealtimeBridge stream error: {e}; reconnecting...")
                context._connected.clear()
            finally:
                context._connected.clear()
            if not context._stop.is_set():
                context._stop.wait(timeout=1.0)

    def _stream_loop(self, context: RealtimeBridgeContext):
        session = self._get_http_session()
        logger.info(f"RealtimeBridge connecting to {context.config.bridge_url} ...")

        # Generator yields audio chunks as they arrive; requests streams the body.
        def body_gen():
            while not context._stop.is_set():
                chunk = self._pop_chunk(context, timeout=0.5)
                if chunk is None:
                    # keepalive: send a comment line to keep the connection alive
                    continue
                yield json.dumps({"type": "audio", "audio_b64": chunk}) + "\n"

        with session.post(
            context.config.bridge_url,
            data=body_gen(),
            headers={"Content-Type": "application/x-ndjson"},
            stream=True,
            timeout=context.config.request_timeout,
        ) as resp:
            resp.raise_for_status()
            context._connected.set()
            logger.info("RealtimeBridge HTTP stream connected")
            # Read newline-delimited JSON events from the bridge.
            for line in resp.iter_lines(decode_unicode=True):
                if context._stop.is_set():
                    break
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                self._handle_bridge_event(context, msg)

    def _pop_chunk(self, context: RealtimeBridgeContext, timeout: float) -> Optional[str]:
        with context._outbound_cv:
            if not context._outbound:
                context._outbound_cv.wait(timeout=timeout)
            if not context._outbound:
                return None
            return context._outbound.pop(0)

    def _handle_bridge_event(self, context: RealtimeBridgeContext, msg: dict):
        mtype = msg.get("type")
        if mtype == "audio":
            self._emit_audio(context, msg.get("audio_b64", ""))
        elif mtype == "text":
            logger.debug(f"RealtimeBridge transcript: {msg.get('text', '')}")
        elif mtype == "done":
            pass

    def _emit_audio(self, context: RealtimeBridgeContext, audio_b64: str):
        if not audio_b64:
            return
        try:
            raw = base64.b64decode(audio_b64)
        except Exception:
            return
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return
        frame = context.config.output_frame_size
        for i in range(0, audio.size, frame):
            chunk = audio[i:i + frame]
            if chunk.size == 0:
                continue
            definition = DataBundleDefinition()
            definition.add_entry(DataBundleEntry.create_audio_entry(
                "avatar_audio", 1, context.config.output_sample_rate))
            bundle = DataBundle(definition)
            bundle.set_main_data(np.expand_dims(chunk, axis=0))
            bundle.add_meta("avatar_speech_end", False)
            bundle.add_meta("speech_id", f"rt-{context.session_id}-{i}")
            chat_data = ChatData(type=ChatDataType.AVATAR_AUDIO, data=bundle)
            context.submit_data(chat_data)

    # ---- handle(): receive mic audio, resample 24k, push to outbound ----

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(RealtimeBridgeContext, context)
        if inputs.type != ChatDataType.MIC_AUDIO:
            return
        if not context._connected.is_set():
            return
        audio = inputs.data.get_main_data()
        if audio is None:
            return
        audio = audio.squeeze()
        if audio.size == 0:
            return
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        resampled = self._resample(audio, 16000, REALTIME_SAMPLE_RATE)
        pcm = np.clip(resampled * 32767.0, -32768, 32767).astype(np.int16)
        b64 = base64.b64encode(pcm.tobytes()).decode()
        with context._outbound_cv:
            context._outbound.append(b64)
            context._outbound_cv.notify_all()

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio
        n_out = int(round(audio.size * dst_sr / src_sr))
        x_old = np.linspace(0.0, audio.size - 1, num=audio.size)
        x_new = np.linspace(0.0, audio.size - 1, num=n_out)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def destroy_context(self, context: HandlerContext):
        context = cast(RealtimeBridgeContext, context)
        context._stop.set()
        if context.thread is not None:
            context.thread.join(timeout=2.0)
        logger.info("RealtimeBridge context destroyed")
