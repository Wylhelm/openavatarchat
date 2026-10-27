"""
RealtimeBridge handler — full-duplex speech-to-speech via OpenAI Realtime API.

Replaces SileroVad + WhisperLocal + CoquiLocal with a single expressive voice
bridge. Raw mic audio (MIC_AUDIO) is streamed to the avatar-bridge over a
WebSocket; the bridge relays it to the OpenAI Realtime API (semantic VAD + STT +
expressive TTS), routes the transcript through the OpenClaw gateway (avatar
agent brain + tools), and streams the expressive audio back. This handler emits
that audio as AVATAR_AUDIO for the LAM driver / WebRTC player.

Transport: WebSocket (sync client from `websockets`, already in the engine
image). An HTTP streaming POST cannot work here: `requests` uploads the whole
body before reading the response, so nothing ever came back.

Wire protocol (engine <-> bridge), JSON text frames:
  engine -> bridge:
    {"type":"audio","audio_b64":"<base64 16-bit PCM 24kHz>"}
  bridge -> engine:
    {"type":"audio","audio_b64":"...","response_id":"..."}  # 24kHz PCM
    {"type":"audio_done","response_id":"..."}               # end of a reply
    {"type":"speech_started"}                               # user barge-in
    {"type":"text","text":"..."}                            # transcript (debug)

Reversible: this is an ADDITION. SileroVad/Whisper/Coqui stay intact; enabling
this handler and disabling the other three in the yaml switches modes.
"""
import base64
import json
import threading
from abc import ABC
from typing import Dict, Optional, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry

# OpenAI Realtime input/output is 24kHz mono 16-bit PCM.
REALTIME_SAMPLE_RATE = 24000
# ~10 s of 100 ms chunks; older mic audio is dropped if the bridge stalls.
MAX_OUTBOUND_CHUNKS = 100


class RealtimeBridgeConfig(HandlerBaseConfigModel, BaseModel):
    # ws:// or wss:// (http(s):// is accepted and converted).
    bridge_url: str = Field(default="ws://10.0.0.253:9000/realtime/voice")
    api_key: Optional[str] = Field(default=None)
    # Mic arrives at 16kHz (WebRTC input); we upsample to 24kHz for Realtime.
    input_sample_rate: int = Field(default=16000)
    output_sample_rate: int = Field(default=24000)
    # Emit output audio in frames of this many samples (480 == 20ms @ 24kHz).
    output_frame_size: int = Field(default=480)
    # Max audio samples buffered before flushing a chunk to the bridge (2400 == 100ms @ 24kHz).
    chunk_samples: int = Field(default=2400)
    reconnect_delay_seconds: float = Field(default=1.0)
    connect_timeout: float = Field(default=10.0)


class RealtimeBridgeContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[RealtimeBridgeConfig] = None
        self.shared_states = None
        self.thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Outbound base64 PCM chunks pushed from handle(), drained by the sender thread.
        self._outbound: list = []
        self._outbound_cv = threading.Condition(threading.Lock())
        # Pending mic samples (24 kHz float32) not yet flushed as a chunk.
        self._pending = np.zeros(0, dtype=np.float32)
        self._connected = threading.Event()
        self.ws = None
        # speech_id of the reply currently being played, so LAM keeps one stream per reply.
        self.current_speech_id: Optional[str] = None


class HandlerRealtimeBridge(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.config: Optional[RealtimeBridgeConfig] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="RealtimeBridge",
            config_model=RealtimeBridgeConfig,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        self.config = cast(RealtimeBridgeConfig, handler_config)
        logger.info(f"RealtimeBridge handler loaded: bridge_url='{self._ws_url()}'")

    def _ws_url(self) -> str:
        url = self.config.bridge_url
        if url.startswith("http://"):
            url = "ws://" + url[len("http://"):]
        elif url.startswith("https://"):
            url = "wss://" + url[len("https://"):]
        return url

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

    # ---- worker: WebSocket connection; receive here, send from a sender thread ----

    def _worker(self, context: RealtimeBridgeContext):
        while not context._stop.is_set():
            try:
                self._stream_loop(context)
            except Exception as e:
                if not context._stop.is_set():
                    logger.warning(f"RealtimeBridge stream error: {e!r}; reconnecting...")
            finally:
                context._connected.clear()
                context.ws = None
                with context._outbound_cv:
                    context._outbound_cv.notify_all()
            if not context._stop.is_set():
                context._stop.wait(timeout=context.config.reconnect_delay_seconds)

    def _stream_loop(self, context: RealtimeBridgeContext):
        url = self._ws_url()
        logger.info(f"RealtimeBridge connecting to {url} ...")
        headers = {}
        if context.config.api_key:
            headers["Authorization"] = f"Bearer {context.config.api_key}"
        with ws_connect(url, additional_headers=headers, open_timeout=context.config.connect_timeout,
                        max_size=None) as ws:
            context.ws = ws
            with context._outbound_cv:
                context._outbound.clear()
            context._pending = np.zeros(0, dtype=np.float32)
            context._connected.set()
            logger.info("RealtimeBridge WebSocket connected")
            sender = threading.Thread(target=self._sender, args=(context, ws), daemon=True)
            sender.start()
            try:
                while not context._stop.is_set():
                    try:
                        raw = ws.recv(timeout=0.5)
                    except TimeoutError:
                        continue
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    self._handle_bridge_event(context, msg)
            finally:
                context._connected.clear()
                with context._outbound_cv:
                    context._outbound_cv.notify_all()
                sender.join(timeout=2.0)

    def _sender(self, context: RealtimeBridgeContext, ws):
        try:
            while not context._stop.is_set() and context._connected.is_set():
                chunk = self._pop_chunk(context, timeout=0.5)
                if chunk is None:
                    continue
                ws.send(json.dumps({"type": "audio", "audio_b64": chunk}))
        except ConnectionClosed:
            pass
        except Exception as e:
            logger.warning(f"RealtimeBridge sender error: {e!r}")

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
            self._emit_audio(context, msg.get("audio_b64", ""), msg.get("response_id") or "")
        elif mtype == "audio_done":
            self._emit_speech_end(context)
        elif mtype == "speech_started":
            logger.info("RealtimeBridge: user speech started")
        elif mtype == "text":
            logger.info(f"RealtimeBridge transcript: {msg.get('text', '')}")

    def _make_bundle(self, context: RealtimeBridgeContext, audio: np.ndarray, speech_id: str,
                     speech_end: bool) -> ChatData:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry(
            "avatar_audio", 1, context.config.output_sample_rate))
        bundle = DataBundle(definition)
        bundle.set_main_data(np.expand_dims(audio, axis=0))
        bundle.add_meta("avatar_speech_end", speech_end)
        bundle.add_meta("speech_id", speech_id)
        return ChatData(type=ChatDataType.AVATAR_AUDIO, data=bundle)

    def _emit_audio(self, context: RealtimeBridgeContext, audio_b64: str, response_id: str):
        if not audio_b64:
            return
        try:
            raw = base64.b64decode(audio_b64)
        except Exception:
            return
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return
        speech_id = f"rt-{response_id}" if response_id else f"rt-{context.session_id}"
        if context.current_speech_id is not None and context.current_speech_id != speech_id:
            self._emit_speech_end(context)
        context.current_speech_id = speech_id
        frame = context.config.output_frame_size
        for i in range(0, audio.size, frame):
            chunk = audio[i:i + frame]
            if chunk.size:
                context.submit_data(self._make_bundle(context, chunk, speech_id, False))

    def _emit_speech_end(self, context: RealtimeBridgeContext):
        if context.current_speech_id is None:
            return
        silence = np.zeros(context.config.output_frame_size, dtype=np.float32)
        context.submit_data(self._make_bundle(context, silence, context.current_speech_id, True))
        context.current_speech_id = None

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
        resampled = self._resample(audio, context.config.input_sample_rate, REALTIME_SAMPLE_RATE)
        context._pending = np.concatenate([context._pending, resampled])
        if context._pending.size < context.config.chunk_samples:
            return
        pcm = np.clip(context._pending * 32767.0, -32768, 32767).astype(np.int16)
        context._pending = np.zeros(0, dtype=np.float32)
        b64 = base64.b64encode(pcm.tobytes()).decode()
        with context._outbound_cv:
            context._outbound.append(b64)
            if len(context._outbound) > MAX_OUTBOUND_CHUNKS:
                del context._outbound[0]
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
        ws = context.ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        with context._outbound_cv:
            context._outbound_cv.notify_all()
        if context.thread is not None:
            context.thread.join(timeout=2.0)
        logger.info("RealtimeBridge context destroyed")
