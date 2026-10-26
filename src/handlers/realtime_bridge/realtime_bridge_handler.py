"""
RealtimeBridge handler — true full-duplex speech-to-speech via OpenAI Realtime API.

This handler replaces SileroVad + WhisperLocal + CoquiLocal with a single
full-duplex audio bridge. The mic audio (MIC_AUDIO) is streamed verbatim to the
avatar-bridge over a WebSocket; the bridge relays it to the OpenAI Realtime API
(which does its own semantic VAD, transcription, and expressive TTS), runs the
transcript through the OpenClaw gateway (the avatar agent's brain + tools), and
streams the resulting expressive audio back. This handler emits that audio as
AVATAR_AUDIO for the WebRTC player.

Design goals:
- Reversible: this is an ADDITION. SileroVad/Whisper/Coqui stay intact; enabling
  this handler and disabling the other three in the yaml switches modes.
- No local VAD: turn-taking is delegated to the Realtime API (semantic_vad),
  which is what makes conversation feel natural (interruptions, backchannel).
- Streaming: audio is sent in small chunks as it arrives, and received audio is
  emitted as soon as deltas arrive, for low latency.

The bridge exposes a single WebSocket endpoint:  ws://<bridge>/realtime/voice
The wire protocol (both directions) is a JSON envelope:
  { "type": "audio",  "audio_b64": "<base64 16-bit PCM mono 24kHz>" }
  { "type": "end" }
Server -> client:
  { "type": "audio",  "audio_b64": "..." }     # 24kHz PCM mono, playable directly
  { "type": "text",   "text": "..." }          # optional transcript (debug)
  { "type": "done" }
"""
import asyncio
import base64
import json
import threading
from abc import ABC
from typing import Dict, Optional, cast

import numpy as np
import websockets
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry

# Realtime API input/output is 24kHz mono 16-bit PCM.
REALTIME_SAMPLE_RATE = 24000


class RealtimeBridgeConfig(HandlerBaseConfigModel, BaseModel):
    ws_url: str = Field(default="ws://10.0.0.253:9000/realtime/voice")
    # Mic arrives at 16kHz (WebRTC input); we upsample to 24kHz for Realtime.
    input_sample_rate: int = Field(default=16000)
    output_sample_rate: int = Field(default=24000)
    # Emit output audio in frames of this many samples (480 == 20ms @ 24kHz).
    output_frame_size: int = Field(default=480)
    # Reconnect backoff when the bridge WebSocket drops.
    reconnect_delay_seconds: float = Field(default=1.0)
    # Max audio samples to buffer before flushing to the bridge (2400 == 100ms @ 24kHz).
    chunk_samples: int = Field(default=2400)


class RealtimeBridgeContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[RealtimeBridgeConfig] = None
        self.shared_states = None
        # WebSocket + worker loop (run in a background thread)
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Outbound queue: audio chunks (bytes) pushed from handle(), drained by worker.
        self._outbound: asyncio.Queue = None  # set in worker thread
        self._connected = threading.Event()
        # Resampler (16k -> 24k) — simple linear interp; avoids librosa dependency here.
        self._resample_ratio = REALTIME_SAMPLE_RATE / 16000


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
        logger.info(f"RealtimeBridge handler loaded: ws_url='{self.config.ws_url}'")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        context = RealtimeBridgeContext(session_context.session_info.session_id)
        context.config = self.config
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        context = cast(RealtimeBridgeContext, handler_context)
        context.loop = asyncio.new_event_loop()
        context._outbound = asyncio.Queue()

        def run():
            asyncio.set_event_loop(context.loop)
            context.loop.run_until_complete(self._worker(context))

        context.thread = threading.Thread(target=run, daemon=True)
        context.thread.start()
        logger.info("RealtimeBridge worker thread started")

    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        # Input: raw mic audio (no VAD). Output: expressive avatar audio.
        input_def = DataBundleDefinition()
        input_def.add_entry(DataBundleEntry.create_audio_entry("mic_audio", 1, self.config.input_sample_rate))

        output_def = DataBundleDefinition()
        output_def.add_entry(DataBundleEntry.create_audio_entry(
            "avatar_audio", 1, self.config.output_sample_rate))

        inputs = {ChatDataType.MIC_AUDIO: HandlerDataInfo(type=ChatDataType.MIC_AUDIO)}
        outputs = {ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
            type=ChatDataType.AVATAR_AUDIO, definition=output_def)}
        return HandlerDetail(inputs=inputs, outputs=outputs)

    # ---- worker (background thread): WS connect + outbound drain + inbound emit ----

    async def _worker(self, context: RealtimeBridgeContext):
        outbound = context._outbound
        while not context._stop.is_set():
            try:
                logger.info(f"RealtimeBridge connecting to {context.config.ws_url} ...")
                async with websockets.connect(context.config.ws_url, max_size=None) as ws:
                    context.ws = ws
                    context._connected.set()
                    logger.info("RealtimeBridge WebSocket connected")
                    await self._run_loop(context, ws, outbound)
            except Exception as e:
                logger.warning(f"RealtimeBridge WS error: {e}; reconnect in {context.config.reconnect_delay_seconds}s")
                context._connected.clear()
            finally:
                context._connected.clear()
                context.ws = None
            if not context._stop.is_set():
                await asyncio.sleep(context.config.reconnect_delay_seconds)

    async def _run_loop(self, context: RealtimeBridgeContext, ws, outbound: asyncio.Queue):
        """Simultaneously drain outbound mic audio and read inbound avatar audio."""
        async def drain_outbound():
            while not context._stop.is_set():
                try:
                    chunk = await asyncio.wait_for(outbound.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if chunk is None:
                    # sentinel: end of utterance (not used now; audio is continuous)
                    continue
                await ws.send(json.dumps({"type": "audio", "audio_b64": chunk}))

        async def read_inbound():
            async for raw in ws:
                if context._stop.is_set():
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                mtype = msg.get("type")
                if mtype == "audio":
                    self._emit_audio(context, msg.get("audio_b64", ""))
                elif mtype == "text":
                    logger.debug(f"RealtimeBridge transcript: {msg.get('text', '')}")
                elif mtype == "done":
                    pass

        # Run both concurrently; exit when either the stop flag is set or WS closes.
        await asyncio.gather(
            drain_outbound(),
            read_inbound(),
        )

    def _emit_audio(self, context: RealtimeBridgeContext, audio_b64: str):
        """Decode 24kHz PCM audio from the bridge and submit as AVATAR_AUDIO."""
        if not audio_b64:
            return
        try:
            raw = base64.b64decode(audio_b64)
        except Exception:
            return
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return
        # Emit in frame-size chunks
        frame = context.config.output_frame_size
        for i in range(0, audio.size, frame):
            chunk = audio[i:i + frame]
            if chunk.size == 0:
                continue
            data = DataBundle(None)
            # build a fresh definition each emit (cheap) to carry the audio
            definition = DataBundleDefinition()
            definition.add_entry(DataBundleEntry.create_audio_entry(
                "avatar_audio", 1, context.config.output_sample_rate))
            bundle = DataBundle(definition)
            bundle.set_main_data(np.expand_dims(chunk, axis=0))
            bundle.add_meta("avatar_speech_end", False)
            bundle.add_meta("speech_id", f"rt-{context.session_id}-{i}")
            chat_data = ChatData(type=ChatDataType.AVATAR_AUDIO, data=bundle)
            context.submit_data(chat_data)

    # ---- handle(): receive mic audio, resample to 24k, push to outbound ----

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(RealtimeBridgeContext, context)
        if inputs.type != ChatDataType.MIC_AUDIO:
            return
        if not context._connected.is_set():
            # Bridge not ready; drop (or could buffer). Dropping is safest for latency.
            return
        audio = inputs.data.get_main_data()
        if audio is None:
            return
        audio = audio.squeeze()
        if audio.size == 0:
            return
        # Normalize to float32 [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        # Resample 16k -> 24k (linear interpolation)
        resampled = self._resample(audio, 16000, REALTIME_SAMPLE_RATE)
        # Convert to int16 PCM
        pcm = np.clip(resampled * 32767.0, -32768, 32767).astype(np.int16)
        b64 = base64.b64encode(pcm.tobytes()).decode()
        # Push to worker (non-blocking; if queue full, drop oldest to stay real-time)
        try:
            context._outbound.put_nowait(b64)
        except asyncio.QueueFull:
            logger.debug("RealtimeBridge outbound queue full, dropping chunk")

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Linear-interpolation resampler (16k -> 24k). Accurate enough for speech."""
        if src_sr == dst_sr:
            return audio
        n_out = int(round(audio.size * dst_sr / src_sr))
        x_old = np.linspace(0.0, audio.size - 1, num=audio.size)
        x_new = np.linspace(0.0, audio.size - 1, num=n_out)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def destroy_context(self, context: HandlerContext):
        context = cast(RealtimeBridgeContext, context)
        context._stop.set()
        if context.loop is not None:
            try:
                context.loop.call_soon_threadsafe(context.loop.stop)
            except Exception:
                pass
        if context.thread is not None:
            context.thread.join(timeout=2.0)
        logger.info("RealtimeBridge context destroyed")
