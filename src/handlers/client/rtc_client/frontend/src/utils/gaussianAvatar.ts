import { Player } from '@/helpers/player'
import { Processor } from '@/helpers/processor'
import { type WS } from '@/helpers/ws.js'
import { EventTypes, PlayerEventTypes } from '@/interface/eventType'
import { TYVoiceChatState } from '@/interface/voiceChat'
import { AvatarLife, attachAvatarLife } from '@/utils/avatarLife'
import { ensureBlob } from '@/utils/binaryUtils'
import { WebcamFaceTracker } from '@/utils/webcamTracker'
import EventEmitter from 'eventemitter3'
// import * as GaussianSplats3D from "./gaussian-splats-3d.module.js";
import { WsEventTypes } from '@/interface/eventType'
// @ts-ignore for lam render
import * as GaussianSplats3D from 'gaussian-splat-renderer-for-lam'

// Falls back to Idle if no reply audio arrives (LLM error, empty reply)
const THINKING_TIMEOUT_MS = 30000

// A phone renders the splats at a 3x pixel ratio with no frame cap. That
// saturates its CPU within a couple of turns, and the starved audio thread
// then crackles. Face data is 30 fps, so rendering faster adds nothing.
const MOBILE_MAX_PIXEL_RATIO = 1.5
const MOBILE_FRAME_MS = 1000 / 30

function isMobileDevice() {
  return window.matchMedia?.('(pointer: coarse)').matches ?? false
}

function capDevicePixelRatio(max: number) {
  const native = window.devicePixelRatio || 1
  if (native <= max) return
  Object.defineProperty(window, 'devicePixelRatio', {
    configurable: true,
    get: () => max,
  })
}

function capFrameRate(renderer: any, frameMs: number) {
  if (!renderer || typeof renderer.render !== 'function' || renderer.__frameCapped) return
  renderer.__frameCapped = true
  const renderFrame = renderer.render.bind(renderer)
  let lastFrame = 0
  renderer.render = () => {
    const now = performance.now()
    if (now - lastFrame < frameMs - 2 && renderer.viewer) {
      renderer.viewer.requestFrameId = requestAnimationFrame(() => renderer.render())
      return
    }
    lastFrame = now
    renderFrame()
  }
}

interface GaussianOptions {
  container: HTMLDivElement
  assetsPath: string
  ws: WS
  avatarScale?: number
  trackingVideoElement?: HTMLVideoElement | null
  downloadProgress?: (percent: number) => void
  loadProgress?: (percent: number) => void
}

export class GaussianAvatar extends EventEmitter {
  private _avatarDivEle: HTMLDivElement
  private _assetsPath = ''
  private _ws: WS
  private _downloadProgress: (percent: number) => void
  private _loadProgress: (percent: number) => void
  private _loadPercent = 0
  private _downloadPercent = 0
  private _processor!: Processor
  private _renderer: any
  private _audioMute = false
  private _thinkingTimer: ReturnType<typeof setTimeout> | null = null
  private _exited = false
  private _avatarScale = 1
  private _life = new AvatarLife()
  private _trackingVideoElement: HTMLVideoElement | null = null
  private _faceTracker = new WebcamFaceTracker((target) => {
    if (target) {
      this._life.setUserTarget(target.x, target.y, true, {
        nodding: target.nodding,
        smile: target.smile,
      })
    } else {
      this._life.setUserTarget(0, 0, false)
    }
  })
  curState = TYVoiceChatState.Idle
  constructor(options: GaussianOptions) {
    const {
      container,
      assetsPath,
      ws,
      avatarScale,
      trackingVideoElement,
      downloadProgress,
      loadProgress,
    } = options
    super()
    if (avatarScale && avatarScale > 0) {
      this._avatarScale = avatarScale
    }
    this._trackingVideoElement = trackingVideoElement ?? null
    this._avatarDivEle = container
    this._assetsPath = assetsPath
    this._ws = ws
    if (downloadProgress) {
      this._downloadProgress = (percent: number) => {
        this._downloadPercent = percent
        downloadProgress(percent)
      }
    } else {
      this._downloadProgress = (percent: number) => {
        this._downloadPercent = percent
      }
    }
    if (loadProgress) {
      this._loadProgress = (percent: number) => {
        this._loadPercent = percent
        loadProgress(percent)
      }
    } else {
      this._loadProgress = (percent: number) => {
        this._loadPercent = percent
      }
    }
    this._init()
  }
  private _init() {
    if (!this._avatarDivEle || !this._assetsPath || !this._ws) {
      throw new Error('Lack of necessary initialization parameters for gaussian render')
    }
    this._processor = new Processor(this)
    this._bindEventTypes()
  }
  start() {
    this.getData()
    this.render()
    if (this._trackingVideoElement) {
      this._faceTracker.start(this._trackingVideoElement)
    }
  }

  setTrackingVideoElement(videoEl: HTMLVideoElement | null | undefined) {
    this._trackingVideoElement = videoEl ?? null
    if (this._trackingVideoElement && !this._exited) {
      this._faceTracker.start(this._trackingVideoElement)
    } else {
      this._faceTracker.stop()
    }
  }

  async getData() {
    this._ws.on(WsEventTypes.WS_MESSAGE, (data: unknown) => {
      // Text frames are control messages (server Ping). Motion data is binary.
      const blob = ensureBlob(data)
      if (!blob) {
        return
      }
      if (this._downloadPercent < 1 || this._loadPercent < 1) {
        // 本地数字人未加载完成前，不处理数据
        return
      }

      this.emit(EventTypes.MessageReceived, this.curState)

      this._processor.add({
        avatar_motion_data: {
          first_package: true, // 是否首包
          segment_num: 1, // 分片数量，首包存在该值
          binary_size: blob.size, // 数据大小，首包存在该值
          use_binary_frame: false, // 是否使用二进制帧，首包存在该值
        },
      })

      this._processor.add({
        avatar_motion_data: {
          first_package: false,
          motion_data_slice: blob, // 数据分片，非首包存在该值
          is_audio_mute: this._audioMute, // 音频片段是否静音，非首包存在该值
        },
      })
    })
  }

  async render() {
    const mobile = isMobileDevice()
    if (mobile) capDevicePixelRatio(MOBILE_MAX_PIXEL_RATIO)
    this._renderer = await GaussianSplats3D.GaussianSplatRenderer.getInstance(
      this._avatarDivEle,
      this._assetsPath,
      {
        getChatState: this.getChatState.bind(this),
        getExpressionData: this.getArkitFaceFrame.bind(this),
        downloadProgress: this._downloadProgress.bind(this),
        loadProgress: (percent: number) => {
          this._loadProgress(percent)
          if (percent >= 1) this._attachLife()
        },
        backgroundColor: (() => {
          const theme = document.documentElement.getAttribute('data-theme') || 'dark'
          const isLight =
            theme === 'light' ||
            (theme === 'system' && window.matchMedia('(prefers-color-scheme: light)').matches)
          return isLight ? '0xf4f4f8' : '0x0a0a0f'
        })(),
      }
    )
    if (mobile) capFrameRate(this._renderer, MOBILE_FRAME_MS)
    this._applyAvatarScale()
    this._attachLife()
  }
  private _attachLife() {
    attachAvatarLife(this._renderer, this._life)
  }
  private _applyAvatarScale() {
    const camera = this._renderer?.viewer?.camera
    if (!camera || this._avatarScale === 1) {
      return
    }
    camera.zoom = this._avatarScale
    camera.updateProjectionMatrix?.()
    this._renderer?.viewer?.forceRenderNextFrame?.()
  }
  setBackgroundColor(hexColor: string) {
    try {
      const color = parseInt(hexColor.replace(/^(#|0x)/, ''), 16)
      if (this._renderer?.viewer?.renderer) {
        this._renderer.viewer.renderer.setClearColor(color, 1.0)
        this._renderer.viewer.backgroundColor = color
        this._renderer.viewer.forceRenderNextFrame()
      }
    } catch (e) {
      console.warn('Failed to set avatar background color:', e)
    }
  }
  setAvatarMute(isMute: boolean) {
    this._processor.setMute(isMute)
    this._audioMute = isMute
  }
  getChatState() {
    return this.curState
  }
  getArkitFaceFrame() {
    const speech = this._processor?.getArkitFaceFrame()?.arkitFace ?? null
    return this._life.tick(this.curState, undefined, speech)
  }
  startListening(): void {
    if (this.curState === TYVoiceChatState.Responding) {
      return
    }
    this._clearThinkingTimer()
    this.curState = TYVoiceChatState.Listening
    this.emit(EventTypes.StateChanged, this.curState)
  }
  setEmotion(emotion: string): void {
    this._life.setEmotion(emotion)
  }
  stopListening(): void {
    if (this.curState === TYVoiceChatState.Listening) {
      this._clearThinkingTimer()
      this.curState = TYVoiceChatState.Idle
      this.emit(EventTypes.StateChanged, this.curState)
    }
  }
  startThinking(): void {
    if (this.curState === TYVoiceChatState.Responding) {
      return
    }
    this._clearThinkingTimer()
    this._processor?.clear()
    this.curState = TYVoiceChatState.Thinking
    this.emit(EventTypes.StateChanged, this.curState)
    this._thinkingTimer = setTimeout(() => {
      this._thinkingTimer = null
      if (this.curState === TYVoiceChatState.Thinking) {
        this.curState = TYVoiceChatState.Idle
        this.emit(EventTypes.StateChanged, this.curState)
      }
    }, THINKING_TIMEOUT_MS)
  }
  private _clearThinkingTimer() {
    if (this._thinkingTimer) {
      clearTimeout(this._thinkingTimer)
      this._thinkingTimer = null
    }
  }
  interrupt(): void {
    this._clearThinkingTimer()
    this._ws.send('%interrupt%') // 约定的打断标识
    this._processor?.interrupt()
    this.curState = TYVoiceChatState.Idle
    this.emit(EventTypes.StateChanged, this.curState)
  }
  handleBargeIn(): void {
    this._clearThinkingTimer()
    this._processor?.interrupt()
    this.curState = TYVoiceChatState.Listening
    this.emit(EventTypes.StateChanged, this.curState)
  }
  sendSpeech(data: string | Int8Array | Uint8Array) {
    this._ws.send(data)
    this.curState = TYVoiceChatState.Listening
    this.emit(EventTypes.StateChanged, this.curState)
    this._processor?.clear()
  }
  exit() {
    if (this._exited) return
    this._exited = true
    this._faceTracker.stop()
    this._clearThinkingTimer()
    const renderer = this._renderer
    this._renderer = null
    renderer?.dispose()
    this.curState = TYVoiceChatState.Idle
    this._downloadPercent = 0
    this._loadPercent = 0
    this._processor?.clear()
    this.removeAllListeners()
  }
  private _bindEventTypes() {
    this.on(PlayerEventTypes.Player_StartSpeaking, (player: Player) => {
      console.log('startSpeach')
      this._clearThinkingTimer()
      this.curState = TYVoiceChatState.Responding
      this.emit(EventTypes.StateChanged, this.curState)
      this._ws.send(
        JSON.stringify({
          header: { name: EventTypes.StartSpeech },
          payload: {},
        })
      )
    })
    this.on(PlayerEventTypes.Player_EndSpeaking, (player: Player) => {
      console.log('endSpeach')
      this.curState = TYVoiceChatState.Idle
      this.emit(EventTypes.StateChanged, this.curState)
      this._ws.send(JSON.stringify({ header: { name: EventTypes.EndSpeech }, payload: {} }))
    })
    this.on(EventTypes.ErrorReceived, (data) => {
      const message = typeof data === 'string' ? data : String(data)
      console.log('ErrorReceived', message)
      this.curState = TYVoiceChatState.Idle
      this.emit(EventTypes.StateChanged, this.curState)
      this._ws.send(
        JSON.stringify({
          header: { name: EventTypes.ErrorReceived },
          payload: { message },
        })
      )
    })
    this._ws.on(WsEventTypes.WS_CLOSE, () => {
      // Do not destroy the 3D avatar renderer on transient WebSocket close;
      // WS auto-reconnects while the RTC session is active, and store/index.ts
      // explicitly calls exit() when the session ends.
      this._clearThinkingTimer()
    })
  }
}
