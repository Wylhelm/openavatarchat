import type EventEmitter from 'eventemitter3'
import { nanoid } from 'nanoid'

import { PlayerEventTypes } from '../interface/eventType'
interface IOption {
  // 传入的数据是采用多少位编码，默认16位
  channels: number
  // 缓存时间 单位 ms
  fftSize: number

  inputCodec: 'Int8' | 'Int16' | 'Int32' | 'Float32'
  // analyserNode fftSize
  onended: (extParams?: IExtInfo) => void
  // 采样率 单位Hz
  sampleRate: number
  // 是否静音
  isMute: boolean
}
interface ITypedArrays {
  Float32: typeof Float32Array
  Int16: typeof Int16Array
  Int32: typeof Int32Array
  Int8: typeof Int8Array
}
type IExtInfo = Record<string, unknown>
interface ISamples {
  data: Float32Array
  end_of_batch: boolean
  startTime?: number
}
interface IScheduledChunk {
  source: AudioBufferSourceNode
  envelope: GainNode
  end: number
}

// Speech arrives in one-second slices over the network. Playing a slice the
// moment it lands leaves no slack, so any late slice cuts the voice mid-word.
const PREBUFFER_SECONDS = 0.4
const REBUFFER_SECONDS = 0.25
const FADE_SECONDS = 0.006

const sharedContexts = new Map<number, AudioContext>()
const sharedOutputs = new WeakMap<AudioContext, AudioNode>()
let outputBoost = 1

// Out of the phone-call audio path the voice plays at media level, which is
// quieter. Boost it, with a limiter in front so peaks don't clip.
export function setPlayerOutputBoost(gain: number) {
  outputBoost = gain
}

function getOutput(ctx: AudioContext) {
  const existing = sharedOutputs.get(ctx)
  if (existing) return existing
  let output: AudioNode = ctx.destination
  if (outputBoost !== 1) {
    const limiter = ctx.createDynamicsCompressor()
    limiter.threshold.value = -20
    limiter.knee.value = 6
    limiter.ratio.value = 6
    limiter.attack.value = 0.003
    limiter.release.value = 0.2
    const makeup = ctx.createGain()
    makeup.gain.value = outputBoost
    limiter.connect(makeup)
    makeup.connect(ctx.destination)
    output = limiter
  }
  sharedOutputs.set(ctx, output)
  return output
}

function getAudioContext(sampleRate: number) {
  const existing = sharedContexts.get(sampleRate)
  if (existing && existing.state !== 'closed') return existing
  const AudioContextCtor =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext
  // The avatar renderer keeps a phone busy; the default interactive latency
  // leaves the audio device so little slack that it crackles under that load.
  let ctx: AudioContext
  try {
    ctx = new AudioContextCtor({ sampleRate, latencyHint: 'playback' })
  } catch {
    try {
      ctx = new AudioContextCtor({ latencyHint: 'playback' })
    } catch {
      ctx = new AudioContextCtor()
    }
  }
  sharedContexts.set(sampleRate, ctx)
  return ctx
}

export class Player {
  static isTypedArray(
    data: Int8Array | Int16Array | Int32Array | Float32Array,
  ) {
    // 检测输入的数据是否为 TypedArray 类型或 ArrayBuffer 类型
    return (
      (data.byteLength &&
        data.buffer &&
        data.buffer.constructor === ArrayBuffer) ||
      data.constructor === ArrayBuffer
    )
  }
  id = nanoid()
  analyserNode?: AnalyserNode
  audioCtx?: AudioContext
  // 是否自动播放
  autoPlay = true
  bufferSource?: AudioBufferSourceNode
  convertValue = 32768
  ee: EventEmitter
  gainNode?: GainNode
  option: IOption = {
    inputCodec: 'Int16', // 传入的数据是采用多少位编码，默认16位
    channels: 1, // 声道数
    sampleRate: 8000, // 采样率 单位Hz
    fftSize: 2048, // analyserNode fftSize
    onended: () => {},
    isMute: false,
  }
  samplesList: ISamples[] = []

  startTime?: number
  typedArray?:
    | typeof Int8Array
    | typeof Int16Array
    | typeof Int32Array
    | typeof Float32Array

  _firstStartRelativeTime?: number
  _firstStartAbsoluteTime?: number
  private cursor?: number
  private timeBase?: number
  private lastChunk?: IScheduledChunk
  private scheduled = new Set<AudioBufferSourceNode>()
  private destroyed = false

  constructor(option: IOption, ee: EventEmitter) {
    this.ee = ee
    this.init(option)
  }

  async continue() {
    await this.audioCtx!.resume()
  }
  destroy() {
    this.destroyed = true
    this.samplesList = []
    this.scheduled.forEach((source) => {
      source.onended = null
      try {
        source.stop()
      } catch {
        // already stopped
      }
    })
    this.scheduled.clear()
    this.lastChunk = undefined
    this.gainNode?.disconnect()
  }
  feed(audioOptions: {
    audio: Int8Array | Int16Array | Int32Array | Float32Array
    end_of_batch: boolean
  }) {
    let { audio } = audioOptions
    const { end_of_batch } = audioOptions
    if (!audio) {
      return
    }
    this._isSupported(audio)
    // 获取格式化后的buffer
    audio = this._getFormattedValue(audio)
    const data = new Float32Array(audio.length)
    data.set(audio, 0)
    const samples = {
      data,
      end_of_batch,
    }
    this.samplesList.push(samples)
    this.flush(samples, this.samplesList.length - 1)
  }
  flush(samples: ISamples, index: number) {
    if (!(samples && this.autoPlay && this.audioCtx) || this.destroyed) return
    const ctx = this.audioCtx
    void ctx.resume()
    const mono = this.mono(samples.data)
    const now = ctx.currentTime
    const first = this.timeBase === undefined
    const late = this.cursor === undefined || this.cursor < now + 0.02
    const startAt = late
      ? now + (first ? PREBUFFER_SECONDS : REBUFFER_SECONDS)
      : this.cursor!

    if (first) {
      this.timeBase = startAt
      this._firstStartAbsoluteTime = Date.now() + (startAt - now) * 1000
      this._firstStartRelativeTime = 0
    }
    // The mouth looks up its frame from this offset, so it must match the
    // moment the slice is actually heard, gaps included.
    samples.startTime = startAt - this.timeBase!
    if (first) this.ee.emit(PlayerEventTypes.Player_StartSpeaking, this)

    if (!mono.length) {
      if (samples.end_of_batch) this.finishAt(Math.max(startAt, this.cursor ?? now))
      return
    }
    const buffer = ctx.createBuffer(1, mono.length, this.option.sampleRate)
    buffer.getChannelData(0).set(mono)
    const source = ctx.createBufferSource()
    source.buffer = buffer
    const envelope = ctx.createGain()
    source.connect(envelope)
    envelope.connect(this.gainNode!)
    const end = startAt + buffer.duration
    const fade = Math.min(FADE_SECONDS, buffer.duration / 4)

    if (late) {
      envelope.gain.setValueAtTime(0, startAt)
      envelope.gain.linearRampToValueAtTime(1, startAt + fade)
    } else {
      envelope.gain.setValueAtTime(1, startAt)
      this.keepPreviousOpen()
    }
    // Fade out unless the next slice shows up in time to cancel it, so a late
    // slice leaves a soft pause instead of a click.
    envelope.gain.setValueAtTime(1, end - fade)
    envelope.gain.linearRampToValueAtTime(0, end)

    source.start(startAt)
    this.scheduled.add(source)
    this.lastChunk = { source, envelope, end }
    this.cursor = end
    source.onended = () => {
      this.scheduled.delete(source)
      envelope.disconnect()
      if (this.destroyed) return
      if (!samples.end_of_batch && index === this.samplesList.length - 1) {
        this.ee.emit(PlayerEventTypes.Player_WaitNextAudioClip)
      }
      if (samples.end_of_batch) this.option.onended()
    }
  }
  private keepPreviousOpen() {
    const previous = this.lastChunk
    if (!previous) return
    const fade = Math.min(FADE_SECONDS, (previous.source.buffer?.duration ?? 0) / 4)
    const gain = previous.envelope.gain
    gain.cancelScheduledValues(previous.end - fade)
    gain.setValueAtTime(1, previous.end - fade)
  }
  private finishAt(time: number) {
    const ctx = this.audioCtx!
    const delay = Math.max(0, time - ctx.currentTime) * 1000
    setTimeout(() => {
      if (!this.destroyed) this.option.onended()
    }, delay)
  }
  private mono(data: Float32Array) {
    if (this.option.channels <= 1) return data
    const length = data.length / this.option.channels
    const out = new Float32Array(length)
    for (let i = 0; i < length; i++) out[i] = data[i * this.option.channels]
    return out
  }
  init(option: IOption) {
    this.option = Object.assign(this.option, option) // 实例最终配置参数
    this.convertValue = this._getConvertValue()
    this.typedArray = this._getTypedArray()
    this.initAudioContext()
  }
  initAudioContext() {
    this.audioCtx = getAudioContext(this.option.sampleRate)
    this.gainNode = this.audioCtx.createGain()
    this.gainNode.gain.value = this.option.isMute ? 0 : 1
    this.gainNode.connect(getOutput(this.audioCtx))
    this.startTime = this.audioCtx.currentTime
    this.analyserNode = this.audioCtx.createAnalyser()
    this.analyserNode.fftSize = this.option.fftSize
  }
  setMute(isMute: boolean) {
    this.gainNode!.gain.value = isMute ? 0 : 1
  }
  async pause() {
    await this.audioCtx!.suspend()
  }
  async updateAutoPlay(value: boolean) {
    if (this.autoPlay !== value && value) {
      this.autoPlay = value
      this.samplesList.forEach((sample, index) => {
        this.flush(sample, index)
      })
    } else {
      this.autoPlay = value
    }
  }

  volume(volume: number) {
    this.gainNode!.gain.value = volume
  }
  _getFormattedValue(data: Int8Array | Int16Array | Int32Array | Float32Array) {
    const TargetArray = this.typedArray!
    if (data.constructor === ArrayBuffer) {
      data = new TargetArray(data)
    } else {
      data = new TargetArray(data.buffer)
    }

    const float32 = new Float32Array(data.length)

    for (let i = 0; i < data.length; i++) {
      // buffer 缓冲区的数据，需要是IEEE754 里32位的线性PCM，范围从-1到+1
      // 所以对数据进行除法
      // 除以对应的位数范围，得到-1到+1的数据
      // float32[i] = data[i] / 0x8000;
      float32[i] = data[i] / this.convertValue
    }
    return float32
  }

  private _isSupported(
    data: Int8Array | Int16Array | Int32Array | Float32Array,
  ) {
    // 数据类型是否支持
    // 目前支持 ArrayBuffer 或者 TypedArray
    if (!Player.isTypedArray(data))
      throw new Error('请传入ArrayBuffer或者任意TypedArray')
    return true
  }

  private _getConvertValue() {
    // 根据传入的目标编码位数
    // 选定转换数据所需要的基本值
    const inputCodecs = {
      Int8: 128,
      Int16: 32768,
      Int32: 2147483648,
      Float32: 1,
    }
    if (!inputCodecs[this.option.inputCodec])
      throw new Error(
        'wrong codec.please input one of these codecs:Int8,Int16,Int32,Float32',
      )
    return inputCodecs[this.option.inputCodec]
  }

  private _getTypedArray() {
    // 根据传入的目标编码位数
    // 选定前端的所需要的保存的二进制数据格式
    // 完整TypedArray请看文档
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/TypedArray
    const typedArrays: ITypedArrays = {
      Int8: Int8Array,
      Int16: Int16Array,
      Int32: Int32Array,
      Float32: Float32Array,
    }
    if (!typedArrays[this.option.inputCodec])
      throw new Error(
        'wrong codec.please input one of these codecs:Int8,Int16,Int32,Float32',
      )
    return typedArrays[this.option.inputCodec]
  }
}
