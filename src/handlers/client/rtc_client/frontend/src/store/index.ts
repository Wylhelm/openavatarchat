import { setPlayerOutputBoost } from '@/helpers/player'
import { isCoarsePointer, noEchoCancel } from '@/utils/audioMode'
import { WS } from '@/helpers/ws'
import { EventTypes, WsEventTypes } from '@/interface/eventType'
import { StreamState, TYVoiceChatState } from '@/interface/voiceChat'
import { GaussianAvatar } from '@/utils/gaussianAvatar'
import {
  createSimulatedAudioTrack,
  createSimulatedVideoTrack,
  getDevices,
  getStream,
  setAvailableDevices,
} from '@/utils/streamUtils'
import { setupWebRTC, stop } from '@/utils/webrtcUtils'
import { message } from 'ant-design-vue'
import { defineStore } from 'pinia'
import { useVisionStore } from './vision'

if (noEchoCancel) setPlayerOutputBoost(2.5)

const track_constraints = {
  video: {
    width: 500,
    height: 500,
  },
  audio: noEchoCancel
    ? { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    : true,
}

// "disconnected" is a transient ICE state. Closing immediately dropped the
// avatar during a long call whenever the link blipped.
const DISCONNECT_GRACE_MS = 20000
let disconnectGraceTimer: ReturnType<typeof setTimeout> | null = null

function clearDisconnectGrace() {
  if (disconnectGraceTimer !== null) {
    clearTimeout(disconnectGraceTimer)
    disconnectGraceTimer = null
  }
}

interface VideoChatState {
  devices: MediaDeviceInfo[]
  availableVideoDevices: MediaDeviceInfo[]
  availableAudioDevices: MediaDeviceInfo[]
  selectedVideoDevice: MediaDeviceInfo | null
  selectedAudioDevice: MediaDeviceInfo | null
  streamState: StreamState
  stream: MediaStream | null
  peerConnection: RTCPeerConnection | null
  localStream: MediaStream | null
  webcamAccessed: boolean
  webRTCId: string

  avatarType: '' | 'lam'
  avatarWSRoute: string
  avatarAssetsPath: string
  avatarScale: number
  rtcConfig: RTCConfiguration | undefined
  trackConstraints:
    | {
        video: MediaTrackConstraints | boolean
        audio: MediaTrackConstraints | boolean
      }
    | undefined
  gsLoadPercent: number

  volumeMuted: boolean
  micMuted: boolean
  cameraOff: boolean

  hasCamera: boolean
  hasCameraPermission: boolean
  hasMic: boolean
  hasMicPermission: boolean
  showChatRecords: boolean

  localAvatarRenderer: any

  chatDataChannel: RTCDataChannel | null
  replying: boolean
  chatRecords: Array<{ id: string; role: 'human' | 'avatar'; message: string }>
}

export const useVideoChatStore = defineStore('videoChatStore', {
  state: (): VideoChatState => {
    return {
      devices: [],
      availableVideoDevices: [],
      availableAudioDevices: [],
      selectedVideoDevice: null,
      selectedAudioDevice: null,
      streamState: StreamState.closed,
      stream: null,
      peerConnection: null,
      localStream: null,
      webRTCId: '',
      webcamAccessed: false,
      avatarType: '',
      avatarWSRoute: '',
      avatarAssetsPath: '',
      avatarScale: 1,
      rtcConfig: undefined,
      trackConstraints: track_constraints,
      gsLoadPercent: 0,
      volumeMuted: false,
      micMuted: false,
      cameraOff: false,
      hasCamera: false,
      hasCameraPermission: true,
      hasMic: false,
      hasMicPermission: true,
      showChatRecords: false,

      localAvatarRenderer: null,

      chatDataChannel: null,
      replying: false,
      chatRecords: [],
    }
  },
  getters: {},
  actions: {
    async accessDevice() {
      try {
        this.micMuted = false
        this.cameraOff = false
        this.volumeMuted = false
        if (!navigator.mediaDevices) {
          message.error('无法获取媒体设备，请确保用localhost访问或https协议访问')
          return
        }
        const devices = await getDevices().catch(() => [] as MediaDeviceInfo[])
        this.devices = devices
        const videoDeviceId =
          this.selectedVideoDevice &&
          devices.some((device) => device.deviceId && device.deviceId === this.selectedVideoDevice?.deviceId)
            ? this.selectedVideoDevice.deviceId
            : ''
        const audioDeviceId =
          this.selectedAudioDevice &&
          devices.some((device) => device.deviceId && device.deviceId === this.selectedAudioDevice?.deviceId)
            ? this.selectedAudioDevice.deviceId
            : ''
        await this.fillStream(audioDeviceId, videoDeviceId)
        this.webcamAccessed = true
      } catch (err: any) {
        console.log(err)
        message.error(err)
      }
    },
    async init() {
      fetch('/openavatarchat/initconfig')
        .then((res) => res.json())
        .then((config) => {
          if (config.rtc_configuration) {
            this.rtcConfig = config.rtc_configuration
          }

          console.log(config)
          if (config.avatar_config) {
            this.avatarType = config.avatar_config.avatar_type
            this.avatarWSRoute = config.avatar_config.avatar_ws_route
            this.avatarAssetsPath = config.avatar_config.avatar_assets_path
            const scale = Number(config.avatar_config.avatar_scale)
            this.avatarScale = scale > 0 ? scale : 1
          }
          if (config.track_constraints && !noEchoCancel) {
            this.trackConstraints = config.track_constraints
          }
        })
        .catch(() => {
          message.error('服务端链接失败，请检查是否能正确访问到 OpenAvatarChat 服务端')
        })
    },
    handleCameraOff() {
      this.cameraOff = !this.cameraOff
      this.stream?.getTracks().forEach((track) => {
        if (track.kind.includes('video')) track.enabled = !this.cameraOff
      })
    },
    handleMicMuted() {
      this.micMuted = !this.micMuted
      this.stream?.getTracks().forEach((track) => {
        if (track.kind.includes('audio')) track.enabled = !this.micMuted
      })
    },
    handleVolumeMute() {
      this.volumeMuted = !this.volumeMuted
      if (this.avatarType === 'lam') {
        this.localAvatarRenderer?.setAvatarMute(this.volumeMuted)
      }
    },
    async handleDeviceChange(deviceId: string) {
      const device_id = deviceId
      const devices = await getDevices()
      this.devices = devices
      console.log('🚀 ~ handle_device_change ~ devices:', devices)
      let videoDeviceId =
        this.selectedVideoDevice &&
        devices.some((device) => device.deviceId === this.selectedVideoDevice?.deviceId)
          ? this.selectedVideoDevice.deviceId
          : ''
      let audioDeviceId =
        this.selectedAudioDevice &&
        devices.some((device) => device.deviceId === this.selectedAudioDevice?.deviceId)
          ? this.selectedAudioDevice.deviceId
          : ''
      if (this.availableVideoDevices.find((video_device) => video_device.deviceId === device_id)) {
        videoDeviceId = device_id
        this.cameraOff = false
      } else if (
        this.availableAudioDevices.find((audio_device) => audio_device.deviceId === device_id)
      ) {
        audioDeviceId = device_id
        this.micMuted = false
      }
      this.fillStream(audioDeviceId, videoDeviceId)
    },
    handleSubtitleToggle() {
      this.showChatRecords = !this.showChatRecords
      const visionState = useVisionStore()
      const { wrapperRef, wrapperRect } = visionState
      console.log(wrapperRect, wrapperRef)
      if (!wrapperRef || !wrapperRect) return
      wrapperRef.getBoundingClientRect()
      wrapperRect.width = wrapperRef!.clientWidth
      wrapperRect.height = wrapperRef!.clientHeight
      visionState.isLandscape = wrapperRect.width > wrapperRect.height
    },
    async updateAvailableDevices() {
      const devices = await getDevices()
      this.availableVideoDevices = setAvailableDevices(devices, 'videoinput')
      this.availableAudioDevices = setAvailableDevices(devices, 'audioinput')
    },
    async fillStream(audioDeviceId: string, videoDeviceId: string) {
      const visionState = useVisionStore()
      const node = visionState.localVideoRef
      if (this.stream) {
        this.stream.getTracks().forEach((t) => t.stop())
        this.stream = null
      }
      const wantAudio =
        audioDeviceId && audioDeviceId !== 'default'
          ? { deviceId: { exact: audioDeviceId } }
          : this.hasMicPermission
      const wantVideo =
        videoDeviceId && videoDeviceId !== 'default'
          ? { deviceId: { exact: videoDeviceId } }
          : this.hasCameraPermission

      let local_stream: MediaStream | null = null
      try {
        local_stream = await getStream(wantAudio, wantVideo, this.trackConstraints)
      } catch {
        // Fallback: try audio-only (e.g. desktop without webcam), then video-only
        try {
          local_stream = await getStream(wantAudio, false, this.trackConstraints)
          this.hasCameraPermission = false
        } catch {
          this.hasMicPermission = false
          try {
            local_stream = await getStream(false, wantVideo, this.trackConstraints)
          } catch (e) {
            this.hasCameraPermission = false
            console.error('image.no_webcam_support', e)
          }
        }
      }

      if (local_stream) {
        this.stream = local_stream
        const devices = await getDevices().catch(() => [] as MediaDeviceInfo[])
        this.devices = devices
        await this.updateAvailableDevices()
        this.hasMic =
          this.stream.getAudioTracks().length > 0 && this.hasMicPermission
        this.hasCamera =
          this.stream.getVideoTracks().length > 0 && this.hasCameraPermission
        const used_devices = this.stream
          .getTracks()
          .map((track) => track.getSettings()?.deviceId)
        used_devices.forEach((device_id) => {
          const used_device = devices.find((device) => device.deviceId === device_id)
          if (used_device && used_device?.kind.includes('video')) {
            this.selectedVideoDevice = used_device
          } else if (used_device && used_device?.kind.includes('audio')) {
            this.selectedAudioDevice = used_device
          }
        })
        !this.selectedVideoDevice && (this.selectedVideoDevice = this.availableVideoDevices[0])
      }

      if (!this.stream) {
        this.stream = new MediaStream()
      }
      if (!this.stream.getTracks().find((item) => item.kind === 'audio')) {
        this.stream.addTrack(createSimulatedAudioTrack())
      }
      if (!this.stream.getTracks().find((item) => item.kind === 'video')) {
        this.stream.addTrack(createSimulatedVideoTrack())
      }
      this.webcamAccessed = true
      this.localStream = this.stream
      if (node) {
        node.srcObject = this.localStream
        node.muted = true
        node?.play()
        this.localAvatarRenderer?.setTrackingVideoElement?.(node)
      }
    },
    armDisconnectGrace() {
      if (disconnectGraceTimer !== null) return
      disconnectGraceTimer = setTimeout(() => {
        disconnectGraceTimer = null
        if (this.peerConnection?.connectionState === 'disconnected') {
          console.warn('WebRTC stayed disconnected, closing the avatar session')
          this.endLiveSession()
        }
      }, DISCONNECT_GRACE_MS)
    },
    endLiveSession() {
      clearDisconnectGrace()
      const pc = this.peerConnection
      if (!pc && this.streamState === StreamState.closed) {
        return
      }
      this.peerConnection = null
      this.streamState = StreamState.closed
      this.chatDataChannel = null
      this.replying = false
      if (this.avatarType === 'lam') {
        this.localAvatarRenderer?.exit()
        this.localAvatarRenderer = null
        this.gsLoadPercent = 0
      }
      if (pc && pc.connectionState !== 'closed') {
        stop(pc)
      }
    },
    async startWebRTC() {
      const visionState = useVisionStore()
      if (this.streamState === 'closed') {
        this.chatRecords = []
        this.peerConnection = new RTCPeerConnection(this.rtcConfig)
        this.peerConnection.addEventListener('connectionstatechange', () => {
          const pc = this.peerConnection
          if (!pc) return
          switch (pc.connectionState) {
            case 'connected':
              clearDisconnectGrace()
              this.streamState = StreamState.open
              break
            case 'disconnected':
              console.warn('WebRTC disconnected, waiting to see if it recovers')
              this.armDisconnectGrace()
              break
            case 'failed':
            case 'closed':
              this.endLiveSession()
              break
            default:
              break
          }
        })
        this.streamState = StreamState.waiting
        await setupWebRTC(this.stream!, this.peerConnection!, visionState.remoteVideoRef!)
          .then(([dataChannel, webRTCId]) => {
            this.streamState = StreamState.open
            this.webRTCId = webRTCId as string
            this.chatDataChannel = dataChannel as any

            if (this.avatarType && this.avatarWSRoute) {
              // Delay WS motion connection to allow server to register session delegate
              // after WebRTC SDP negotiation (race condition fix)
              const wsRoute = this.avatarWSRoute
              const webRTCId = this.webRTCId
              const avatarType = this.avatarType
              setTimeout(() => {
                const ws = this.initWebsocket(wsRoute, webRTCId)
                if (avatarType === 'lam') {
                  this.localAvatarRenderer = this.doGaussianRender(ws)
                }
              }, 2000)
            }
          })
          .catch((e) => {
            console.info('catching', e)
            this.streamState = StreamState.closed
            message.error(e)
            message.error('请检查是否超过数字人并发上限')
          })
      } else if (this.streamState === 'waiting') {
        // waiting 中不允许操作
      } else {
        this.endLiveSession()
        this.chatRecords = []
        await this.accessDevice()
      }
    },
    initWebsocket(ws_route: string, webRTCId: string) {
      const ws = new WS(
        `${window.location.protocol.includes('https') ? 'wss' : 'ws'}://${window.location.host}${ws_route}/${webRTCId}`
      )
      ws.on(WsEventTypes.WS_OPEN, () => {
        console.log('socket opened')
        ws.send(
          JSON.stringify({
            header: { name: 'ClientAudioMode' },
            payload: { no_aec: noEchoCancel },
          })
        )
      })
      ws.on(WsEventTypes.WS_CLOSE, () => {
        console.log('socket closed')
      })
      ws.on(WsEventTypes.WS_ERROR, (event) => {
        console.log('socket error', event)
      })
      ws.on(WsEventTypes.WS_MESSAGE, (data) => {
        console.log('socket on message', data)
      })
      return ws
    },
    doGaussianRender(ws: WS) {
      const visionState = useVisionStore()
      const gaussianAvatar = new GaussianAvatar({
        container: visionState.remoteVideoContainerRef!,
        assetsPath: this.avatarAssetsPath,
        avatarScale: this.avatarScale,
        trackingVideoElement: visionState.localVideoRef,
        ws,
        loadProgress: (progress) => {
          console.log('gs loadProgress', progress)
          this.gsLoadPercent = progress
          if (progress >= 1) {
            // visionState.computeRemotePosition();
          }
        },
      })
      gaussianAvatar.start()
      return gaussianAvatar
    },
  },
})
