import EventEmitter from 'eventemitter3'
import { WsEventTypes } from '../interface/eventType'

const KEEPALIVE_MS = 20000
const RECONNECT_DELAY_MS = 1500

export class WS extends EventEmitter {
  engine: WebSocket | undefined

  private _inited = false
  private _stopped = false
  private _url = ''
  private _keepalive: ReturnType<typeof setInterval> | null = null
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null

  constructor(url: string) {
    super()
    this._url = url
    this._init(url)
  }
  private _init(url: string) {
    if (this._inited || this._stopped) {
      return
    }
    this._inited = true
    this._clearReconnectTimer()
    const ws = new WebSocket(url)
    this.engine = ws
    // Motion packets are parsed as Blobs. The default is already "blob";
    // set it explicitly so a runtime that defaults to "arraybuffer" cannot
    // change the decode path.
    ws.binaryType = 'blob'
    ws.addEventListener('error', (event) => {
      if (this.engine !== ws) return
      this.emit(WsEventTypes.WS_ERROR, event)
    })
    ws.addEventListener('open', () => {
      if (this.engine !== ws) return
      this._startKeepalive()
      this.emit(WsEventTypes.WS_OPEN)
    })
    ws.addEventListener('message', (event) => {
      if (this.engine !== ws) return
      this.emit(WsEventTypes.WS_MESSAGE, event.data)
    })
    ws.addEventListener('close', () => {
      if (this.engine !== ws) return
      this._stopKeepalive()
      this._inited = false
      if (this._stopped) {
        this.emit(WsEventTypes.WS_CLOSE)
      } else {
        this._scheduleReconnect()
      }
    })
  }
  private _scheduleReconnect() {
    if (this._stopped || this._reconnectTimer || !this._url) {
      return
    }
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null
      if (!this._stopped) {
        this._init(this._url)
      }
    }, RECONNECT_DELAY_MS)
  }
  private _clearReconnectTimer() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer)
      this._reconnectTimer = null
    }
  }
  private _startKeepalive() {
    this._stopKeepalive()
    this._keepalive = setInterval(() => {
      if (this.engine?.readyState === WebSocket.OPEN) {
        this.engine.send(JSON.stringify({ header: { name: 'Ping' } }))
      }
    }, KEEPALIVE_MS)
  }
  private _stopKeepalive() {
    if (this._keepalive) {
      clearInterval(this._keepalive)
      this._keepalive = null
    }
  }
  send(data: string | Int8Array | Uint8Array) {
    if (this.engine?.readyState === WebSocket.OPEN) {
      this.engine.send(data)
    }
  }
  stop() {
    this._stopped = true
    this._clearReconnectTimer()
    this._stopKeepalive()
    this.emit(WsEventTypes.WS_CLOSE)
    this._inited = false
    const ws = this.engine
    this.engine = undefined
    ws?.close()
  }
}
