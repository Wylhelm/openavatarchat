import EventEmitter from 'eventemitter3'
import { WsEventTypes } from '../interface/eventType'

const KEEPALIVE_MS = 20000

export class WS extends EventEmitter {
  engine: WebSocket | undefined

  private _inited = false
  private _keepalive: ReturnType<typeof setInterval> | null = null

  constructor(url: string) {
    super()
    this._init(url)
  }
  private _init(url: string) {
    if (this._inited) {
      return
    }
    this._inited = true
    this.engine = new WebSocket(url)
    // Motion packets are parsed as Blobs. The default is already "blob";
    // set it explicitly so a runtime that defaults to "arraybuffer" cannot
    // change the decode path.
    this.engine.binaryType = 'blob'
    this.engine.addEventListener('error', (event) => {
      this.emit(WsEventTypes.WS_ERROR, event)
    })
    this.engine.addEventListener('open', () => {
      this._startKeepalive()
      this.emit(WsEventTypes.WS_OPEN)
    })
    this.engine.addEventListener('message', (event) => {
      this.emit(WsEventTypes.WS_MESSAGE, event.data)
    })
    this.engine.addEventListener('close', () => {
      this._stopKeepalive()
      this.emit(WsEventTypes.WS_CLOSE)
    })
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
    this.engine?.send(data)
  }
  stop() {
    this._stopKeepalive()
    this.emit(WsEventTypes.WS_CLOSE)
    this._inited = false
    this.engine?.close()
  }
}
