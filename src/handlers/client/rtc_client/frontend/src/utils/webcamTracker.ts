// Lightweight client-side webcam user tracker (~12 Hz) that detects the user's
// horizontal and vertical position in the local video feed so the 3D Gaussian
// Splatting avatar can follow the user with her eyes, head, and neck.
//
// Uses an 8x6 spatial block grid (each block averages 8x8 pixels, reducing
// camera sensor noise by 8x) plus native FaceDetector when available, with a
// strict hysteresis deadband so the target stays 100% still when the user is at rest.

export interface UserTargetPoint {
  // Normalized user offset in [-1, 1]:
  // x > 0 means user is to the right side of the user's screen (left in raw camera image)
  // y > 0 means user is higher up in the frame, y < 0 means lower
  x: number
  y: number
  confidence: number
  nodding?: boolean
  smile?: number
}

const SAMPLE_W = 64
const SAMPLE_H = 48
const BLOCK_SIZE = 8
const GRID_W = SAMPLE_W / BLOCK_SIZE // 8
const GRID_H = SAMPLE_H / BLOCK_SIZE // 6
const TRACK_INTERVAL_MS = 80 // ~12.5 Hz
const HOLD_TIMEOUT_MS = 6000
const NOD_WINDOW_MS = 900
const NOD_COOLDOWN_MS = 3500
const POSITION_DEADBAND_X = 0.055
const POSITION_DEADBAND_Y = 0.055

export function computeSkinAndMotionCentroid(
  rgba: Uint8ClampedArray,
  prevLuma: Float32Array | null,
  width: number,
  height: number
): { x: number; y: number; confidence: number; smile: number; moving: boolean } {
  const bw = Math.max(1, Math.floor(width / GRID_W))
  const bh = Math.max(1, Math.floor(height / GRID_H))
  const blockLuma = new Float32Array(GRID_W * GRID_H)
  const blockSkin = new Float32Array(GRID_W * GRID_H)
  const blockDiff = new Float32Array(GRID_W * GRID_H)

  let totalSkinPixels = 0

  for (let by = 0; by < GRID_H; by++) {
    for (let bx = 0; bx < GRID_W; bx++) {
      let lumaSum = 0
      let skinCount = 0
      const count = bw * bh
      const y0 = by * bh
      const x0 = bx * bw

      for (let py = y0; py < y0 + bh; py++) {
        for (let px = x0; px < x0 + bw; px++) {
          const idx = (py * width + px) * 4
          const r = rgba[idx]
          const g = rgba[idx + 1]
          const b = rgba[idx + 2]
          const luma = 0.299 * r + 0.587 * g + 0.114 * b
          lumaSum += luma

          const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
          const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
          const isSkin =
            luma > 35 &&
            luma < 245 &&
            r > 55 &&
            g > 35 &&
            b > 20 &&
            r > g &&
            r > b &&
            r - Math.min(g, b) > 12 &&
            cr >= 133 &&
            cr <= 178 &&
            cb >= 77 &&
            cb <= 132

          if (isSkin) {
            skinCount++
            totalSkinPixels++
          }
        }
      }

      const bIdx = by * GRID_W + bx
      const avgLuma = lumaSum / count
      blockLuma[bIdx] = avgLuma
      blockSkin[bIdx] = skinCount / count

      if (prevLuma) {
        // Store block average luma in the first GRID_W * GRID_H entries of prevLuma
        const prev = prevLuma[bIdx]
        prevLuma[bIdx] = avgLuma
        if (prev > 0) {
          blockDiff[bIdx] = Math.abs(avgLuma - prev)
        }
      }
    }
  }

  // Coherent block motion: only blocks whose 8x8 average luma shifted by > 5.5
  let movingBlocks = 0
  let motionSumX = 0
  let motionSumY = 0
  let motionWeight = 0

  for (let by = 0; by < GRID_H; by++) {
    const vBias = by < GRID_H - 1 ? 1.0 : 0.4
    for (let bx = 0; bx < GRID_W; bx++) {
      const bIdx = by * GRID_W + bx
      const diff = blockDiff[bIdx]
      if (diff > 5.5) {
        movingBlocks++
        const skinBoost = 0.7 + blockSkin[bIdx] * 1.3
        const w = (diff - 5.0) * skinBoost * vBias
        motionSumX += (bx + 0.5) * w
        motionSumY += (by + 0.5) * w
        motionWeight += w
      }
    }
  }

  // Coherent movement requires at least 2 blocks (128 pixels) shifting together
  if (movingBlocks >= 2 && motionWeight > 0) {
    const cx = motionSumX / motionWeight / GRID_W
    const cy = motionSumY / motionWeight / GRID_H
    const normX = Math.max(-1, Math.min(1, (0.5 - cx) * 2.3))
    const normY = Math.max(-1, Math.min(1, (0.46 - cy) * 2.0))
    const confidence = Math.min(1, movingBlocks / 6)
    return { x: normX, y: normY, confidence, smile: 0, moving: true }
  }

  // Static fallback: find the strongest contiguous skin cluster (peak block + its 3x3 neighborhood)
  let bestBx = -1
  let bestBy = -1
  let bestScore = 0
  for (let by = 0; by < GRID_H - 1; by++) {
    for (let bx = 0; bx < GRID_W; bx++) {
      const s = blockSkin[by * GRID_W + bx]
      if (s > bestScore) {
        bestScore = s
        bestBx = bx
        bestBy = by
      }
    }
  }

  const area = width * height
  const skinRatio = totalSkinPixels / area
  if (bestScore < 0.2 || skinRatio < 0.012) {
    return { x: 0, y: 0, confidence: 0, smile: 0, moving: false }
  }

  // Compute centroid only within the 3x3 neighborhood of the peak skin block
  // so distant warm-colored walls cannot dilute the face position
  let clusterSumX = 0
  let clusterSumY = 0
  let clusterWeight = 0
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const nx = bestBx + dx
      const ny = bestBy + dy
      if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
        const w = blockSkin[ny * GRID_W + nx]
        if (w > 0.1) {
          clusterSumX += (nx + 0.5) * w
          clusterSumY += (ny + 0.5) * w
          clusterWeight += w
        }
      }
    }
  }

  if (clusterWeight <= 0) {
    return { x: 0, y: 0, confidence: 0, smile: 0, moving: false }
  }

  const cx = clusterSumX / clusterWeight / GRID_W
  const cy = clusterSumY / clusterWeight / GRID_H
  const normX = Math.max(-1, Math.min(1, (0.5 - cx) * 2.3))
  const normY = Math.max(-1, Math.min(1, (0.46 - cy) * 2.0))
  const confidence = Math.min(1, bestScore * 0.85)
  return { x: normX, y: normY, confidence, smile: 0, moving: false }
}

export function detectNodGesture(
  samples: Array<{ t: number; x: number; y: number }>
): boolean {
  if (samples.length < 6) return false
  let minX = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  let reversals = 0
  let prevDySign = 0

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]
    if (s.x < minX) minX = s.x
    if (s.x > maxX) maxX = s.x
    if (s.y < minY) minY = s.y
    if (s.y > maxY) maxY = s.y
    if (i > 0) {
      const dy = s.y - samples[i - 1].y
      if (Math.abs(dy) > 0.022) {
        const sign = dy > 0 ? 1 : -1
        if (prevDySign !== 0 && sign !== prevDySign) {
          reversals++
        }
        prevDySign = sign
      }
    }
  }

  const spanX = maxX - minX
  const spanY = maxY - minY
  return spanY >= 0.085 && spanY <= 0.45 && spanY > spanX * 1.6 && reversals >= 2
}

export class WebcamFaceTracker {
  private videoEl: HTMLVideoElement | null = null
  private timer: ReturnType<typeof setInterval> | null = null
  private canvas: HTMLCanvasElement | null = null
  private ctx: CanvasRenderingContext2D | null = null
  private prevLuma: Float32Array | null = null
  private nativeDetector: any = null
  private nativeFailed = false
  private busy = false
  private lastSeenAt = 0
  private lastNodAt = 0
  private lastX = 0
  private lastY = 0
  private hasInitialLock = false
  private history: Array<{ t: number; x: number; y: number }> = []
  private onUpdate: (target: UserTargetPoint | null) => void

  constructor(onUpdate: (target: UserTargetPoint | null) => void) {
    this.onUpdate = onUpdate
    if (typeof window !== 'undefined' && 'FaceDetector' in window) {
      try {
        const FaceDetectorCtor = (window as any).FaceDetector
        this.nativeDetector = new FaceDetectorCtor({ fastMode: true, maxDetectedFaces: 1 })
      } catch {
        this.nativeDetector = null
      }
    }
  }

  start(videoEl: HTMLVideoElement | null | undefined) {
    this.stop()
    if (!videoEl || typeof document === 'undefined') return
    this.videoEl = videoEl
    this.canvas = document.createElement('canvas')
    this.canvas.width = SAMPLE_W
    this.canvas.height = SAMPLE_H
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })
    this.prevLuma = new Float32Array(SAMPLE_W * SAMPLE_H)
    this.history = []
    this.hasInitialLock = false

    this.timer = setInterval(() => {
      void this.sampleFrame()
    }, TRACK_INTERVAL_MS)
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer)
      this.timer = null
    }
    this.videoEl = null
    this.busy = false
    this.history = []
    this.hasInitialLock = false
    this.onUpdate(null)
  }

  private updateSmoothedTarget(now: number, rawX: number, rawY: number, confidence: number) {
    if (!this.hasInitialLock) {
      this.lastX = rawX
      this.lastY = rawY
      this.hasInitialLock = true
    } else {
      const dx = rawX - this.lastX
      const dy = rawY - this.lastY
      // Hysteresis deadband: ignore tiny fluctuations so the avatar is 100% still at rest
      if (Math.abs(dx) > POSITION_DEADBAND_X) {
        this.lastX += dx * 0.35
      }
      if (Math.abs(dy) > POSITION_DEADBAND_Y) {
        this.lastY += dy * 0.3
      }
    }
    this.lastSeenAt = now
    const nodding = this.recordAndCheckNod(now, this.lastX, this.lastY)
    this.onUpdate({
      x: this.lastX,
      y: this.lastY,
      confidence,
      nodding,
      smile: 0,
    })
  }

  private recordAndCheckNod(now: number, x: number, y: number): boolean {
    this.history.push({ t: now, x, y })
    const cutoff = now - NOD_WINDOW_MS
    while (this.history.length > 0 && this.history[0].t < cutoff) {
      this.history.shift()
    }
    if (now - this.lastNodAt < NOD_COOLDOWN_MS) {
      return false
    }
    if (detectNodGesture(this.history)) {
      this.lastNodAt = now
      this.history = []
      return true
    }
    return false
  }

  private async sampleFrame() {
    if (this.busy) return
    const video = this.videoEl
    if (
      !video ||
      video.paused ||
      video.ended ||
      video.readyState < 2 ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      this.checkHoldTimeout()
      return
    }

    const mediaStream = video.srcObject as MediaStream | null
    const videoTrack = mediaStream?.getVideoTracks?.()?.[0]
    if (videoTrack && !videoTrack.enabled) {
      this.onUpdate(null)
      return
    }

    this.busy = true
    try {
      const now = performance.now()
      // 1. Try hardware/native FaceDetector if supported by the browser
      if (this.nativeDetector && !this.nativeFailed) {
        try {
          const faces = await this.nativeDetector.detect(video)
          if (Array.isArray(faces) && faces.length > 0) {
            const box = faces[0].boundingBox
            const cx = (box.x + box.width * 0.5) / video.videoWidth
            const cy = (box.y + box.height * 0.42) / video.videoHeight
            const normX = Math.max(-1, Math.min(1, (0.5 - cx) * 2.2))
            const normY = Math.max(-1, Math.min(1, (0.46 - cy) * 2.0))
            this.updateSmoothedTarget(now, normX, normY, 1)
            return
          }
        } catch {
          this.nativeFailed = true
        }
      }

      // 2. Downsampled 8x6 block motion + local skin cluster tracker
      if (!this.ctx) return
      this.ctx.drawImage(video, 0, 0, SAMPLE_W, SAMPLE_H)
      const img = this.ctx.getImageData(0, 0, SAMPLE_W, SAMPLE_H)
      const res = computeSkinAndMotionCentroid(img.data, this.prevLuma, SAMPLE_W, SAMPLE_H)
      // Update target on coherent motion OR strong skin cluster presence (e.g. mobile
      // where the user tilts/shifts the phone or moves smoothly across the camera frame).
      if ((res.moving && res.confidence > 0.22) || res.confidence > 0.35) {
        this.updateSmoothedTarget(now, res.x, res.y, res.confidence)
      } else {
        this.checkHoldTimeout()
      }
    } catch {
      // Ignore transient canvas/video read errors
    } finally {
      this.busy = false
    }
  }

  private checkHoldTimeout() {
    if (this.lastSeenAt > 0 && performance.now() - this.lastSeenAt < HOLD_TIMEOUT_MS) {
      // Hold last known position steadily while the user sits still
      this.onUpdate({
        x: this.lastX,
        y: this.lastY,
        confidence: 0.5,
        nodding: false,
        smile: 0,
      })
    } else {
      this.onUpdate(null)
    }
  }
}
