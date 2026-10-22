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
const TRACK_INTERVAL_MS = 66 // ~15 Hz
const HOLD_TIMEOUT_MS = 6000
const NOD_WINDOW_MS = 900
const NOD_COOLDOWN_MS = 3500
const POSITION_DEADBAND_X = 0.045
const POSITION_DEADBAND_Y = 0.045

export function computeSkinAndMotionCentroid(
  rgba: Uint8ClampedArray,
  prevLuma: Float32Array | null,
  width: number,
  height: number,
  priorX?: number,
  priorY?: number
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
  let skinMovingBlocks = 0
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
        if (blockSkin[bIdx] >= 0.15) {
          skinMovingBlocks++
        }
        const skinBoost = 0.5 + blockSkin[bIdx] * 1.8
        const w = (diff - 5.0) * skinBoost * vBias
        motionSumX += (bx + 0.5) * w
        motionSumY += (by + 0.5) * w
        motionWeight += w
      }
    }
  }

  // Coherent movement: between 2 and 18 blocks shifting together (if >18 blocks move,
  // it's global handheld camera shake or auto-exposure shift on mobile, so use the skin cluster).
  if (
    movingBlocks >= 2 &&
    movingBlocks <= 18 &&
    (skinMovingBlocks >= 1 || totalSkinPixels < (width * height) * 0.015) &&
    motionWeight > 0
  ) {
    const cx = motionSumX / motionWeight / GRID_W
    const cy = motionSumY / motionWeight / GRID_H
    const normX = Math.max(-1, Math.min(1, (0.5 - cx) * 2.15))
    const normY = Math.max(-0.9, Math.min(0.9, (0.46 - cy) * 1.85))
    const confidence = Math.min(1, movingBlocks / 6)
    return { x: normX, y: normY, confidence, smile: 0, moving: true }
  }

  // Static fallback: find the strongest 3x3 neighborhood skin cluster, with temporal continuity
  // toward the previous tracked block so the peak never hops left-right between two blocks ("head shake").
  const priorBx =
    priorX !== undefined && Number.isFinite(priorX)
      ? (0.5 - priorX / 2.1) * GRID_W - 0.5
      : null
  const priorBy =
    priorY !== undefined && Number.isFinite(priorY)
      ? (0.46 - priorY / 1.8) * GRID_H - 0.5
      : null

  let bestBx = -1
  let bestBy = -1
  let bestClusterScore = 0
  let bestPeakSkin = 0
  for (let by = 0; by < GRID_H - 1; by++) {
    for (let bx = 0; bx < GRID_W; bx++) {
      const rawSkin = blockSkin[by * GRID_W + bx]
      if (rawSkin < 0.18) continue
      let neighborhoodSum = 0
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = bx + dx
          const ny = by + dy
          if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
            const weight = dx === 0 && dy === 0 ? 1.5 : 0.65
            neighborhoodSum += blockSkin[ny * GRID_W + nx] * weight
          }
        }
      }
      const edgePenalty = bx === 0 || bx === GRID_W - 1 ? 0.78 : 1.0
      let temporalBonus = 1.0
      if (priorBx !== null && priorBy !== null) {
        const distSq = (bx - priorBx) * (bx - priorBx) + (by - priorBy) * (by - priorBy)
        temporalBonus = 1.0 + 0.35 * Math.exp(-distSq * 0.45)
      }
      const score = neighborhoodSum * edgePenalty * temporalBonus
      if (score > bestClusterScore) {
        bestClusterScore = score
        bestPeakSkin = rawSkin
        bestBx = bx
        bestBy = by
      }
    }
  }

  const area = width * height
  const skinRatio = totalSkinPixels / area
  if (bestBx < 0 || bestPeakSkin < 0.18 || skinRatio < 0.01) {
    return { x: 0, y: 0, confidence: 0, smile: 0, moving: false }
  }

  // Compute centroid over a 5x3 horizontal/vertical window around the peak skin block
  // so the centroid glides smoothly across block boundaries without discrete jumps.
  let clusterSumX = 0
  let clusterSumY = 0
  let clusterWeight = 0
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const nx = bestBx + dx
      const ny = bestBy + dy
      if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
        const w = blockSkin[ny * GRID_W + nx]
        if (w > 0.08) {
          const falloff = Math.abs(dx) === 2 ? 0.5 : 1.0
          const effW = w * falloff
          clusterSumX += (nx + 0.5) * effW
          clusterSumY += (ny + 0.5) * effW
          clusterWeight += effW
        }
      }
    }
  }

  if (clusterWeight <= 0) {
    return { x: 0, y: 0, confidence: 0, smile: 0, moving: false }
  }

  const cx = clusterSumX / clusterWeight / GRID_W
  const cy = clusterSumY / clusterWeight / GRID_H
  const normX = Math.max(-0.95, Math.min(0.95, (0.5 - cx) * 2.1))
  const normY = Math.max(-0.85, Math.min(0.85, (0.46 - cy) * 1.8))
  const confidence = Math.min(1, bestPeakSkin * 0.95)
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
  private baselineX = 0
  private baselineY = 0
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
    this.baselineX = 0
    this.baselineY = 0

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

  private updateSmoothedTarget(
    now: number,
    rawX: number,
    rawY: number,
    confidence: number,
    useAdaptiveCenter = false
  ) {
    let adjX = rawX
    let adjY = rawY
    if (useAdaptiveCenter) {
      if (!this.hasInitialLock) {
        // Treat only part of the initial static offset as background bias so real off-center
        // faces still register immediately, while static room asymmetry slowly calibrates out.
        this.baselineX = rawX * 0.35
        this.baselineY = rawY * 0.35
      } else {
        this.baselineX += (rawX - this.baselineX) * 0.025
        this.baselineY += (rawY - this.baselineY) * 0.025
      }
      adjX = Math.max(-1, Math.min(1, (rawX - this.baselineX) * 1.25))
      adjY = Math.max(-0.9, Math.min(0.9, (rawY - this.baselineY) * 1.2))
    }

    if (!this.hasInitialLock) {
      this.lastX = adjX
      this.lastY = adjY
      this.hasInitialLock = true
    } else {
      const dx = adjX - this.lastX
      const dy = adjY - this.lastY
      // Soft-threshold deadband: subtract the deadband radius before applying gain so
      // the target is 100% still at rest and starts moving with zero jump when leaving the deadband.
      if (Math.abs(dx) > POSITION_DEADBAND_X) {
        const excessX = dx - Math.sign(dx) * POSITION_DEADBAND_X
        this.lastX += excessX * 0.35
      }
      if (Math.abs(dy) > POSITION_DEADBAND_Y) {
        const excessY = dy - Math.sign(dy) * POSITION_DEADBAND_Y
        this.lastY += excessY * 0.30
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
            const normX = Math.max(-1, Math.min(1, (0.5 - cx) * 2.15))
            const normY = Math.max(-0.9, Math.min(0.9, (0.46 - cy) * 1.85))
            this.updateSmoothedTarget(now, normX, normY, 1)
            return
          }
        } catch {
          this.nativeFailed = true
        }
      }

      // 2. Downsampled 8x6 block motion + local skin cluster tracker
      if (!this.ctx) return
      const vw = video.videoWidth
      const vh = video.videoHeight
      // Crop the center 4:3 region without stretching when the camera is 9:16 portrait (mobile)
      // or 16:9 widescreen, so horizontal/vertical proportions are never distorted.
      const targetAspect = SAMPLE_W / SAMPLE_H
      const videoAspect = vw / vh
      let sx = 0
      let sy = 0
      let sw = vw
      let sh = vh
      if (videoAspect < targetAspect) {
        // Portrait (e.g. mobile 9:16): keep full width, crop top/bottom margins
        sh = Math.round(vw / targetAspect)
        sy = Math.round((vh - sh) * 0.42)
      } else if (videoAspect > targetAspect * 1.15) {
        // Wide landscape (16:9): crop extreme left/right room edges
        sw = Math.round(vh * targetAspect)
        sx = Math.round((vw - sw) * 0.5)
      }
      this.ctx.drawImage(video, sx, sy, sw, sh, 0, 0, SAMPLE_W, SAMPLE_H)
      const img = this.ctx.getImageData(0, 0, SAMPLE_W, SAMPLE_H)
      const res = computeSkinAndMotionCentroid(
        img.data,
        this.prevLuma,
        SAMPLE_W,
        SAMPLE_H,
        this.hasInitialLock ? this.lastX + this.baselineX : undefined,
        this.hasInitialLock ? this.lastY + this.baselineY : undefined
      )
      if ((res.moving && res.confidence > 0.2) || res.confidence > 0.25) {
        this.updateSmoothedTarget(now, res.x, res.y, res.confidence, true)
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
