// Life motion while the avatar is not speaking.
// LAM only emits ARKit frames for reply audio, and the renderer then zeroes every
// morph, which is the statue. Listening and thinking already exist as chat states;
// the asset has no listen clip (it replays idle). Brows and blinks are morphs.
// A nod and a breath are rotations and a few millimetres on the head bone, applied
// after the skeletal clip so they disappear as soon as she answers.

export type LifeMode = 'idle' | 'listen' | 'think' | 'speak'

const FACE_CHANNELS = [
  'browDownLeft',
  'browDownRight',
  'browInnerUp',
  'browOuterUpLeft',
  'browOuterUpRight',
  'cheekPuff',
  'cheekSquintLeft',
  'cheekSquintRight',
  'eyeBlinkLeft',
  'eyeBlinkRight',
  'eyeLookDownLeft',
  'eyeLookDownRight',
  'eyeLookInLeft',
  'eyeLookInRight',
  'eyeLookOutLeft',
  'eyeLookOutRight',
  'eyeLookUpLeft',
  'eyeLookUpRight',
  'eyeSquintLeft',
  'eyeSquintRight',
  'eyeWideLeft',
  'eyeWideRight',
  'jawForward',
  'jawLeft',
  'jawOpen',
  'jawRight',
  'mouthClose',
  'mouthDimpleLeft',
  'mouthDimpleRight',
  'mouthFrownLeft',
  'mouthFrownRight',
  'mouthFunnel',
  'mouthLeft',
  'mouthLowerDownLeft',
  'mouthLowerDownRight',
  'mouthPressLeft',
  'mouthPressRight',
  'mouthPucker',
  'mouthRight',
  'mouthRollLower',
  'mouthRollUpper',
  'mouthShrugLower',
  'mouthShrugUpper',
  'mouthSmileLeft',
  'mouthSmileRight',
  'mouthStretchLeft',
  'mouthStretchRight',
  'mouthUpperUpLeft',
  'mouthUpperUpRight',
  'noseSneerLeft',
  'noseSneerRight',
  'tongueOut',
] as const

// Same 7-frame blink shapes the LAM post-process uses, played over ~160 ms.
const BLINK_PATTERNS = [
  [0.365, 0.95, 0.956, 0.917, 0.367, 0.119, 0.025],
  [0.235, 0.91, 0.945, 0.778, 0.191, 0.235, 0.089],
  [0.87, 0.95, 0.949, 0.696, 0.191, 0.073, 0.007],
  [0.0, 0.557, 0.953, 0.942, 0.426, 0.148, 0.018],
]
const BLINK_DURATION = 0.16

// Head rest pose is in centimetres under a 0.01 armature scale, so 0.55 local
// is about 5.5 mm in the scene. Nod angles are radians on the bone's local X
// (chin down). Listening is a clearer nod than thinking.
const BREATH_AMP = 0.55
const DRIFT_X = 0.03
const DRIFT_Z = 0.016
const NOD_LISTEN = 0.14
const NOD_THINK = 0.08

const LIFE_FLAG = '__avatarLifeWrapped'

export function lifeMode(state: string): LifeMode {
  if (state === 'Listening') return 'listen'
  if (state === 'Thinking') return 'think'
  if (state === 'Responding') return 'speak'
  return 'idle'
}

function blankFace(): Record<string, number> {
  const face: Record<string, number> = {}
  for (const name of FACE_CHANNELS) face[name] = 0
  return face
}

function follow(current: number, target: number, dt: number, tau: number): number {
  if (dt <= 0) return current
  return current + (target - current) * (1 - Math.exp(-dt / tau))
}

function blinkSample(elapsed: number, pattern: number, scale: number): number {
  if (elapsed < 0 || elapsed >= BLINK_DURATION) return 0
  const shape = BLINK_PATTERNS[pattern] ?? BLINK_PATTERNS[0]
  const f = (elapsed / BLINK_DURATION) * (shape.length - 1)
  const i = Math.floor(f)
  const t = f - i
  const a = shape[i]
  const b = shape[Math.min(i + 1, shape.length - 1)]
  return (a + (b - a) * t) * scale
}

function rotateLocal(
  q: { x: number; y: number; z: number; w: number },
  axis: 'x' | 'z',
  angle: number
) {
  if (Math.abs(angle) < 1e-5) return
  const half = angle * 0.5
  const s = Math.sin(half)
  const bx = axis === 'x' ? s : 0
  const bz = axis === 'z' ? s : 0
  const bw = Math.cos(half)
  const ax = q.x
  const ay = q.y
  const az = q.z
  const aw = q.w
  q.x = aw * bx + ax * bw + ay * bz - az * 0
  q.y = aw * 0 - ax * bz + ay * bw + az * bx
  q.z = aw * bz + ax * 0 - ay * bx + az * bw
  q.w = aw * bw - ax * bx - ay * 0 - az * bz
}

export interface LifeBone {
  quaternion: { x: number; y: number; z: number; w: number }
  position: { y: number }
}

export class AvatarLife {
  weights: Record<string, number> = blankFace()
  nod = 0
  breath = 0
  driftX = 0
  driftZ = 0

  private time = 0
  private lastNow = -1
  private blinkT = -1
  private blinkPattern = 0
  private blinkScale = 0.9
  private nextBlink = 0.9
  private nodT = -1
  private nextNod = Number.POSITIVE_INFINITY
  private nodAmp = NOD_LISTEN
  private nodDur = 1.05
  private presence = 0
  private shape: 'listen' | 'think' = 'listen'
  private mode: LifeMode = 'idle'
  private rand: () => number

  constructor(rand: () => number = Math.random) {
    this.rand = rand
    this.nextBlink = 0.7 + this.rand() * 0.8
  }

  tick(state: string, nowSeconds?: number): Record<string, number> {
    const dt = this.advance(nowSeconds)
    const mode = lifeMode(state)
    this.onMode(mode)
    const blink = this.advanceBlink(dt, mode)
    const nodTarget = this.advanceNod(dt, mode)
    this.nod = follow(this.nod, nodTarget, dt, 0.09)
    const alive = mode !== 'speak'
    this.driftX = follow(this.driftX, alive ? Math.sin(this.time * 0.56) * DRIFT_X : 0, dt, 0.25)
    this.driftZ = follow(
      this.driftZ,
      alive ? Math.sin(this.time * 0.82 + 1.1) * DRIFT_Z : 0,
      dt,
      0.25
    )
    this.breath = follow(this.breath, alive ? Math.sin(this.time * 1.38) * BREATH_AMP : 0, dt, 0.3)
    const attentive = mode === 'listen' || mode === 'think'
    this.presence = follow(this.presence, attentive ? 1 : 0, dt, 0.16)
    if (attentive) this.shape = mode
    this.weights = this.composeFace(blink)
    return this.weights
  }

  applyBones(head: LifeBone, neck: LifeBone | null) {
    rotateLocal(head.quaternion, 'x', this.nod * 0.72 + this.driftX)
    rotateLocal(head.quaternion, 'z', this.driftZ)
    if (neck) rotateLocal(neck.quaternion, 'x', this.nod * 0.28)
    head.position.y += this.breath
  }

  private advance(nowSeconds?: number): number {
    const now = nowSeconds ?? performance.now() / 1000
    if (this.lastNow < 0) {
      this.lastNow = now
      return 0
    }
    const dt = Math.min(0.05, Math.max(0, now - this.lastNow))
    this.lastNow = now
    this.time += dt
    return dt
  }

  private onMode(mode: LifeMode) {
    const wasAttentive = this.mode === 'listen' || this.mode === 'think'
    const attentive = mode === 'listen' || mode === 'think'
    if (attentive && !wasAttentive) {
      this.nextNod = Math.min(this.nextNod, this.time + 0.35 + this.rand() * 0.4)
    }
    this.mode = mode
  }

  private advanceBlink(dt: number, mode: LifeMode): number {
    if (mode === 'speak') {
      this.blinkT = -1
      if (this.nextBlink < this.time + 0.8) {
        this.nextBlink = this.time + 0.8 + this.rand() * 2.2
      }
      return 0
    }
    if (this.blinkT < 0 && this.time >= this.nextBlink) {
      this.blinkT = 0
      this.blinkPattern = Math.floor(this.rand() * BLINK_PATTERNS.length)
      this.blinkScale = 0.75 + this.rand() * 0.25
    }
    if (this.blinkT < 0) return 0
    const value = blinkSample(this.blinkT, this.blinkPattern, this.blinkScale)
    this.blinkT += dt
    if (this.blinkT >= BLINK_DURATION) {
      this.blinkT = -1
      const doubled = this.rand() < 0.22
      this.nextBlink = this.time + (doubled ? 0.16 + this.rand() * 0.1 : 2.4 + this.rand() * 3.4)
    }
    return value
  }

  private advanceNod(dt: number, mode: LifeMode): number {
    if (mode !== 'listen' && mode !== 'think') {
      this.nodT = -1
      return 0
    }
    if (this.nodT < 0 && this.time >= this.nextNod) {
      this.nodT = 0
      this.nodAmp = mode === 'listen' ? NOD_LISTEN : NOD_THINK
      this.nodDur = mode === 'listen' ? 1.05 : 1.35
    }
    if (this.nodT < 0) return 0
    const u = Math.min(1, this.nodT / this.nodDur)
    this.nodT += dt
    if (this.nodT >= this.nodDur) {
      this.nodT = -1
      const gap = mode === 'listen' ? 1.5 : 2.4
      this.nextNod = this.time + gap + this.rand() * 1.8
    }
    return Math.sin(Math.PI * u) * this.nodAmp
  }

  private composeFace(blink: number): Record<string, number> {
    const face = blankFace()
    face.eyeBlinkLeft = blink
    face.eyeBlinkRight = blink
    const p = this.presence
    if (p < 0.01) return face
    const wobble = 0.5 + 0.5 * Math.sin(this.time * 1.7)
    if (this.shape === 'listen') {
      face.browOuterUpLeft = (0.48 + 0.06 * wobble) * p
      face.browOuterUpRight = 0.12 * p
      face.browInnerUp = 0.22 * p
      face.mouthPressLeft = 0.2 * p
      face.mouthPressRight = 0.15 * p
      face.mouthShrugLower = 0.14 * p
      face.jawOpen = 0.04 * p
    } else {
      face.browInnerUp = (0.34 + 0.06 * wobble) * p
      face.browOuterUpLeft = 0.18 * p
      face.browOuterUpRight = 0.16 * p
      face.eyeLookUpLeft = 0.22 * p
      face.eyeLookUpRight = 0.22 * p
      face.mouthPressLeft = 0.12 * p
      face.mouthPressRight = 0.1 * p
      face.mouthShrugLower = 0.08 * p
    }
    return face
  }
}

export function attachAvatarLife(renderer: any, life: AvatarLife): boolean {
  const mixer = renderer?.mixer
  const bones: Array<{
    name?: string
    quaternion: LifeBone['quaternion']
    position: LifeBone['position']
  }> = renderer?.viewer?.skinModel?.skeleton?.bones
  if (!mixer || !bones || mixer[LIFE_FLAG]) return Boolean(mixer?.[LIFE_FLAG])
  const head = bones.find((bone) => bone.name === 'head')
  const neck = bones.find((bone) => bone.name === 'neckLower') ?? null
  if (!head) return false
  const original = mixer.update.bind(mixer)
  mixer.update = (delta: number) => {
    original(delta)
    life.applyBones(head, neck)
  }
  mixer[LIFE_FLAG] = true
  return true
}
