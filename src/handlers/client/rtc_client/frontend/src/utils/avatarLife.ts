// Life motion while the avatar is idle, listening, thinking, or speaking.
// LAM only emits ARKit frames for reply audio, and the renderer then zeroes every
// morph when silent. Listening and thinking are driven procedurally, while speaking
// combines natural 3-axis head/neck prosody, breathing, and gaze micro-saccades with
// smooth cross-fading into and out of LAM lip-sync frames.

export type LifeMode = 'idle' | 'listen' | 'think' | 'speak'

export const FACE_CHANNELS = [
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

const GAZE_CHANNELS = new Set([
  'eyeLookDownLeft',
  'eyeLookDownRight',
  'eyeLookInLeft',
  'eyeLookInRight',
  'eyeLookOutLeft',
  'eyeLookOutRight',
  'eyeLookUpLeft',
  'eyeLookUpRight',
])

const EYE_WIDE_CHANNELS = new Set(['eyeWideLeft', 'eyeWideRight'])

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
const BREATH_AMP = 0.52
const DRIFT_X = 0.018
const DRIFT_Y = 0.0 // Zero horizontal yaw oscillation at rest so she never appears to shake her head "no"
const DRIFT_Z = 0.01
const NOD_LISTEN = 0.14
const NOD_THINK = 0.08
const SPEECH_NOD_GAIN = 0.085
const SPEECH_YAW_GAIN = 0.012
const SPEECH_ROLL_GAIN = 0.014
const TRACK_HEAD_YAW_MAX = 0.28 // ~16 deg max horizontal head turn toward user
const TRACK_HEAD_PITCH_MAX = 0.11 // ~6.3 deg max vertical head tilt toward user
const TRACK_HEAD_ROLL_MAX = 0.035 // subtle empathetic head tilt when looking sideways
const TRACK_GAZE_X_MAX = 0.50
const TRACK_GAZE_Y_MAX = 0.34

export type AvatarEmotion =
  | 'neutral'
  | 'joy'
  | 'sadness'
  | 'surprise'
  | 'anger'
  | 'fear'
  | 'disgust'

// Lingering mood morph targets applied in Idle/Listening/Thinking so the avatar
// keeps an emotional afterglow between turns rather than snapping to a blank face.
const MOOD_PRESETS: Record<AvatarEmotion, Record<string, number>> = {
  neutral: {},
  joy: {
    mouthSmileLeft: 0.24,
    mouthSmileRight: 0.23,
    cheekSquintLeft: 0.16,
    cheekSquintRight: 0.16,
    mouthDimpleLeft: 0.08,
    mouthDimpleRight: 0.08,
    browOuterUpLeft: 0.06,
    browOuterUpRight: 0.06,
  },
  sadness: {
    browInnerUp: 0.28,
    mouthFrownLeft: 0.15,
    mouthFrownRight: 0.15,
    mouthPressLeft: 0.09,
    mouthPressRight: 0.09,
    eyeSquintLeft: 0.08,
    eyeSquintRight: 0.08,
  },
  surprise: {
    browInnerUp: 0.22,
    browOuterUpLeft: 0.2,
    browOuterUpRight: 0.2,
    eyeWideLeft: 0.18,
    eyeWideRight: 0.18,
    jawOpen: 0.04,
  },
  anger: {
    browDownLeft: 0.22,
    browDownRight: 0.22,
    eyeSquintLeft: 0.15,
    eyeSquintRight: 0.15,
    noseSneerLeft: 0.1,
    noseSneerRight: 0.1,
    mouthPressLeft: 0.14,
    mouthPressRight: 0.14,
    jawForward: 0.05,
  },
  fear: {
    browInnerUp: 0.26,
    browOuterUpLeft: 0.14,
    browOuterUpRight: 0.14,
    eyeWideLeft: 0.15,
    eyeWideRight: 0.15,
    mouthStretchLeft: 0.08,
    mouthStretchRight: 0.08,
  },
  disgust: {
    noseSneerLeft: 0.2,
    noseSneerRight: 0.18,
    browDownLeft: 0.12,
    browDownRight: 0.12,
    mouthUpperUpLeft: 0.1,
    mouthUpperUpRight: 0.08,
  },
}

const MOOD_PITCH_OFFSET: Record<AvatarEmotion, number> = {
  neutral: 0,
  joy: -0.02, // slight confident chin lift
  sadness: 0.045, // empathetic downward bow
  surprise: -0.03,
  anger: 0.02,
  fear: 0.015,
  disgust: -0.01,
}

const LIFE_FLAG = '__avatarLifeWrapped'
const ANIM_PATCH_FLAG = '__avatarLifeAnimPatched'

export function sanitizeAnimManager(animManager: any, chatState?: string) {
  if (!animManager) return
  const ctor = animManager.constructor
  const stateBuckets = [
    animManager.hello,
    animManager.idle,
    animManager.listen,
    animManager.think,
    animManager.speak,
  ]

  // 1. Ensure all cloned actions (such as listen.actions[0]) are registered in
  // AnimationManager.actions and patch PrepareCrossFade against self-crossfades.
  if (ctor && !animManager[ANIM_PATCH_FLAG]) {
    animManager[ANIM_PATCH_FLAG] = true
    if (Array.isArray(ctor.actions)) {
      for (const bucket of stateBuckets) {
        const actions = bucket?.actions
        if (!Array.isArray(actions)) continue
        for (const action of actions) {
          if (action && !ctor.actions.includes(action)) {
            ctor.actions.push(action)
          }
        }
      }
    }
    if (typeof ctor.PrepareCrossFade === 'function' && !ctor[ANIM_PATCH_FLAG]) {
      ctor[ANIM_PATCH_FLAG] = true
      const origPrepare = ctor.PrepareCrossFade.bind(ctor)
      ctor.PrepareCrossFade = (startAction: any, endAction: any, duration: number) => {
        if (!endAction) return
        if (!startAction || startAction === endAction) {
          ctor.UnPauseAllActions?.()
          endAction.enabled = true
          endAction.paused = false
          endAction.setEffectiveTimeScale?.(1)
          endAction.setEffectiveWeight?.(1)
          endAction.play?.()
          return
        }
        origPrepare(startAction, endAction, duration)
      }
    }
  }

  // 2. Unblock Think state machine when stage 2 (think_part_3) gets stuck
  // either because state re-entered 'Thinking' during stage 2 or stage 2 clamped/paused at end.
  const think = animManager.think
  if (think?.isPlaying && Array.isArray(think.actions)) {
    const stage2 = think.actions[2]
    if (think.stage === 2 && stage2) {
      const clipDur = stage2.getClip?.()?.duration ?? 0
      const nearEnd = clipDur > 0 && stage2.time >= clipDur - 0.31
      if (chatState === 'Thinking' || stage2.paused || nearEnd) {
        think.isPlaying = false
        if (ctor) {
          ctor.LastAction = stage2
          ctor.CurPlaying = undefined
        }
        if (chatState === 'Thinking') {
          think.update?.(chatState)
        }
      }
    }
  }

  // 3. Ensure the currently active state's current stage action is never left paused/disabled
  for (const bucket of stateBuckets) {
    if (!bucket?.isPlaying || !Array.isArray(bucket.actions)) continue
    const activeAction = bucket.actions[bucket.stage ?? 0]
    if (!activeAction) continue
    if (activeAction.paused || !activeAction.enabled) {
      activeAction.enabled = true
      activeAction.paused = false
      activeAction.setEffectiveTimeScale?.(1)
      if ((activeAction.getEffectiveWeight?.() ?? 1) <= 0.01) {
        activeAction.setEffectiveWeight?.(1)
      }
      activeAction.play?.()
    }
  }
}

export function lifeMode(state: string): LifeMode {
  if (state === 'Listening') return 'listen'
  if (state === 'Thinking') return 'think'
  if (state === 'Responding') return 'speak'
  return 'idle'
}

export function blankFace(): Record<string, number> {
  const face: Record<string, number> = {}
  for (const name of FACE_CHANNELS) face[name] = 0
  return face
}

export function follow(current: number, target: number, dt: number, tau: number): number {
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

export function rotateLocal(
  q: { x: number; y: number; z: number; w: number },
  axis: 'x' | 'y' | 'z',
  angle: number
) {
  if (Math.abs(angle) < 1e-5) return
  const half = angle * 0.5
  const s = Math.sin(half)
  const bx = axis === 'x' ? s : 0
  const by = axis === 'y' ? s : 0
  const bz = axis === 'z' ? s : 0
  const bw = Math.cos(half)
  const ax = q.x
  const ay = q.y
  const az = q.z
  const aw = q.w
  q.x = aw * bx + ax * bw + ay * bz - az * by
  q.y = aw * by - ax * bz + ay * bw + az * bx
  q.z = aw * bz + ax * by - ay * bx + az * bw
  q.w = aw * bw - ax * bx - ay * by - az * bz
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
  driftY = 0
  driftZ = 0
  gazeX = 0
  gazeY = 0
  trackHeadX = 0
  trackHeadY = 0
  mood: AvatarEmotion = 'neutral'
  moodIntensity = 0
  empatheticSmile = 0
  speechPresence = 0

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
  private targetGazeX = 0
  private targetGazeY = 0
  private saccadeOffsetX = 0
  private saccadeOffsetY = 0
  private userTargetX = 0
  private userTargetY = 0
  private smoothTargetX = 0
  private smoothTargetY = 0
  private userTracked = false
  private userSmile = 0
  private moodPitch = 0
  private moodWeights: Record<string, number> = {}
  private nextSaccade = 0.8
  private speechEnergy = 0
  private speechNod = 0
  private lastSpeechWeights: Record<string, number> | null = null
  private rand: () => number

  constructor(rand: () => number = Math.random) {
    this.rand = rand
    this.nextBlink = 0.7 + this.rand() * 0.8
    this.nextSaccade = 0.6 + this.rand() * 1.1
  }

  /**
   * Set the avatar's current emotional mood so it lingers naturally between sentences.
   */
  setEmotion(emotion: string) {
    const normalized = (emotion || 'neutral').trim().toLowerCase() as AvatarEmotion
    if (normalized in MOOD_PRESETS) {
      this.mood = normalized
      this.moodIntensity = normalized === 'neutral' ? 0 : 1.0
    }
  }

  /**
   * Trigger an immediate warm empathetic nod when the user nods while she is actively listening.
   */
  triggerEmpatheticNod() {
    if (this.mode !== 'listen') return
    if (this.nodT < 0) {
      this.nodT = 0
      this.nodAmp = NOD_LISTEN
      this.nodDur = 0.95
    }
    this.empatheticSmile = Math.max(this.empatheticSmile, 0.22)
  }

  /**
   * Update the user's normalized position and empathetic cues from the webcam tracker.
   * @param x Horizontal position in [-1, 1] (positive = user is to the right of screen)
   * @param y Vertical position in [-1, 1] (positive = user is higher up)
   * @param active Whether a user is currently detected in the camera feed
   * @param cues Optional user gesture cues (nodding, smile)
   */
  setUserTarget(
    x: number,
    y: number,
    active = true,
    cues?: { nodding?: boolean; smile?: number }
  ) {
    if (!active || !Number.isFinite(x) || !Number.isFinite(y)) {
      this.userTracked = false
      this.userTargetX = 0
      this.userTargetY = 0
      this.userSmile = 0
      return
    }
    this.userTracked = true
    this.userTargetX = Math.max(-1, Math.min(1, x))
    this.userTargetY = Math.max(-1, Math.min(1, y))
    if (cues?.smile !== undefined && Number.isFinite(cues.smile)) {
      this.userSmile = Math.max(0, Math.min(1, cues.smile))
    }
    if (cues?.nodding) {
      this.triggerEmpatheticNod()
    }
  }

  tick(
    state: string,
    nowSeconds?: number,
    speechFrame?: Record<string, number> | null
  ): Record<string, number> {
    const dt = this.advance(nowSeconds)
    const mode = lifeMode(state)
    this.onMode(mode)

    const rawJaw = speechFrame ? Math.max(0, speechFrame.jawOpen ?? 0) : 0
    const rawMouth = speechFrame
      ? Math.max(
          rawJaw,
          (speechFrame.mouthFunnel ?? 0) * 0.6,
          (speechFrame.mouthPucker ?? 0) * 0.5,
          ((speechFrame.mouthLowerDownLeft ?? 0) + (speechFrame.mouthLowerDownRight ?? 0)) * 0.4
        )
      : 0
    const activeSpeech = speechFrame !== null && speechFrame !== undefined
    this.speechEnergy = follow(this.speechEnergy, activeSpeech ? rawMouth : 0, dt, 0.08)

    const blink = this.advanceBlink(dt, mode)
    const nodTarget = this.advanceNod(dt, mode)
    this.nod = follow(this.nod, nodTarget, dt, 0.09)

    // Calm, low-frequency organic head drift across all 3 rotational axes.
    const driftScale = mode === 'speak' ? 0.65 : 1.0
    const targetDriftX = Math.sin(this.time * 0.56) * DRIFT_X * driftScale
    const targetDriftY = Math.sin(this.time * 0.38 + 0.5) * DRIFT_Y * driftScale
    const targetDriftZ = Math.sin(this.time * 0.82 + 1.1) * DRIFT_Z * driftScale

    this.driftX = follow(this.driftX, targetDriftX, dt, 0.35)
    this.driftY = follow(this.driftY, targetDriftY, dt, 0.38)
    this.driftZ = follow(this.driftZ, targetDriftZ, dt, 0.35)

    // Prosodic syllable-coupled head nod & subtle turn while speaking
    const targetSpeechNod =
      mode === 'speak'
        ? this.speechEnergy * SPEECH_NOD_GAIN * (0.7 + 0.3 * Math.sin(this.time * 3.1))
        : 0
    this.speechNod = follow(this.speechNod, targetSpeechNod, dt, 0.09)

    // Continuous natural breathing in all modes
    const breathFreq = mode === 'speak' ? 1.55 : 1.38
    const breathScale = mode === 'speak' ? 0.72 : 1.0
    this.breath = follow(
      this.breath,
      Math.sin(this.time * breathFreq) * BREATH_AMP * breathScale,
      dt,
      0.3
    )

    // Two-stage exponential low-pass filter on the 12.5 Hz webcam target so 60 fps head
    // and eye motion is critically damped (C1-continuous velocity, zero step-response jerks).
    const rawTargetX = this.userTracked ? this.userTargetX : 0
    const rawTargetY = this.userTracked ? this.userTargetY : 0
    this.smoothTargetX = follow(this.smoothTargetX, rawTargetX, dt, 0.22)
    this.smoothTargetY = follow(this.smoothTargetY, rawTargetY, dt, 0.25)

    this.advanceGaze(dt, mode)

    // Smoothly follow the user's webcam position with natural head/neck inertia
    const headTrackScale = mode === 'think' ? 0.65 : 1.0
    const targetHeadX = this.smoothTargetX * TRACK_HEAD_YAW_MAX * headTrackScale
    const deadzoneY = Math.abs(this.smoothTargetY) < 0.03 ? 0 : this.smoothTargetY
    const targetHeadY = -deadzoneY * TRACK_HEAD_PITCH_MAX * headTrackScale
    this.trackHeadX = follow(this.trackHeadX, targetHeadX, dt, 0.32)
    this.trackHeadY = follow(this.trackHeadY, targetHeadY, dt, 0.36)

    // Update lingering mood afterglow (decays gently toward a 0.28 baseline over ~25s)
    const moodFloor = this.mood === 'neutral' ? 0 : 0.28
    const moodDecayTau = mode === 'speak' ? 8.0 : 22.0
    this.moodIntensity = follow(this.moodIntensity, moodFloor, dt, moodDecayTau)
    this.moodPitch = follow(
      this.moodPitch,
      (MOOD_PITCH_OFFSET[this.mood] ?? 0) * this.moodIntensity,
      dt,
      0.35
    )

    // Smoothly interpolate individual mood morphs so mood switches never snap
    const activePreset = MOOD_PRESETS[this.mood] ?? {}
    for (const name of FACE_CHANNELS) {
      const targetVal = (activePreset[name] ?? 0) * this.moodIntensity
      const prevVal = this.moodWeights[name] ?? 0
      if (targetVal > 0 || prevVal > 0.001) {
        this.moodWeights[name] = follow(prevVal, targetVal, dt, 0.28)
      }
    }

    // Decay empathetic smile boost from user nods/smiles
    const targetEmpSmile = this.userTracked ? this.userSmile * 0.22 : 0
    this.empatheticSmile = follow(this.empatheticSmile, targetEmpSmile, dt, 0.65)

    const attentive = mode === 'listen' || mode === 'think'
    this.presence = follow(this.presence, attentive ? 1 : 0, dt, 0.16)
    if (attentive) this.shape = mode

    const lifeFace = this.composeFace(blink, mode)
    this.weights = lifeFace

    if (speechFrame === undefined) {
      return lifeFace
    }

    return this.blendWithSpeech(lifeFace, speechFrame, dt)
  }

  applyBones(head: LifeBone, neck: LifeBone | null) {
    const pitch = this.nod + this.speechNod + this.driftX + this.trackHeadY + this.moodPitch
    const yaw =
      this.driftY +
      this.trackHeadX +
      (this.mode === 'speak' ? Math.sin(this.time * 1.9) * this.speechEnergy * SPEECH_YAW_GAIN : 0)
    const roll =
      this.driftZ +
      this.trackHeadX * (TRACK_HEAD_ROLL_MAX / TRACK_HEAD_YAW_MAX) +
      (this.mode === 'speak'
        ? Math.cos(this.time * 1.45) * this.speechEnergy * SPEECH_ROLL_GAIN
        : 0)

    rotateLocal(head.quaternion, 'x', pitch * 0.72)
    rotateLocal(head.quaternion, 'y', yaw * 0.7)
    rotateLocal(head.quaternion, 'z', roll * 0.75)
    if (neck) {
      rotateLocal(neck.quaternion, 'x', pitch * 0.28)
      rotateLocal(neck.quaternion, 'y', yaw * 0.3)
      rotateLocal(neck.quaternion, 'z', roll * 0.25)
    }
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
    if (mode !== this.mode) {
      // Trigger an immediate gaze update on mode shift (e.g., look up on think, lock eyes on listen)
      this.nextSaccade = Math.min(this.nextSaccade, this.time + 0.08)
    }
    this.mode = mode
  }

  private advanceGaze(dt: number, mode: LifeMode) {
    if (this.time >= this.nextSaccade) {
      const r1 = this.rand() * 2 - 1
      const r2 = this.rand() * 2 - 1
      const eyeContactBias = this.rand() < 0.68
      if (mode === 'think') {
        // Gentle reflective upward glance while maintaining primary connection
        if (this.rand() < 0.45) {
          this.saccadeOffsetX = r1 * 0.035
          this.saccadeOffsetY = 0.06 + Math.abs(r2) * 0.05
        } else {
          this.saccadeOffsetX = (r1 >= 0 ? 1 : -1) * (0.04 + Math.abs(r1) * 0.07)
          this.saccadeOffsetY = 0.12 + Math.abs(r2) * 0.08
        }
        this.nextSaccade = this.time + 1.0 + this.rand() * 1.5
      } else if (mode === 'listen') {
        // Attentive eye triangle saccades (left eye -> right eye -> bridge)
        this.saccadeOffsetX = eyeContactBias ? 0 : r1 * 0.035
        this.saccadeOffsetY = eyeContactBias ? 0 : r2 * 0.025
        this.nextSaccade = this.time + 1.5 + this.rand() * 2.2
      } else if (mode === 'speak') {
        //Steady conversational eye contact with subtle micro-saccades
        const centered = eyeContactBias || this.speechEnergy > 0.22
        this.saccadeOffsetX = centered ? r1 * 0.018 : r1 * 0.045
        this.saccadeOffsetY = centered ? r2 * 0.015 : r2 * 0.03
        this.nextSaccade = this.time + 1.2 + this.rand() * 1.9
      } else {
        // Idle: calm micro-saccades around the user's position (or center)
        this.saccadeOffsetX = eyeContactBias ? r1 * 0.014 : r1 * 0.038
        this.saccadeOffsetY = eyeContactBias ? r2 * 0.01 : r2 * 0.025
        this.nextSaccade = this.time + 2.2 + this.rand() * 2.8
      }
    }

    // Anchor gaze on the tracked user (with VOR compensation as the head catches up)
    const userAnchorScale = mode === 'think' ? 0.65 : 1.0
    const userGazeX =
      (this.smoothTargetX * TRACK_GAZE_X_MAX - this.trackHeadX * 0.35) * userAnchorScale
    const userGazeY =
      (this.smoothTargetY * TRACK_GAZE_Y_MAX + this.trackHeadY * 0.35) * userAnchorScale

    this.targetGazeX = Math.max(-0.55, Math.min(0.55, userGazeX + this.saccadeOffsetX))
    this.targetGazeY = Math.max(-0.45, Math.min(0.45, userGazeY + this.saccadeOffsetY))

    // Smooth eye tracking (tau ~110 ms) without abrupt twitching
    this.gazeX = follow(this.gazeX, this.targetGazeX, dt, 0.11)
    this.gazeY = follow(this.gazeY, this.targetGazeY, dt, 0.11)
  }

  private advanceBlink(dt: number, mode: LifeMode): number {
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
      const doubled = mode !== 'speak' && this.rand() < 0.22
      const baseGap = mode === 'speak' ? 3.0 + this.rand() * 2.8 : 2.4 + this.rand() * 3.4
      this.nextBlink = this.time + (doubled ? 0.16 + this.rand() * 0.1 : baseGap)
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

  private applyGazeToFace(face: Record<string, number>, blink: number) {
    const openFactor = Math.max(0, 1 - blink * 0.9)
    const gx = this.gazeX * openFactor
    const gy = this.gazeY * openFactor

    // ARKit eye morphs are named from the avatar's anatomical perspective:
    // In gbot's skin.glb, eyeLookOutRight + eyeLookInLeft shift the pupils toward +X
    // (the right side of the user's screen), matching head +Y yaw rotation (trackHeadX > 0).
    if (gx >= 0) {
      face.eyeLookOutRight = Math.min(1, (face.eyeLookOutRight ?? 0) + gx)
      face.eyeLookInLeft = Math.min(1, (face.eyeLookInLeft ?? 0) + gx)
    } else {
      const mag = -gx
      face.eyeLookOutLeft = Math.min(1, (face.eyeLookOutLeft ?? 0) + mag)
      face.eyeLookInRight = Math.min(1, (face.eyeLookInRight ?? 0) + mag)
    }

    if (gy >= 0) {
      face.eyeLookUpLeft = Math.min(1, (face.eyeLookUpLeft ?? 0) + gy)
      face.eyeLookUpRight = Math.min(1, (face.eyeLookUpRight ?? 0) + gy)
    } else {
      const down = -gy
      face.eyeLookDownLeft = Math.min(1, (face.eyeLookDownLeft ?? 0) + down)
      face.eyeLookDownRight = Math.min(1, (face.eyeLookDownRight ?? 0) + down)
      // Upper eyelids naturally follow large downward gaze only (avoids micro-saccade eyelid flutter)
      if (down > 0.12) {
        const lidDrop = (down - 0.12) * 0.12
        face.eyeBlinkLeft = Math.min(1, (face.eyeBlinkLeft ?? 0) + lidDrop)
        face.eyeBlinkRight = Math.min(1, (face.eyeBlinkRight ?? 0) + lidDrop)
      }
    }
  }

  private composeFace(blink: number, mode: LifeMode): Record<string, number> {
    const face = blankFace()
    face.eyeBlinkLeft = blink
    face.eyeBlinkRight = blink

    // Steady idle micro-warmth + empathetic smile mimicry from webcam (no oscillation at rest)
    const baseWarmth = mode === 'idle' ? 0.025 : 0
    const sharedSmile = Math.min(0.32, baseWarmth + this.empatheticSmile)
    face.mouthSmileLeft = sharedSmile
    face.mouthSmileRight = sharedSmile * 0.94
    face.cheekSquintLeft = sharedSmile * 0.55
    face.cheekSquintRight = sharedSmile * 0.55

    // Apply lingering emotional mood morphs (blink-safe for eyeWide)
    const openFactor = Math.max(0, 1 - blink * 0.9)
    for (const [name, val] of Object.entries(this.moodWeights)) {
      if (val <= 0.001) continue
      const eff = EYE_WIDE_CHANNELS.has(name) ? val * openFactor : val
      face[name] = Math.min(1, Math.max(face[name] ?? 0, (face[name] ?? 0) + eff))
    }

    this.applyGazeToFace(face, blink)

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
      face.mouthSmileLeft = Math.max(face.mouthSmileLeft, 0.08 * p)
      face.mouthSmileRight = Math.max(face.mouthSmileRight, 0.07 * p)
      face.cheekSquintLeft = Math.max(face.cheekSquintLeft, 0.06 * p)
      face.cheekSquintRight = Math.max(face.cheekSquintRight, 0.05 * p)
      face.jawOpen = 0.04 * p
    } else {
      face.browInnerUp = (0.34 + 0.06 * wobble) * p
      face.browOuterUpLeft = 0.18 * p
      face.browOuterUpRight = 0.16 * p
      face.eyeLookUpLeft = Math.max(face.eyeLookUpLeft, 0.22 * p * (1 - blink))
      face.eyeLookUpRight = Math.max(face.eyeLookUpRight, 0.22 * p * (1 - blink))
      face.mouthPressLeft = 0.12 * p
      face.mouthPressRight = 0.1 * p
      face.mouthShrugLower = 0.08 * p
    }
    return face
  }

  private blendWithSpeech(
    lifeFace: Record<string, number>,
    speechFrame: Record<string, number> | null,
    dt: number
  ): Record<string, number> {
    const hasSpeech = speechFrame !== null && speechFrame !== undefined
    if (hasSpeech) {
      this.lastSpeechWeights = speechFrame
    }
    const tau = hasSpeech ? 0.1 : 0.16
    this.speechPresence = follow(this.speechPresence, hasSpeech ? 1 : 0, dt, tau)

    if (this.speechPresence < 0.005 || !this.lastSpeechWeights) {
      return lifeFace
    }

    const sp = this.speechPresence
    const src = this.lastSpeechWeights
    const out = blankFace()
    const speechBlink = Math.max(src.eyeBlinkLeft ?? 0, src.eyeBlinkRight ?? 0, lifeFace.eyeBlinkLeft ?? 0)
    const blinkDamp = Math.max(0, 1 - speechBlink * 0.9)

    for (const name of FACE_CHANNELS) {
      const lifeVal = lifeFace[name] ?? 0
      const speechVal = src[name] ?? 0
      if (GAZE_CHANNELS.has(name)) {
        // Drive gaze exclusively from AvatarLife so audio-only LAM eye noise never breaks eye contact
        out[name] = Math.min(1, lifeVal * blinkDamp)
      } else if (EYE_WIDE_CHANNELS.has(name)) {
        // Suppress eyeWide during blinks so eyelids close completely
        out[name] = Math.min(1, (lifeVal * (1 - sp) + speechVal * sp) * blinkDamp)
      } else if (name === 'eyeBlinkLeft' || name === 'eyeBlinkRight') {
        out[name] = Math.min(1, Math.max(lifeVal, speechVal * sp))
      } else {
        out[name] = Math.min(1, lifeVal * (1 - sp) + speechVal * sp)
      }
    }

    return out
  }
}

function slerpQuatInPlace(
  q: { x: number; y: number; z: number; w: number },
  rest: { x: number; y: number; z: number; w: number },
  keepAnimFactor: number
) {
  let rx = rest.x
  let ry = rest.y
  let rz = rest.z
  let rw = rest.w
  if (q.x * rx + q.y * ry + q.z * rz + q.w * rw < 0) {
    rx = -rx
    ry = -ry
    rz = -rz
    rw = -rw
  }
  const nx = rx + (q.x - rx) * keepAnimFactor
  const ny = ry + (q.y - ry) * keepAnimFactor
  const nz = rz + (q.z - rz) * keepAnimFactor
  const nw = rw + (q.w - rw) * keepAnimFactor
  const invLen = 1 / (Math.hypot(nx, ny, nz, nw) || 1)
  q.x = nx * invLen
  q.y = ny * invLen
  q.z = nz * invLen
  q.w = nw * invLen
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

  // Capture camera-facing bind/rest pose of head & neck so mocap clips in animation.glb
  // (speak_3, speak_4, think_part_1..3 which yaw the head 6-8 deg sideways) are damped
  // by 75%, keeping the avatar naturally facing the user while preserving subtle life.
  const restHeadQuat = {
    x: head.quaternion.x,
    y: head.quaternion.y,
    z: head.quaternion.z,
    w: head.quaternion.w,
  }
  const restNeckQuat = neck
    ? {
        x: neck.quaternion.x,
        y: neck.quaternion.y,
        z: neck.quaternion.z,
        w: neck.quaternion.w,
      }
    : null

  // Save clean post-mixer bone transforms so AvatarLife offsets never accumulate
  // across frames when Three.js PropertyMixer skips unchanged/paused bone tracks.
  let baseHeadY = head.position.y
  let baseHeadQuat = { ...restHeadQuat }
  let baseNeckQuat = restNeckQuat ? { ...restNeckQuat } : null

  sanitizeAnimManager(renderer?.animManager, renderer?.chatState)

  const original = mixer.update.bind(mixer)
  mixer.update = (delta: number) => {
    head.position.y = baseHeadY
    head.quaternion.x = baseHeadQuat.x
    head.quaternion.y = baseHeadQuat.y
    head.quaternion.z = baseHeadQuat.z
    head.quaternion.w = baseHeadQuat.w
    if (neck && baseNeckQuat) {
      neck.quaternion.x = baseNeckQuat.x
      neck.quaternion.y = baseNeckQuat.y
      neck.quaternion.z = baseNeckQuat.z
      neck.quaternion.w = baseNeckQuat.w
    }

    sanitizeAnimManager(renderer?.animManager, renderer?.chatState)
    original(delta)

    baseHeadY = head.position.y
    baseHeadQuat = {
      x: head.quaternion.x,
      y: head.quaternion.y,
      z: head.quaternion.z,
      w: head.quaternion.w,
    }
    if (neck) {
      baseNeckQuat = {
        x: neck.quaternion.x,
        y: neck.quaternion.y,
        z: neck.quaternion.z,
        w: neck.quaternion.w,
      }
    }

    // Lock head & neck horizontal yaw to camera-facing rest pose before applying AvatarLife
    // tracking, so baked mocap loops in animation.glb (idle: -0.6..+1.8 deg yaw every 8s,
    // speak_1..4: -3.6..+5.9 deg yaw) never make her shake her head "no" at rest or while speaking.
    slerpQuatInPlace(head.quaternion, restHeadQuat, 0.15)
    head.quaternion.y = restHeadQuat.y
    const hNorm = 1 / (Math.hypot(head.quaternion.x, head.quaternion.y, head.quaternion.z, head.quaternion.w) || 1)
    head.quaternion.x *= hNorm
    head.quaternion.y *= hNorm
    head.quaternion.z *= hNorm
    head.quaternion.w *= hNorm
    if (neck && restNeckQuat) {
      slerpQuatInPlace(neck.quaternion, restNeckQuat, 0.15)
      neck.quaternion.y = restNeckQuat.y
      const nNorm = 1 / (Math.hypot(neck.quaternion.x, neck.quaternion.y, neck.quaternion.z, neck.quaternion.w) || 1)
      neck.quaternion.x *= nNorm
      neck.quaternion.y *= nNorm
      neck.quaternion.z *= nNorm
      neck.quaternion.w *= nNorm
    }

    life.applyBones(head, neck)
  }
  mixer[LIFE_FLAG] = true
  return true
}
