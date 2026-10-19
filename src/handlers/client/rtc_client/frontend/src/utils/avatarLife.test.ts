import assert from 'node:assert/strict'
import { AvatarLife, attachAvatarLife, blankFace, sanitizeAnimManager } from './avatarLife.ts'
import { computeSkinAndMotionCentroid, detectNodGesture } from './webcamTracker.ts'

function step(
  life: AvatarLife,
  state: string,
  seconds: number,
  stepSeconds = 1 / 60,
  speechFrame?: Record<string, number> | null
) {
  let now = (life as any).lastNow >= 0 ? (life as any).lastNow : 0
  if ((life as any).lastNow < 0) {
    life.tick(state, now, speechFrame)
  }
  let maxBlink = 0
  let maxBrow = 0
  let maxBreath = 0
  let maxNod = 0
  let maxGaze = 0
  let lastFace = life.weights
  const end = now + seconds
  for (let t = now + stepSeconds; t <= end + 1e-6; t += stepSeconds) {
    lastFace = life.tick(state, t, speechFrame)
    maxBlink = Math.max(maxBlink, lastFace.eyeBlinkLeft)
    maxBrow = Math.max(maxBrow, lastFace.browOuterUpLeft)
    maxBreath = Math.max(maxBreath, Math.abs(life.breath))
    maxNod = Math.max(maxNod, life.nod)
    maxGaze = Math.max(
      maxGaze,
      lastFace.eyeLookInLeft,
      lastFace.eyeLookOutLeft,
      lastFace.eyeLookUpLeft,
      lastFace.eyeLookDownLeft
    )
    now = t
  }
  return { maxBlink, maxBrow, maxBreath, maxNod, maxGaze, lastFace, now }
}

// 1. Idle behavior: blinks, breathing, micro-saccades, and subtle warmth
let seed = 0.17
const pseudoRand = () => {
  seed = (seed * 9301 + 49297) % 233280
  return seed / 233280
}
const idle = new AvatarLife(pseudoRand)
const idleRun = step(idle, 'Idle', 8)
assert.ok(idleRun.maxBlink > 0.7, `idle blink ${idleRun.maxBlink}`)
assert.ok(idleRun.maxBrow < 0.02, `idle brow ${idleRun.maxBrow}`)
assert.ok(idleRun.maxBreath > 0.3, `idle breath ${idleRun.maxBreath}`)
assert.ok(idleRun.maxGaze > 0.01, `idle gaze saccades ${idleRun.maxGaze}`)
assert.equal(idle.weights.jawOpen, 0)
assert.equal(idle.weights.mouthPressLeft, 0)

// 2. Listening behavior: asymmetric brow lift, nod, warm attentive expression
const listen = new AvatarLife(() => 0.5)
step(listen, 'Idle', 0.2)
const listenRun = step(listen, 'Listening', 2)
assert.ok(listenRun.maxBrow > 0.4, `listen brow ${listenRun.maxBrow}`)
assert.ok(listen.weights.mouthShrugLower > 0.08, `listen hmm ${listen.weights.mouthShrugLower}`)
assert.ok(listenRun.maxNod > 0.08, `listen nod ${listenRun.maxNod}`)
assert.ok(listen.weights.browOuterUpLeft > listen.weights.browOuterUpRight)

// 3. Thinking behavior: inner brow lift and cognitive upward gaze
const think = new AvatarLife(() => 0.5)
step(think, 'Thinking', 1.2)
assert.ok(think.weights.browInnerUp > 0.2, `think brow ${think.weights.browInnerUp}`)
assert.ok(think.weights.eyeLookUpLeft > 0.1, `think gaze ${think.weights.eyeLookUpLeft}`)

// 4. Transition from Listening to Responding and speech cross-fading
const back = new AvatarLife(() => 0.5)
step(back, 'Listening', 1.2)
assert.ok(back.nod > 0.02, `nod before speech ${back.nod}`)
const speechFrame = blankFace()
speechFrame.jawOpen = 0.55
speechFrame.mouthFunnel = 0.3
speechFrame.eyeWideLeft = 0.4
speechFrame.eyeBlinkLeft = 0.9
speechFrame.eyeBlinkRight = 0.9
const speakRun = step(back, 'Responding', 0.5, 1 / 60, speechFrame)
assert.ok(back.nod < 0.02, `nod during speech ${back.nod}`)
assert.ok(back.speechPresence > 0.9, `speechPresence ramped up ${back.speechPresence}`)
assert.ok(speakRun.lastFace.jawOpen > 0.45, `jawOpen blended ${speakRun.lastFace.jawOpen}`)
// eyeWide is suppressed when speech frame is blinking
assert.ok(speakRun.lastFace.eyeWideLeft < 0.1, `eyeWide damped by blink ${speakRun.lastFace.eyeWideLeft}`)
// Smooth cross-fade back to Idle when speech frames end (null)
const fadeStep = back.tick('Idle', (back as any).lastNow + 1 / 60, null)
assert.ok(
  fadeStep.jawOpen > 0.1 && fadeStep.jawOpen < 0.55,
  `smooth release instead of hard snap: ${fadeStep.jawOpen}`
)

// 5. 3-axis bone rotation (pitch, yaw, roll) and neck distribution
const head = { quaternion: { x: 0, y: 0, z: 0, w: 1 }, position: { y: 4.48 } }
const neck = { quaternion: { x: 0, y: 0, z: 0, w: 1 }, position: { y: 18 } }
const posed = new AvatarLife(() => 0.8)
step(posed, 'Listening', 1.5)
posed.applyBones(head, neck)
assert.ok(Math.abs(head.quaternion.x) > 0.01, `head pitch ${head.quaternion.x}`)
assert.ok(Math.abs(head.quaternion.y) > 0.001, `head yaw ${head.quaternion.y}`)
assert.ok(Math.abs(neck.quaternion.y) > 0.0005, `neck yaw ${neck.quaternion.y}`)
assert.ok(Math.abs(head.position.y - 4.48) > 0.05, `breath offset ${head.position.y}`)
const norm = Math.hypot(head.quaternion.x, head.quaternion.y, head.quaternion.z, head.quaternion.w)
assert.ok(Math.abs(norm - 1) < 1e-6, `quat norm ${norm}`)

// 6. Mixer wrapper attachment
let updates = 0
const bone = { name: 'head', quaternion: { x: 0, y: 0, z: 0, w: 1 }, position: { y: 0 } }
const neckBone = { name: 'neckLower', quaternion: { x: 0, y: 0, z: 0, w: 1 }, position: { y: 0 } }
const life = new AvatarLife(() => 0.5)
const mixer = {
  update(delta: number) {
    updates += delta
    bone.quaternion.x = 0
    bone.quaternion.y = 0
    bone.quaternion.z = 0
    bone.quaternion.w = 1
    bone.position.y = 4
  },
}
const renderer = { mixer, viewer: { skinModel: { skeleton: { bones: [bone, neckBone] } } } }
assert.equal(attachAvatarLife(renderer, life), true)
assert.equal(attachAvatarLife(renderer, life), true)
step(life, 'Idle', 2)
mixer.update(1 / 60)
assert.equal(updates, 1 / 60)
assert.ok(Math.abs(bone.position.y - 4) > 0.05, `wrapped breath ${bone.position.y}`)

// 7. Webcam user tracking: eyes lead fast, head/neck follow with smooth inertia
const tracked = new AvatarLife(() => 0.5)
tracked.setUserTarget(0.8, 0.5, true)
step(tracked, 'Listening', 0.8)
assert.ok(tracked.gazeX > 0.1, `tracked gazeX looks toward user ${tracked.gazeX}`)
assert.ok(tracked.trackHeadX > 0.08, `tracked head yaw turns toward user ${tracked.trackHeadX}`)
assert.ok(tracked.trackHeadY < -0.02, `tracked head pitch tilts up ${tracked.trackHeadY}`)
assert.ok(
  tracked.weights.eyeLookOutRight > 0.1 && tracked.weights.eyeLookInLeft > 0.1,
  `eye morphs follow user right: ${tracked.weights.eyeLookOutRight}`
)
// Releasing user target smoothly returns toward 0
tracked.setUserTarget(0, 0, false)
step(tracked, 'Listening', 1.2)
assert.ok(Math.abs(tracked.trackHeadX) < 0.02, `head returns to center ${tracked.trackHeadX}`)

// 8. Skin + motion centroid detector on synthetic 64x48 frame (face on left side of camera -> user on right side of screen)
const w = 64
const h = 48
const rgba = new Uint8ClampedArray(w * h * 4)
for (let py = 10; py < 28; py++) {
  for (let px = 8; px < 22; px++) {
    const idx = (py * w + px) * 4
    rgba[idx] = 215
    rgba[idx + 1] = 160
    rgba[idx + 2] = 135
    rgba[idx + 3] = 255
  }
}
const centroid = computeSkinAndMotionCentroid(rgba, null, w, h)
assert.ok(centroid.confidence > 0.2, `skin confidence ${centroid.confidence}`)
assert.ok(centroid.x > 0.3, `mirrored x > 0 when face is on left of raw camera: ${centroid.x}`)

// 9. Persistent mood afterglow between turns (e.g., [joy] and [sadness])
const moody = new AvatarLife(() => 0.5)
moody.setEmotion('joy')
step(moody, 'Idle', 0.8)
assert.ok(
  moody.weights.mouthSmileLeft > 0.15,
  `joy mood persists in Idle: ${moody.weights.mouthSmileLeft}`
)
moody.setEmotion('sadness')
step(moody, 'Idle', 0.8)
assert.ok(
  moody.weights.browInnerUp > 0.15 && moody.weights.mouthFrownLeft > 0.08,
  `sadness mood persists in Idle: brow=${moody.weights.browInnerUp}, frown=${moody.weights.mouthFrownLeft}`
)

// 10. Empathetic nod & smile mimicry from webcam gesture (only nods in Listening, never in Idle)
const nodSamples = [
  { t: 0, x: 0.01, y: 0.0 },
  { t: 120, x: 0.01, y: -0.06 },
  { t: 240, x: 0.02, y: -0.11 },
  { t: 360, x: 0.01, y: 0.02 },
  { t: 480, x: 0.0, y: 0.07 },
  { t: 600, x: 0.01, y: -0.04 },
  { t: 720, x: 0.01, y: 0.01 },
]
assert.equal(detectNodGesture(nodSamples), true, 'nod gesture detected from vertical oscillation')
const empathetic = new AvatarLife(() => 0.5)
empathetic.setUserTarget(0.1, 0.0, true, { nodding: true, smile: 0.7 })
const idleNoNod = step(empathetic, 'Idle', 0.6)
assert.equal(idleNoNod.maxNod, 0, `never nods in Idle: ${idleNoNod.maxNod}`)
assert.ok(
  empathetic.weights.mouthSmileLeft > 0.1,
  `empathetic smile mirrored: ${empathetic.weights.mouthSmileLeft}`
)
step(empathetic, 'Listening', 0.1)
empathetic.setUserTarget(0.1, 0.0, true, { nodding: true, smile: 0.7 })
const empListenRun = step(empathetic, 'Listening', 0.6)
assert.ok(empListenRun.maxNod > 0.05, `empathetic nod triggered in Listening: ${empListenRun.maxNod}`)

// 11. Non-accumulation when Three.js PropertyMixer skips unchanged/paused bone frames
const pausedBone = { name: 'head', quaternion: { x: 0, y: 0, z: 0, w: 1 }, position: { y: 4.48 } }
const pausedNeck = { name: 'neckLower', quaternion: { x: 0, y: 0, z: 0, w: 1 }, position: { y: 18.25 } }
const pausedLife = new AvatarLife(() => 0.5)
const pausedMixer = {
  // Simulates Three.js PropertyMixer when an animation action is paused or clamped:
  // it does NOT overwrite bone.position or bone.quaternion!
  update(_delta: number) {},
}
const pausedRenderer = {
  mixer: pausedMixer,
  chatState: 'Listening',
  viewer: { skinModel: { skeleton: { bones: [pausedBone, pausedNeck] } } },
}
assert.equal(attachAvatarLife(pausedRenderer, pausedLife), true)
for (let i = 0; i < 300; i++) {
  pausedLife.tick('Listening', i / 60)
  pausedMixer.update(1 / 60)
}
assert.ok(
  Math.abs(pausedBone.position.y - 4.48) < 0.6,
  `head position never drifts unboundedly when mixer is paused: ${pausedBone.position.y}`
)
assert.ok(
  Math.abs(pausedBone.quaternion.x) < 0.25,
  `head pitch never accumulates unboundedly when mixer is paused: ${pausedBone.quaternion.x}`
)

// 12. sanitizeAnimManager unblocks Think stage 2 deadlock and registers listen.actions
class FakeAnimManager {
  static actions: any[] = []
  static LastAction: any = undefined
  static CurPlaying: string | undefined = 'Thinking'
  static UnPauseAllActions() {
    for (const a of FakeAnimManager.actions) a.paused = false
  }
  static PrepareCrossFade(start: any, end: any, _dur: number) {
    if (start === end) throw new Error('Self crossfade should be guarded')
    end.paused = false
    end.enabled = true
  }
  hello = { isPlaying: false, stage: 0, actions: [] }
  idle = { isPlaying: false, stage: 0, actions: [] }
  listen = {
    isPlaying: true,
    stage: 0,
    actions: [{ paused: true, enabled: false, play() {} }],
  }
  think = {
    isPlaying: true,
    stage: 2,
    actions: [
      { time: 0, paused: false, enabled: true, getClip: () => ({ duration: 1.17 }), play() {} },
      { time: 0, paused: false, enabled: true, getClip: () => ({ duration: 3.0 }), play() {} },
      { time: 0.5, paused: true, enabled: true, getClip: () => ({ duration: 1.33 }), play() {} },
    ],
    update(state: string) {
      if (!this.isPlaying && state === 'Thinking') {
        this.stage = 0
        this.isPlaying = true
      }
    },
  }
  speak = { isPlaying: false, stage: 0, actions: [] }
}
const fakeAnim = new FakeAnimManager()
sanitizeAnimManager(fakeAnim, 'Thinking')
assert.equal(
  FakeAnimManager.actions.includes(fakeAnim.listen.actions[0]),
  true,
  'listen action registered in AnimationManager.actions'
)
assert.equal(fakeAnim.listen.actions[0].paused, false, 'listen action unpaused')
assert.equal(fakeAnim.think.stage, 0, 'think stage 2 deadlock reset back to stage 0 when Thinking')
// Self-crossfade is safely handled without throwing
FakeAnimManager.PrepareCrossFade(fakeAnim.listen.actions[0], fakeAnim.listen.actions[0], 0.2)

console.log('avatarLife ok')
