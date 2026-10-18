import assert from 'node:assert/strict'
import { AvatarLife, attachAvatarLife } from './avatarLife.ts'

function step(life: AvatarLife, state: string, seconds: number, stepSeconds = 1 / 60) {
  let now = 0
  life.tick(state, now)
  let maxBlink = 0
  let maxBrow = 0
  let maxBreath = 0
  let maxNod = 0
  for (let t = stepSeconds; t <= seconds; t += stepSeconds) {
    const face = life.tick(state, t)
    maxBlink = Math.max(maxBlink, face.eyeBlinkLeft)
    maxBrow = Math.max(maxBrow, face.browOuterUpLeft)
    maxBreath = Math.max(maxBreath, Math.abs(life.breath))
    maxNod = Math.max(maxNod, life.nod)
    now = t
  }
  return { maxBlink, maxBrow, maxBreath, maxNod, now }
}

const idle = new AvatarLife(() => 0.5)
const idleRun = step(idle, 'Idle', 8)
assert.ok(idleRun.maxBlink > 0.7, `idle blink ${idleRun.maxBlink}`)
assert.ok(idleRun.maxBrow < 0.02, `idle brow ${idleRun.maxBrow}`)
assert.ok(idleRun.maxBreath > 0.3, `idle breath ${idleRun.maxBreath}`)
assert.equal(idle.weights.jawOpen, 0)
assert.equal(idle.weights.mouthPressLeft, 0)

const listen = new AvatarLife(() => 0.5)
step(listen, 'Idle', 0.2)
const listenRun = step(listen, 'Listening', 2)
assert.ok(listenRun.maxBrow > 0.4, `listen brow ${listenRun.maxBrow}`)
assert.ok(listen.weights.mouthShrugLower > 0.08, `listen hmm ${listen.weights.mouthShrugLower}`)
assert.ok(listenRun.maxNod > 0.08, `listen nod ${listenRun.maxNod}`)
assert.ok(listen.weights.browOuterUpLeft > listen.weights.browOuterUpRight)

const think = new AvatarLife(() => 0.5)
step(think, 'Thinking', 1.2)
assert.ok(think.weights.browInnerUp > 0.2, `think brow ${think.weights.browInnerUp}`)
assert.ok(think.weights.eyeLookUpLeft > 0.1, `think gaze ${think.weights.eyeLookUpLeft}`)

const back = new AvatarLife(() => 0.5)
step(back, 'Listening', 1.2)
assert.ok(back.nod > 0.02, `nod before speech ${back.nod}`)
step(back, 'Responding', 0.45)
assert.ok(back.nod < 0.02, `nod during speech ${back.nod}`)
assert.ok(back.weights.browOuterUpLeft < 0.05, `brow during speech ${back.weights.browOuterUpLeft}`)
assert.equal(back.weights.eyeBlinkLeft, 0)

const head = { quaternion: { x: 0, y: 0, z: 0, w: 1 }, position: { y: 4.48 } }
const neck = { quaternion: { x: 0, y: 0, z: 0, w: 1 }, position: { y: 18 } }
const posed = new AvatarLife(() => 0.5)
step(posed, 'Listening', 1.2)
posed.applyBones(head, neck)
assert.ok(Math.abs(head.quaternion.x) > 0.01, `head pitch ${head.quaternion.x}`)
assert.ok(Math.abs(head.position.y - 4.48) > 0.05, `breath offset ${head.position.y}`)
const norm = Math.hypot(head.quaternion.x, head.quaternion.y, head.quaternion.z, head.quaternion.w)
assert.ok(Math.abs(norm - 1) < 1e-6, `quat norm ${norm}`)

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

console.log('avatarLife ok')
