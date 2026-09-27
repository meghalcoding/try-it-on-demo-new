import * as THREE from 'three'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { MIRRORED_CANONICAL_VERTICES } from '../canonicalFaceModel'
import { trackerFrustum, cmPerNormalizedX } from '../TrackerProjection'

/** Deterministic PRNG so tests are reproducible. */
export function rng(seed: number) {
  let s = seed >>> 0
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 }
}
export function gauss(r: () => number) {
  return Math.sqrt(-2 * Math.log(Math.max(1e-9, r()))) * Math.cos(2 * Math.PI * r())
}

export interface SynthOptions {
  yawDeg: number; pitchDeg?: number; distanceCm: number
  videoW?: number; videoH?: number
  /** Camera vfov used to generate the image (truth). */
  trueVfovDeg?: number
  /** Personal shape deviation applied to the canonical face (face-local cm). */
  personalize?: boolean
  zNoiseCm?: number; xyNoisePx?: number
  /** Unknown offset between MediaPipe's landmark-z origin and the pose origin, cm. */
  zOriginBiasCm?: number
  poseNoiseCm?: number; poseNoiseDeg?: number
  seed?: number
  faceScale?: number
}

export function personalDeformation(i: number): [number, number, number] {
  const c = MIRRORED_CANONICAL_VERTICES
  const x = c[i * 3], y = c[i * 3 + 1], z = c[i * 3 + 2]
  // longer, more prominent nose (+7 mm at tip, tapering), fuller cheeks, deeper brow.
  const nose = Math.exp(-((x * x) / 4 + ((y + 1) * (y + 1)) / 9)) * (z > 4 ? 1 : 0)
  const cheek = Math.exp(-(((Math.abs(x) - 5) ** 2) / 6 + ((y + 1.5) ** 2) / 10))
  return [0, 0.05 * nose, 0.7 * nose + 0.35 * cheek - 0.15 * Math.max(0, y - 4) * (z > 3 ? 1 : 0)]
}

export interface SynthFace {
  landmarks: NormalizedLandmark[]
  pose: { position: { x: number; y: number; z: number }; rotation: { x: number; y: number; z: number; w: number }; scale: number }
  /** true camera-space vertex positions (xyz interleaved) */
  truth: Float32Array
  truePose: SynthFace['pose']
}

export function synthesize(o: SynthOptions): SynthFace {
  const r = rng(o.seed ?? 7)
  const vw = o.videoW ?? 1280, vh = o.videoH ?? 720
  const fr = trackerFrustum(vw, vh, o.trueVfovDeg ?? 63)
  const s = o.faceScale ?? 1
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler((o.pitchDeg ?? 0) * Math.PI / 180, o.yawDeg * Math.PI / 180, 0, 'YXZ'))
  const t = new THREE.Vector3(1.2, 0.8, -o.distanceCm)
  const M = new THREE.Matrix4().compose(t, q, new THREE.Vector3(s, s, s))
  const W = cmPerNormalizedX(o.distanceCm, fr)
  const truth = new Float32Array(468 * 3)
  const landmarks: NormalizedLandmark[] = []
  const v = new THREE.Vector3()
  for (let i = 0; i < 468; i++) {
    v.set(MIRRORED_CANONICAL_VERTICES[i * 3], MIRRORED_CANONICAL_VERTICES[i * 3 + 1], MIRRORED_CANONICAL_VERTICES[i * 3 + 2])
    if (o.personalize) { const d = personalDeformation(i); v.x += d[0]; v.y += d[1]; v.z += d[2] }
    v.applyMatrix4(M)
    truth[i * 3] = v.x; truth[i * 3 + 1] = v.y; truth[i * 3 + 2] = v.z
    const nx = v.x / (-v.z * fr.tanH), ny = v.y / (-v.z * fr.tanV)
    const xm = (nx + 1) / 2 + gauss(r) * (o.xyNoisePx ?? 0) / vw
    const y = (1 - ny) / 2 + gauss(r) * (o.xyNoisePx ?? 0) / vh
    const relZcm = (v.z - t.z) - (o.zOriginBiasCm ?? 0) + gauss(r) * (o.zNoiseCm ?? 0)
    landmarks.push({ x: 1 - xm, y, z: -relZcm / W, visibility: 1 })
  }
  for (let i = 0; i < 10; i++) landmarks.push({ x: 0.5, y: 0.5, z: 0, visibility: 1 })
  const pn = o.poseNoiseCm ?? 0
  const qn = new THREE.Quaternion().setFromEuler(new THREE.Euler(gauss(r) * (o.poseNoiseDeg ?? 0) * Math.PI / 180, gauss(r) * (o.poseNoiseDeg ?? 0) * Math.PI / 180, 0))
  const qp = q.clone().multiply(qn)
  const truePose = { position: { x: t.x, y: t.y, z: t.z }, rotation: { x: q.x, y: q.y, z: q.z, w: q.w }, scale: s }
  const pose = { position: { x: t.x + gauss(r) * pn, y: t.y + gauss(r) * pn, z: t.z + gauss(r) * pn }, rotation: { x: qp.x, y: qp.y, z: qp.z, w: qp.w }, scale: s }
  return { landmarks, pose, truth, truePose }
}
