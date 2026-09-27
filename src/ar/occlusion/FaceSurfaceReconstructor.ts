import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { MIRRORED_CANONICAL_VERTICES, CANONICAL_VERTEX_COUNT } from './canonicalFaceModel'
import {
  cmPerNormalizedX,
  landmarkRay,
  trackerFrustum,
} from './TrackerProjection'

/** Minimal pose shape (structurally compatible with FacePose). */
export interface ReconstructionPose {
  readonly position: { readonly x: number; readonly y: number; readonly z: number }
  readonly rotation: { readonly x: number; readonly y: number; readonly z: number; readonly w: number }
  readonly scale: number
}

export interface ReconstructionParams {
  readonly videoWidth: number
  readonly videoHeight: number
  readonly trackerVerticalFovDeg: number
  /** 0 = rigid canonical face, 1 = landmark relief (each capped by maxDeviationCm). */
  readonly landmarkDepthBlend: number
  readonly maxDeviationCm: number
}

export interface ReconstructionResult {
  readonly ok: boolean
  /** 0..1: how well the landmarks agree with the pose. Low => trust the rigid prior. */
  readonly confidence: number
  /** RMS landmark-vs-pose disagreement as a fraction of face width. */
  readonly fitRatio: number
  readonly distanceCm: number
}

const FAILED: ReconstructionResult = Object.freeze({ ok: false, confidence: 0, fitRatio: Infinity, distanceCm: 0 })

/** Face-oval width of the canonical model (landmarks 234 <-> 454), cm. */
const CANONICAL_FACE_WIDTH_CM = 15.33
const MIN_DISTANCE_CM = 5
const MAX_DISTANCE_CM = 300
const MIN_CAMERA_DEPTH_CM = 1
const FIT_RATIO_GOOD = 0.05
const FIT_RATIO_BAD = 0.12

/**
 * Vertices used to centre landmark depth on the pose. Upper/mid face only:
 * the jaw and mouth move with expression and would bias the centring.
 */
const STABLE_INDICES: Uint16Array = (() => {
  const list: number[] = []
  for (let i = 0; i < CANONICAL_VERTEX_COUNT; i += 1) {
    const y = MIRRORED_CANONICAL_VERTICES[i * 3 + 1]
    const z = MIRRORED_CANONICAL_VERTICES[i * 3 + 2]
    if (y >= -2.5 && z >= -1) list.push(i)
  }
  return Uint16Array.from(list)
})()

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

/**
 * Rebuilds a metrically correct face surface in the app's mirrored canonical
 * FACE-LOCAL space.
 *
 * Model
 * -----
 * For every canonical vertex i the pose gives its expected camera-space
 * position E_i = s R c_i + t. That is metrically exact but generic (average
 * face). The landmarks add the person's actual shape:
 *
 *   - screen position: the landmark's own ray (registration to the video is
 *     exact by construction, depth errors only slide a vertex ALONG its ray);
 *   - depth: E_i.z plus a bounded fraction of the landmark-derived relief.
 *
 * Landmark relief (-z * cmPerNormalizedX) is measured relative to the head
 * centre, so both relief sets are mean-centred over a stable upper-face subset
 * before differencing. The residual is clamped to +-maxDeviationCm, so any
 * scale error in the landmark z (for instance a wrong tracker FOV) has bounded
 * influence, and its weight is multiplied by the pose/landmark agreement
 * confidence so garbage landmarks (a hand covering the face) fall back to the
 * rigid canonical face instead of corrupting the occluder.
 *
 * The result is returned in face-local coordinates (inverse pose applied) so
 * that it can be smoothed independently of head motion and then placed with the
 * SAME smoothed pose as the glasses.
 */
export class FaceSurfaceReconstructor {
  /** Face-local positions, xyz-interleaved, CANONICAL_VERTEX_COUNT vertices. */
  readonly local = new Float32Array(CANONICAL_VERTEX_COUNT * 3)

  private readonly expectedZ = new Float32Array(CANONICAL_VERTEX_COUNT)
  private readonly landmarkRelief = new Float32Array(CANONICAL_VERTEX_COUNT)
  private readonly ray = { x: 0, y: 0 }

  reconstruct(
    landmarks: readonly NormalizedLandmark[],
    pose: ReconstructionPose,
    params: ReconstructionParams,
  ): ReconstructionResult {
    if (landmarks.length < CANONICAL_VERTEX_COUNT) return FAILED

    const { position: t, rotation: q } = pose
    const s = pose.scale
    const values = [t.x, t.y, t.z, q.x, q.y, q.z, q.w, s]
    for (const v of values) if (!Number.isFinite(v)) return FAILED

    const distance = -t.z
    if (!(s > 0) || distance < MIN_DISTANCE_CM || distance > MAX_DISTANCE_CM) return FAILED

    // Rotation matrix from the (normalised) quaternion.
    const qn = Math.hypot(q.x, q.y, q.z, q.w)
    if (!(qn > 1e-6)) return FAILED
    const qx = q.x / qn
    const qy = q.y / qn
    const qz = q.z / qn
    const qw = q.w / qn
    const r00 = 1 - 2 * (qy * qy + qz * qz)
    const r01 = 2 * (qx * qy - qz * qw)
    const r02 = 2 * (qx * qz + qy * qw)
    const r10 = 2 * (qx * qy + qz * qw)
    const r11 = 1 - 2 * (qx * qx + qz * qz)
    const r12 = 2 * (qy * qz - qx * qw)
    const r20 = 2 * (qx * qz - qy * qw)
    const r21 = 2 * (qy * qz + qx * qw)
    const r22 = 1 - 2 * (qx * qx + qy * qy)

    const frustum = trackerFrustum(params.videoWidth, params.videoHeight, params.trackerVerticalFovDeg)
    const cmPerUnit = cmPerNormalizedX(distance, frustum)
    const c = MIRRORED_CANONICAL_VERTICES

    // Pass 1: expected depth + landmark relief.
    for (let i = 0; i < CANONICAL_VERTEX_COUNT; i += 1) {
      const lm = landmarks[i]
      if (!Number.isFinite(lm.x) || !Number.isFinite(lm.y) || !Number.isFinite(lm.z)) return FAILED
      const o = i * 3
      this.expectedZ[i] = s * (r20 * c[o] + r21 * c[o + 1] + r22 * c[o + 2]) + t.z
      this.landmarkRelief[i] = -lm.z * cmPerUnit
    }

    let meanExpected = 0
    let meanRelief = 0
    for (let k = 0; k < STABLE_INDICES.length; k += 1) {
      const i = STABLE_INDICES[k]
      meanExpected += this.expectedZ[i]
      meanRelief += this.landmarkRelief[i]
    }
    meanExpected /= STABLE_INDICES.length
    meanRelief /= STABLE_INDICES.length

    // Pass 2: pose/landmark agreement (angular, in tracker tan-units).
    let sumSq = 0
    for (let k = 0; k < STABLE_INDICES.length; k += 1) {
      const i = STABLE_INDICES[k]
      const o = i * 3
      const ex = s * (r00 * c[o] + r01 * c[o + 1] + r02 * c[o + 2]) + t.x
      const ey = s * (r10 * c[o] + r11 * c[o + 1] + r12 * c[o + 2]) + t.y
      const ez = this.expectedZ[i]
      landmarkRay(landmarks[i].x, landmarks[i].y, frustum, this.ray)
      const dx = this.ray.x - ex / -ez
      const dy = this.ray.y - ey / -ez
      sumSq += dx * dx + dy * dy
    }
    const rmsAngular = Math.sqrt(sumSq / STABLE_INDICES.length)
    const fitRatio = (rmsAngular * distance) / (CANONICAL_FACE_WIDTH_CM * s)
    const confidence = 1 - smoothstep(FIT_RATIO_GOOD, FIT_RATIO_BAD, fitRatio)
    const weight = Math.min(1, Math.max(0, params.landmarkDepthBlend)) * confidence
    const maxDev = Math.max(0, params.maxDeviationCm)

    // Pass 3: final camera-space point per vertex -> face-local.
    for (let i = 0; i < CANONICAL_VERTEX_COUNT; i += 1) {
      const residual = this.landmarkRelief[i] - meanRelief - (this.expectedZ[i] - meanExpected)
      const clamped = residual < -maxDev ? -maxDev : residual > maxDev ? maxDev : residual
      let z = this.expectedZ[i] + weight * clamped
      if (z > -MIN_CAMERA_DEPTH_CM) z = -MIN_CAMERA_DEPTH_CM

      landmarkRay(landmarks[i].x, landmarks[i].y, frustum, this.ray)
      const px = this.ray.x * -z - t.x
      const py = this.ray.y * -z - t.y
      const pz = z - t.z

      // local = R^T (P - t) / s
      const o = i * 3
      this.local[o] = (r00 * px + r10 * py + r20 * pz) / s
      this.local[o + 1] = (r01 * px + r11 * py + r21 * pz) / s
      this.local[o + 2] = (r02 * px + r12 * py + r22 * pz) / s
    }

    return { ok: true, confidence, fitRatio, distanceCm: distance }
  }
}
