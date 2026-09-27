import * as THREE from 'three'
import type { Calibration } from '../../types/Calibration'

/**
 * Fit diagnostics: how much of the eyewear frame is EMBEDDED in the face.
 *
 * Why this exists: with an accurate face surface the occluder correctly hides
 * anything that lies behind the skin. If a product's calibration sinks the
 * frame into the face, that used to be masked by an occluder that sat ~4 cm too
 * deep; now it shows up as parts of the frame being cut away. That is
 * technically correct behaviour but reads as an occlusion bug, so it must be
 * measurable.
 *
 * Everything is evaluated in FACE space (the canonical face-local frame, before
 * the pose is applied), so it does not depend on head rotation or distance:
 *
 *   q = calibration.translation + calibration.scale * R_calibration * p_model
 *
 * which is exactly what GlassesAnchor + the pose compose, divided by the pose.
 */

export const HEIGHT_FIELD = Object.freeze({
  minX: -9,
  maxX: 9,
  minY: -10,
  maxY: 10,
  cell: 0.25,
})

export class FaceHeightField {
  readonly columns = Math.round((HEIGHT_FIELD.maxX - HEIGHT_FIELD.minX) / HEIGHT_FIELD.cell)
  readonly rows = Math.round((HEIGHT_FIELD.maxY - HEIGHT_FIELD.minY) / HEIGHT_FIELD.cell)
  private readonly z = new Float32Array(this.columns * this.rows)
  private valid = false

  isValid(): boolean {
    return this.valid
  }

  /** Front-most surface z per cell, from face-local triangles. */
  rebuild(positions: ArrayLike<number>, triangles: ArrayLike<number>): void {
    this.z.fill(-Infinity)
    const { minX, minY, cell } = HEIGHT_FIELD

    for (let t = 0; t + 2 < triangles.length; t += 3) {
      const ia = triangles[t] * 3
      const ib = triangles[t + 1] * 3
      const ic = triangles[t + 2] * 3
      const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2]
      const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2]
      const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2]

      const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
      if (Math.abs(det) < 1e-9) continue

      const c0 = Math.max(0, Math.floor((Math.min(ax, bx, cx) - minX) / cell))
      const c1 = Math.min(this.columns - 1, Math.ceil((Math.max(ax, bx, cx) - minX) / cell))
      const r0 = Math.max(0, Math.floor((Math.min(ay, by, cy) - minY) / cell))
      const r1 = Math.min(this.rows - 1, Math.ceil((Math.max(ay, by, cy) - minY) / cell))

      for (let r = r0; r <= r1; r += 1) {
        const y = minY + (r + 0.5) * cell
        for (let c = c0; c <= c1; c += 1) {
          const x = minX + (c + 0.5) * cell
          const w0 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det
          const w1 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det
          const w2 = 1 - w0 - w1
          if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue
          const z = w0 * az + w1 * bz + w2 * cz
          const index = r * this.columns + c
          if (z > this.z[index]) this.z[index] = z
        }
      }
    }
    this.valid = true
  }

  /** Surface z at (x, y), or null outside the face footprint or before rebuild(). */
  sample(x: number, y: number): number | null {
    if (!this.valid) return null
    const { minX, minY, cell } = HEIGHT_FIELD
    const c = Math.floor((x - minX) / cell)
    const r = Math.floor((y - minY) / cell)
    if (c < 0 || r < 0 || c >= this.columns || r >= this.rows) return null
    const z = this.z[r * this.columns + c]
    return z === -Infinity ? null : z
  }
}

/** Sample opaque frame vertices in the model root's local space. */
export function collectFrameSamples(root: THREE.Object3D, maxSamples = 1500): Float32Array {
  root.updateMatrixWorld(true)
  const inverseRoot = new THREE.Matrix4().copy(root.matrixWorld).invert()
  const meshes: THREE.Mesh[] = []
  let totalVertices = 0

  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    const materials = Array.isArray(object.material) ? object.material : [object.material]
    // Lenses are meant to sit in front of the eye; only the frame can embed.
    if (materials.some((m: THREE.Material) => m.transparent || m.opacity < 0.9)) return
    const position = object.geometry.getAttribute('position')
    if (!position) return
    meshes.push(object)
    totalVertices += position.count
  })

  const stride = Math.max(1, Math.floor(totalVertices / maxSamples))
  const out: number[] = []
  const v = new THREE.Vector3()
  const toRoot = new THREE.Matrix4()

  for (const mesh of meshes) {
    toRoot.multiplyMatrices(inverseRoot, mesh.matrixWorld)
    const position = mesh.geometry.getAttribute('position')
    for (let i = 0; i < position.count; i += stride) {
      v.fromBufferAttribute(position, i).applyMatrix4(toRoot)
      out.push(v.x, v.y, v.z)
    }
  }
  return Float32Array.from(out)
}

export type FitVerdict = 'ok' | 'marginal' | 'embedded' | 'unknown'

export interface FitReport {
  readonly verdict: FitVerdict
  readonly considered: number
  readonly embeddedFraction: number
  readonly maxPenetrationCm: number
  readonly p90PenetrationCm: number
  /** Adding this to calibration.z moves ~90% of the embedded vertices out of the skin. */
  readonly suggestedForwardCm: number
}

export const UNKNOWN_FIT: FitReport = Object.freeze({
  verdict: 'unknown',
  considered: 0,
  embeddedFraction: 0,
  maxPenetrationCm: 0,
  p90PenetrationCm: 0,
  suggestedForwardCm: 0,
})

/** Vertices further behind the skin than this are arms/temples, not embedding. */
const MAX_EMBED_DEPTH_CM = 3
/** Skin contact of a few mm (pads) is normal. */
const TOLERANCE_CM = 0.3
/** Only the front of the frame over the central face can embed in the nose/brow. */
const MAX_ABS_X_CM = 6
const MIN_FRONT_Z_CM = 1.5

export function computeFitReport(
  frameSamples: Float32Array,
  calibration: Calibration,
  field: FaceHeightField,
): FitReport {
  if (!field.isValid() || frameSamples.length < 3) return UNKNOWN_FIT

  const rotation = new THREE.Matrix4().makeRotationFromEuler(
    new THREE.Euler(calibration.rotationX, calibration.rotationY, calibration.rotationZ, 'XYZ'),
  )
  const q = new THREE.Vector3()
  const penetrations: number[] = []
  let considered = 0

  for (let i = 0; i < frameSamples.length; i += 3) {
    q.set(frameSamples[i], frameSamples[i + 1], frameSamples[i + 2])
      .multiplyScalar(calibration.scale)
      .applyMatrix4(rotation)
    q.x += calibration.x
    q.y += calibration.y
    q.z += calibration.z

    if (Math.abs(q.x) > MAX_ABS_X_CM || q.z < MIN_FRONT_Z_CM) continue
    const surface = field.sample(q.x, q.y)
    if (surface === null) continue

    considered += 1
    const penetration = surface - q.z
    if (penetration > TOLERANCE_CM && penetration < MAX_EMBED_DEPTH_CM) penetrations.push(penetration)
  }

  if (considered < 20) return UNKNOWN_FIT

  penetrations.sort((a, b) => a - b)
  const embeddedFraction = penetrations.length / considered
  const p90 = penetrations.length ? penetrations[Math.floor(penetrations.length * 0.9)] : 0
  const max = penetrations.length ? penetrations[penetrations.length - 1] : 0

  const verdict: FitVerdict =
    embeddedFraction < 0.02 ? 'ok' : embeddedFraction < 0.15 ? 'marginal' : 'embedded'

  return {
    verdict,
    considered,
    embeddedFraction,
    maxPenetrationCm: max,
    p90PenetrationCm: p90,
    suggestedForwardCm: verdict === 'ok' ? 0 : p90 + TOLERANCE_CM,
  }
}
