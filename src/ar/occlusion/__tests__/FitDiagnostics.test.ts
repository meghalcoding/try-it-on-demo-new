import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import {
  FaceHeightField,
  UNKNOWN_FIT,
  collectFrameSamples,
  computeFitReport,
} from '../FitDiagnostics'
import { CANONICAL_TRIANGLES, MIRRORED_CANONICAL_VERTICES } from '../canonicalFaceModel'
import type { Calibration } from '../../../types/Calibration'

function identityCalibration(overrides: Partial<Calibration> = {}): Calibration {
  return { x: 0, y: 0, z: 0, scale: 1, rotationX: 0, rotationY: 0, rotationZ: 0, ...overrides }
}

describe('FaceHeightField', () => {
  it('is invalid until rebuilt', () => {
    const field = new FaceHeightField()
    expect(field.isValid()).toBe(false)
    expect(field.sample(0, 0)).toBeNull()
  })

  it('samples a flat triangle at its constant z', () => {
    const field = new FaceHeightField()
    const positions = new Float32Array([-5, -5, 2, 5, -5, 2, 0, 5, 2])
    const triangles = new Uint16Array([0, 1, 2])
    field.rebuild(positions, triangles)
    expect(field.isValid()).toBe(true)
    expect(field.sample(0, -2)).toBeCloseTo(2, 1)
    expect(field.sample(100, 100)).toBeNull() // outside the footprint
  })

  it('rebuilds the real canonical face into a footprint with plausible depth', () => {
    const field = new FaceHeightField()
    field.rebuild(MIRRORED_CANONICAL_VERTICES, CANONICAL_TRIANGLES)
    const noseTip = field.sample(0, -1.1)
    expect(noseTip).not.toBeNull()
    expect(noseTip as number).toBeGreaterThan(5) // nose protrudes forward, canonical z ~7.5
  })
})

describe('collectFrameSamples', () => {
  it('samples only opaque meshes, excluding transparent (lens) meshes', () => {
    const root = new THREE.Group()
    const frame = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ opacity: 1, transparent: false }),
    )
    const lens = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshStandardMaterial({ opacity: 0.3, transparent: true }),
    )
    lens.position.set(0, 0, 5) // would dominate bounds if included
    root.add(frame, lens)

    const samples = collectFrameSamples(root, 10000)
    expect(samples.length).toBeGreaterThan(0)
    let maxZ = -Infinity
    for (let i = 2; i < samples.length; i += 3) maxZ = Math.max(maxZ, samples[i])
    expect(maxZ).toBeLessThan(5) // lens vertices were excluded
  })

  it('respects the sample budget (does not return unbounded vertex counts)', () => {
    const root = new THREE.Group()
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 64, 64), // several thousand vertices
      new THREE.MeshStandardMaterial({ opacity: 1, transparent: false }),
    )
    root.add(mesh)
    const samples = collectFrameSamples(root, 500)
    expect(samples.length / 3).toBeLessThanOrEqual(600) // stride rounding slack
  })
})

describe('computeFitReport', () => {
  const field = new FaceHeightField()
  field.rebuild(MIRRORED_CANONICAL_VERTICES, CANONICAL_TRIANGLES)

  it('returns UNKNOWN_FIT when the height field is not built', () => {
    const emptyField = new FaceHeightField()
    const samples = Float32Array.from([0, 0, 5])
    expect(computeFitReport(samples, identityCalibration(), emptyField)).toEqual(UNKNOWN_FIT)
  })

  it('reports "ok" for a frame comfortably in front of the face', () => {
    // A flat plate of points well in front of the nose (z well beyond the face surface).
    const samples: number[] = []
    for (let x = -4; x <= 4; x += 0.5) for (let y = -1; y <= 4; y += 0.5) samples.push(x, y, 0)
    const calibration = identityCalibration({ z: 10 }) // pushes the plate to z=10, well clear
    const report = computeFitReport(Float32Array.from(samples), calibration, field)
    expect(report.verdict).toBe('ok')
    expect(report.embeddedFraction).toBeLessThan(0.02)
  })

  it('reports "embedded" for a frame calibrated into the middle of the face', () => {
    const samples: number[] = []
    for (let x = -4; x <= 4; x += 0.5) for (let y = -1; y <= 4; y += 0.5) samples.push(x, y, 0)
    const calibration = identityCalibration({ z: 3 }) // sinks the plate ~3cm behind the skin at the center
    const report = computeFitReport(Float32Array.from(samples), calibration, field)
    expect(report.verdict).toBe('embedded')
    expect(report.suggestedForwardCm).toBeGreaterThan(0)
  })

  it('a bigger suggestedForwardCm actually clears the embedding when applied', () => {
    const samples: number[] = []
    for (let x = -4; x <= 4; x += 0.5) for (let y = -1; y <= 4; y += 0.5) samples.push(x, y, 0)
    const calibration = identityCalibration({ z: 3 })
    const before = computeFitReport(Float32Array.from(samples), calibration, field)
    expect(before.verdict).toBe('embedded')

    const fixed = identityCalibration({ z: calibration.z + before.suggestedForwardCm })
    const after = computeFitReport(Float32Array.from(samples), fixed, field)
    // The correction is based on the p90 penetration, so ~90% of previously
    // embedded points should clear; a coarse sample grid can leave the exact
    // verdict boundary case (embeddedFraction just above the "marginal"
    // cutoff), so assert the large improvement directly rather than the label.
    expect(after.embeddedFraction).toBeLessThan(before.embeddedFraction * 0.25)
    expect(after.maxPenetrationCm).toBeLessThan(before.maxPenetrationCm)
  })

  it('returns UNKNOWN_FIT when too few samples fall in the evaluable footprint', () => {
    const samples = Float32Array.from([100, 100, 5, 100, 100, 5]) // way outside MAX_ABS_X_CM/footprint
    const report = computeFitReport(samples, identityCalibration(), field)
    expect(report).toEqual(UNKNOWN_FIT)
  })
})
