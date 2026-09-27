import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { FaceOccluder } from '../FaceOccluder'
import { DEFAULT_FACE_OCCLUSION_SETTINGS } from '../../types/FaceOcclusion'
import { synthesize } from '../occlusion/__tests__/syntheticFace'
import type { FacePose } from '../FacePose'

function toFacePose(pose: ReturnType<typeof synthesize>['pose'], timestampMs: number): FacePose {
  return { ...pose, trackingState: 'detected', timestampMs }
}

function makeOccluder() {
  const occluder = new FaceOccluder()
  occluder.setVideoSourceDimensions(1280, 720)
  occluder.setSurfaceDimensions(1280, 720)
  return occluder
}

describe('FaceOccluder (hardened mode)', () => {
  it('has no surface before any detection', () => {
    const occluder = makeOccluder()
    expect(occluder.hasFaceSurface()).toBe(false)
    expect(occluder.getDiagnostics().hasSurface).toBe(false)
  })

  it('builds a surface after a valid ingest and exposes it via getPoseRoot()', () => {
    const occluder = makeOccluder()
    const face = synthesize({ yawDeg: 10, distanceCm: 45 })
    occluder.ingest(face.landmarks, face.pose, 1000)
    expect(occluder.hasFaceSurface()).toBe(true)
    expect(occluder.getDiagnostics().ingested).toBe(1)
    expect(occluder.getDiagnostics().rejected).toBe(0)

    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 2000)
    occluder.update(face.landmarks, toFacePose(face.pose, 1000), camera, 1000)
    expect(occluder.getPoseRoot().visible).toBe(true)
  })

  it('rejects a malformed pose without throwing and without corrupting existing state', () => {
    const occluder = makeOccluder()
    const face = synthesize({ yawDeg: 0, distanceCm: 45 })
    occluder.ingest(face.landmarks, face.pose, 1000)
    expect(occluder.hasFaceSurface()).toBe(true)

    const badPose = { position: { x: NaN, y: 0, z: -45 }, rotation: { x: 0, y: 0, z: 0, w: 1 }, scale: 1 }
    expect(() => occluder.ingest(face.landmarks, badPose, 1001)).not.toThrow()
    expect(occluder.getDiagnostics().rejected).toBe(1)
    expect(occluder.hasFaceSurface()).toBe(true)
  })

  it('hides the pose root when the pose is stale beyond the hold window', () => {
    const occluder = makeOccluder()
    const face = synthesize({ yawDeg: 0, distanceCm: 45 })
    occluder.ingest(face.landmarks, face.pose, 1000)
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 2000)

    occluder.update(face.landmarks, toFacePose(face.pose, 1000), camera, 1000)
    expect(occluder.getPoseRoot().visible).toBe(true)

    occluder.update(face.landmarks, toFacePose(face.pose, 5000), camera, 5000)
    expect(occluder.getPoseRoot().visible).toBe(false)
  })

  it('reset() drops the surface and hides the pose root', () => {
    const occluder = makeOccluder()
    const face = synthesize({ yawDeg: 0, distanceCm: 45 })
    occluder.ingest(face.landmarks, face.pose, 1000)
    expect(occluder.hasFaceSurface()).toBe(true)

    occluder.reset()
    expect(occluder.hasFaceSurface()).toBe(false)
    expect(occluder.getPoseRoot().visible).toBe(false)
  })

  it('update(null pose) hides everything and resets', () => {
    const occluder = makeOccluder()
    const face = synthesize({ yawDeg: 0, distanceCm: 45 })
    occluder.ingest(face.landmarks, face.pose, 1000)
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 2000)
    occluder.update(null, null, camera, 2000)
    expect(occluder.getPoseRoot().visible).toBe(false)
    expect(occluder.hasFaceSurface()).toBe(false)
  })

  it('respects enabled=false: ingest is a no-op and the pose root stays hidden', () => {
    const occluder = makeOccluder()
    occluder.setSettings({ ...DEFAULT_FACE_OCCLUSION_SETTINGS, enabled: false })
    const face = synthesize({ yawDeg: 0, distanceCm: 45 })
    occluder.ingest(face.landmarks, face.pose, 1000)
    expect(occluder.hasFaceSurface()).toBe(false)

    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 2000)
    occluder.update(face.landmarks, toFacePose(face.pose, 1000), camera, 1000)
    expect(occluder.getPoseRoot().visible).toBe(false)
  })

  it('switching to legacy mode disables the hardened pose root and switches to the legacy mesh', () => {
    const occluder = makeOccluder()
    const face = synthesize({ yawDeg: 0, distanceCm: 45 })
    occluder.ingest(face.landmarks, face.pose, 1000)
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 2000)
    occluder.update(face.landmarks, toFacePose(face.pose, 1000), camera, 1000)
    expect(occluder.getPoseRoot().visible).toBe(true)

    occluder.setSettings({ ...DEFAULT_FACE_OCCLUSION_SETTINGS, mode: 'legacy' })
    occluder.update(face.landmarks, toFacePose(face.pose, 1001), camera, 1001)
    expect(occluder.getPoseRoot().visible).toBe(false)
    expect(occluder.getDiagnostics().mode).toBe('legacy')
  })

  it('does not throw across a long sequence of frames including gaps and re-detections', () => {
    const occluder = makeOccluder()
    const camera = new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 2000)
    let t = 0
    expect(() => {
      for (let i = 0; i < 50; i += 1) {
        t += 33
        if (i % 7 === 0) {
          occluder.update(null, null, camera, t)
          continue
        }
        const face = synthesize({ yawDeg: (i % 10) * 5, distanceCm: 40 + (i % 5), seed: i })
        occluder.ingest(face.landmarks, face.pose, t)
        occluder.update(face.landmarks, toFacePose(face.pose, t), camera, t)
      }
    }).not.toThrow()
  })

  it('the face geometry carries an aLateral attribute (drives graduated nose/temple clearance)', () => {
    const occluder = makeOccluder()
    const geometry = occluder.getFaceGeometry()
    const attribute = geometry.getAttribute('aLateral')
    expect(attribute).toBeDefined()
    expect(attribute.count).toBe(geometry.getAttribute('position').count)
    // Values must be a real 0..1 gradient, not a constant (or the temple fix is a no-op).
    let min = Infinity
    let max = -Infinity
    for (let i = 0; i < attribute.count; i += 1) {
      min = Math.min(min, attribute.getX(i))
      max = Math.max(max, attribute.getX(i))
    }
    expect(min).toBeCloseTo(0, 5)
    expect(max).toBeCloseTo(1, 5)
  })

  it('dispose() is safe to call and does not throw', () => {
    const occluder = makeOccluder()
    const face = synthesize({ yawDeg: 0, distanceCm: 45 })
    occluder.ingest(face.landmarks, face.pose, 1000)
    expect(() => occluder.dispose()).not.toThrow()
  })
})
