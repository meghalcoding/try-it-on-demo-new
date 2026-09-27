import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { FaceSurfaceReconstructor } from '../FaceSurfaceReconstructor'
import { MIRRORED_CANONICAL_VERTICES } from '../canonicalFaceModel'
import { synthesize } from './syntheticFace'
import { FaceOccluder as LegacyFaceOccluder } from '../LegacyFaceOccluder'
import { DEFAULT_FACE_OCCLUSION_SETTINGS } from '../../../types/FaceOcclusion'

const STABLE = (i: number) =>
  !(MIRRORED_CANONICAL_VERTICES[i * 3 + 1] < -2.5 || MIRRORED_CANONICAL_VERTICES[i * 3 + 2] < -1)

function placeAndMeasure(
  reconstructor: FaceSurfaceReconstructor,
  pose: ReturnType<typeof synthesize>['pose'],
  truth: Float32Array,
): { meanAbsErrCm: number; p95AbsErrCm: number } {
  const matrix = new THREE.Matrix4().compose(
    new THREE.Vector3(pose.position.x, pose.position.y, pose.position.z),
    new THREE.Quaternion(pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w),
    new THREE.Vector3().setScalar(pose.scale),
  )
  const v = new THREE.Vector3()
  const errors: number[] = []
  for (let i = 0; i < 468; i += 1) {
    if (!STABLE(i)) continue
    v.set(reconstructor.local[i * 3], reconstructor.local[i * 3 + 1], reconstructor.local[i * 3 + 2])
    v.applyMatrix4(matrix)
    errors.push(Math.abs(v.z - truth[i * 3 + 2]))
  }
  errors.sort((a, b) => a - b)
  const mean = errors.reduce((sum, e) => sum + e, 0) / errors.length
  return { meanAbsErrCm: mean, p95AbsErrCm: errors[Math.floor(errors.length * 0.95)] }
}

describe('FaceSurfaceReconstructor', () => {
  it('rejects malformed input instead of throwing', () => {
    const reconstructor = new FaceSurfaceReconstructor()
    const params = { videoWidth: 1280, videoHeight: 720, trackerVerticalFovDeg: 63, landmarkDepthBlend: 0.5, maxDeviationCm: 0.8 }
    const badPose = { position: { x: 0, y: 0, z: -45 }, rotation: { x: 0, y: 0, z: 0, w: 1 }, scale: 1 }

    expect(reconstructor.reconstruct([], badPose, params).ok).toBe(false)
    expect(
      reconstructor.reconstruct([], { ...badPose, scale: Number.NaN }, params).ok,
    ).toBe(false)
    expect(
      reconstructor.reconstruct([], { ...badPose, position: { x: 0, y: 0, z: 0 } }, params).ok,
    ).toBe(false) // distance below MIN_DISTANCE_CM
  })

  it('recovers sub-centimetre depth accuracy for a personalised face at realistic noise', () => {
    const reconstructor = new FaceSurfaceReconstructor()
    const face = synthesize({
      yawDeg: 20,
      distanceCm: 45,
      personalize: true,
      zNoiseCm: 0.3,
      xyNoisePx: 0.7,
      zOriginBiasCm: 0.8,
    })
    const result = reconstructor.reconstruct(face.landmarks, face.pose, {
      videoWidth: 1280,
      videoHeight: 720,
      trackerVerticalFovDeg: 63,
      landmarkDepthBlend: 0.5,
      maxDeviationCm: 0.8,
    })
    expect(result.ok).toBe(true)
    expect(result.confidence).toBeGreaterThan(0.9)

    const { meanAbsErrCm, p95AbsErrCm } = placeAndMeasure(reconstructor, face.pose, face.truth)
    expect(meanAbsErrCm).toBeLessThan(0.3)
    expect(p95AbsErrCm).toBeLessThan(0.6)
  })

  it('is far more accurate than the previous (legacy) screen-space occluder', () => {
    // Reproduces the failure mode of the pre-hardening occluder: it derived
    // depth from a fixed ~14cm reference face width regardless of actual
    // distance, so it is only correct when the face fills the whole frame
    // width -- wrong by design at a normal 45cm chat distance.
    const face = synthesize({
      yawDeg: 30,
      distanceCm: 45,
      personalize: true,
      zNoiseCm: 0.3,
      xyNoisePx: 0.7,
      zOriginBiasCm: 0.8,
    })

    const legacy = new LegacyFaceOccluder()
    legacy.setVideoSourceDimensions(1280, 720)
    legacy.setSurfaceDimensions(1280, 720)
    legacy.setSettings(DEFAULT_FACE_OCCLUSION_SETTINGS)
    const camera = new THREE.PerspectiveCamera(60, 1280 / 720, 0.1, 2000)
    legacy.update(
      face.landmarks,
      { ...face.pose, trackingState: 'detected', timestampMs: 1000 } as never,
      camera,
      1000,
    )
    const legacyPosition = (legacy.getObject3D() as THREE.Mesh).geometry.getAttribute('position')
    let legacySum = 0
    let legacyN = 0
    for (let i = 0; i < 468; i += 1) {
      if (!STABLE(i)) continue
      legacySum += Math.abs(legacyPosition.getZ(i) - face.truth[i * 3 + 2])
      legacyN += 1
    }
    const legacyMeanAbsErrCm = legacySum / legacyN

    const reconstructor = new FaceSurfaceReconstructor()
    reconstructor.reconstruct(face.landmarks, face.pose, {
      videoWidth: 1280,
      videoHeight: 720,
      trackerVerticalFovDeg: 63,
      landmarkDepthBlend: 0.5,
      maxDeviationCm: 0.8,
    })
    const { meanAbsErrCm } = placeAndMeasure(reconstructor, face.pose, face.truth)

    // Sanity: the legacy model really is off by centimetres at this distance.
    expect(legacyMeanAbsErrCm).toBeGreaterThan(1)
    // The hardened reconstructor should be at least 5x more accurate.
    expect(meanAbsErrCm).toBeLessThan(legacyMeanAbsErrCm / 5)
  })

  it('bounds landmark deviation to maxDeviationCm even with adversarial landmark noise', () => {
    const reconstructor = new FaceSurfaceReconstructor()
    const face = synthesize({ yawDeg: 0, distanceCm: 45, personalize: false, zNoiseCm: 5 })
    const maxDev = 0.4
    const result = reconstructor.reconstruct(face.landmarks, face.pose, {
      videoWidth: 1280,
      videoHeight: 720,
      trackerVerticalFovDeg: 63,
      landmarkDepthBlend: 1,
      maxDeviationCm: maxDev,
    })
    expect(result.ok).toBe(true)

    // Compare against blend=0 (pure rigid prior) for the same pose: no vertex
    // should differ from the rigid prior by more than maxDeviationCm (+ numerical slack).
    const rigid = new FaceSurfaceReconstructor()
    rigid.reconstruct(face.landmarks, face.pose, {
      videoWidth: 1280,
      videoHeight: 720,
      trackerVerticalFovDeg: 63,
      landmarkDepthBlend: 0,
      maxDeviationCm: maxDev,
    })

    for (let i = 0; i < 468; i += 1) {
      const dz = Math.abs(reconstructor.local[i * 3 + 2] - rigid.local[i * 3 + 2])
      expect(dz).toBeLessThanOrEqual(maxDev + 1e-6)
    }
  })

  it('confidence drops when landmarks disagree with the pose (e.g. an occluded/garbage face)', () => {
    const reconstructor = new FaceSurfaceReconstructor()
    const face = synthesize({ yawDeg: 0, distanceCm: 45 })
    const params = { videoWidth: 1280, videoHeight: 720, trackerVerticalFovDeg: 63, landmarkDepthBlend: 0.5, maxDeviationCm: 0.8 }

    const clean = reconstructor.reconstruct(face.landmarks, face.pose, params)
    expect(clean.confidence).toBeGreaterThan(0.9)

    // Corrupt landmark x/y (as if a hand covered part of the face) but keep the pose.
    const corrupted = face.landmarks.map((lm, i) => (i % 3 === 0 ? { ...lm, x: 0.5, y: 0.5 } : lm))
    const dirty = reconstructor.reconstruct(corrupted, face.pose, params)
    expect(dirty.ok).toBe(true)
    expect(dirty.confidence).toBeLessThan(clean.confidence)
  })

  it('is robust to a wrong assumed tracker FOV (bounded degradation via maxDeviationCm)', () => {
    const face = synthesize({ yawDeg: 20, distanceCm: 45, personalize: true, trueVfovDeg: 72 })
    const wrongFov = new FaceSurfaceReconstructor()
    wrongFov.reconstruct(face.landmarks, face.pose, {
      videoWidth: 1280,
      videoHeight: 720,
      trackerVerticalFovDeg: 63, // wrong on purpose
      landmarkDepthBlend: 0.5,
      maxDeviationCm: 0.8,
    })
    const { meanAbsErrCm } = placeAndMeasure(wrongFov, face.pose, face.truth)
    // Bounded by the rigid-prior error plus the deviation cap; must not blow up.
    expect(meanAbsErrCm).toBeLessThan(1.0)
  })
})
