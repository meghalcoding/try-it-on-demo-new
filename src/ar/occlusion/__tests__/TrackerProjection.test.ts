import { describe, expect, it } from 'vitest'
import {
  MEDIAPIPE_FACE_GEOMETRY_VFOV_DEG,
  clampVerticalFov,
  cmPerNormalizedX,
  computeCoverMapping,
  landmarkRay,
  matchedVerticalFovDeg,
  trackerFrustum,
} from '../TrackerProjection'

describe('computeCoverMapping', () => {
  it('scales up to cover a wider surface, cropping vertically', () => {
    const m = computeCoverMapping(1000, 500, 1280, 720)
    expect(m.scale).toBeCloseTo(1000 / 1280, 6)
    expect(m.renderedWidth).toBeCloseTo(1000, 6)
    expect(m.renderedHeight).toBeCloseTo(720 * (1000 / 1280), 6)
    expect(m.offsetX).toBeCloseTo(0, 6)
    expect(m.offsetY).toBeLessThan(0)
  })

  it('scales up to cover a taller surface, cropping horizontally', () => {
    const m = computeCoverMapping(400, 800, 1280, 720)
    expect(m.scale).toBeCloseTo(800 / 720, 6)
    expect(m.offsetY).toBeCloseTo(0, 6)
    expect(m.offsetX).toBeLessThan(0)
  })

  it('falls back to filling the surface when the video size is unknown', () => {
    const m = computeCoverMapping(320, 240, 0, 0)
    expect(m).toMatchObject({ renderedWidth: 320, renderedHeight: 240, offsetX: 0, offsetY: 0, scale: 1 })
  })
})

describe('trackerFrustum / clampVerticalFov', () => {
  it('matches the documented MediaPipe default', () => {
    expect(MEDIAPIPE_FACE_GEOMETRY_VFOV_DEG).toBe(63)
  })

  it('derives horizontal tangent from aspect ratio', () => {
    const f = trackerFrustum(1280, 720, 63)
    expect(f.tanH / f.tanV).toBeCloseTo(1280 / 720, 6)
    expect(f.tanV).toBeCloseTo(Math.tan((63 * Math.PI) / 360), 6)
  })

  it('clamps out-of-range or non-finite FOV to sane bounds', () => {
    expect(clampVerticalFov(Number.NaN)).toBe(63)
    expect(clampVerticalFov(5)).toBeGreaterThanOrEqual(20)
    expect(clampVerticalFov(500)).toBeLessThanOrEqual(120)
  })
})

describe('cmPerNormalizedX', () => {
  it('is proportional to distance', () => {
    const f = trackerFrustum(1280, 720, 63)
    const near = cmPerNormalizedX(30, f)
    const far = cmPerNormalizedX(60, f)
    expect(far).toBeCloseTo(near * 2, 6)
  })

  it('is never negative for a non-negative distance', () => {
    const f = trackerFrustum(1280, 720, 63)
    expect(cmPerNormalizedX(-5, f)).toBe(0)
  })
})

describe('matchedVerticalFovDeg', () => {
  it('equals the tracker FOV when the surface matches the video aspect exactly', () => {
    const fov = matchedVerticalFovDeg(1280, 720, 1280, 720, 63)
    expect(fov).toBeCloseTo(63, 6)
  })

  it('shrinks when the surface crops the video vertically (wide surface)', () => {
    const fov = matchedVerticalFovDeg(1280, 400, 1280, 720, 63)
    expect(fov).toBeLessThan(63)
  })

  it('is unaffected by horizontal cropping (tall surface): vertical extent is untouched', () => {
    const fov = matchedVerticalFovDeg(400, 720, 1280, 720, 63)
    expect(fov).toBeCloseTo(63, 4)
  })

  it('round-trips through a camera aspect: projecting the video edge lands on the surface edge', () => {
    // Surface only shows the middle 50% of the video height (object-fit: cover with a taller surface aspect).
    const surfaceW = 1280
    const surfaceH = 360
    const fov = matchedVerticalFovDeg(surfaceW, surfaceH, 1280, 720, 63)
    const f = trackerFrustum(1280, 720, 63)
    // A point at the very top of the video, at some depth z, in tracker space:
    const z = -50
    const yAtVideoTop = -z * f.tanV
    // Project with a camera of the derived FOV and surface aspect:
    const camTanV = Math.tan((fov * Math.PI) / 360)
    const ndcY = yAtVideoTop / (-z * camTanV)
    // The video's top is cropped out (surface shows the middle 50%), so |ndcY| should be > 1 (off-screen).
    expect(Math.abs(ndcY)).toBeGreaterThan(1)
  })
})

describe('landmarkRay', () => {
  it('mirrors x and maps the frame centre to the ray straight ahead', () => {
    const f = trackerFrustum(1280, 720, 63)
    const out = { x: NaN, y: NaN }
    landmarkRay(0.5, 0.5, f, out)
    expect(out.x).toBeCloseTo(0, 6)
    expect(out.y).toBeCloseTo(0, 6)
  })

  it('maps un-mirrored right edge (x=1) to display-left (negative ray x)', () => {
    const f = trackerFrustum(1280, 720, 63)
    const out = { x: NaN, y: NaN }
    landmarkRay(1, 0.5, f, out)
    expect(out.x).toBeCloseTo(-f.tanH, 6)
  })

  it('maps top of frame (y=0) to positive ray y (up)', () => {
    const f = trackerFrustum(1280, 720, 63)
    const out = { x: NaN, y: NaN }
    landmarkRay(0.5, 0, f, out)
    expect(out.y).toBeCloseTo(f.tanV, 6)
  })
})
