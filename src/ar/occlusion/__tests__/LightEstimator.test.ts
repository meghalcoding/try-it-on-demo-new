import { describe, expect, it } from 'vitest'
import {
  LightingSmoother,
  NEUTRAL_LIGHTING,
  estimateFromSample,
  sampleFaceLighting,
} from '../LightEstimator'

function makeFrame(width: number, height: number, fn: (x: number, y: number) => number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const v = Math.round(fn(x, y) * 255)
      const o = (y * width + x) * 4
      data[o] = v
      data[o + 1] = v
      data[o + 2] = v
      data[o + 3] = 255
    }
  }
  return data
}

describe('sampleFaceLighting', () => {
  it('returns zero samples for a degenerate box', () => {
    const data = makeFrame(10, 10, () => 0.5)
    const result = sampleFaceLighting(data, 10, 10, { minX: 0.5, minY: 0.5, maxX: 0.5, maxY: 0.5 })
    expect(result.samples).toBe(0)
  })

  it('measures uniform brightness with near-zero horizontal/vertical bias', () => {
    const data = makeFrame(40, 40, () => 0.4)
    const result = sampleFaceLighting(data, 40, 40, { minX: 0.1, minY: 0.1, maxX: 0.9, maxY: 0.9 })
    expect(result.samples).toBeGreaterThan(0)
    expect(result.luminance).toBeCloseTo(0.4, 1)
    expect(result.horizontal).toBeCloseTo(0, 1)
    expect(result.vertical).toBeCloseTo(0, 1)
  })

  it('detects a brighter frame-left half as a positive "displayRight" bias (mirrored)', () => {
    // frame-left (small x) bright, frame-right dark
    const data = makeFrame(40, 40, (x) => (x < 20 ? 0.8 : 0.2))
    const result = sampleFaceLighting(data, 40, 40, { minX: 0.1, minY: 0.1, maxX: 0.9, maxY: 0.9 })
    // frame-left maps to displayRight in the sample's convention.
    expect(result.horizontal).toBeGreaterThan(0)
  })

  it('detects a brighter top half as a positive vertical bias', () => {
    const data = makeFrame(40, 40, (_x, y) => (y < 20 ? 0.8 : 0.2))
    const result = sampleFaceLighting(data, 40, 40, { minX: 0.1, minY: 0.1, maxX: 0.9, maxY: 0.9 })
    expect(result.vertical).toBeGreaterThan(0)
  })
})

describe('estimateFromSample', () => {
  it('returns neutral for zero samples', () => {
    expect(estimateFromSample({ luminance: 0, horizontal: 0, vertical: 0, samples: 0 })).toEqual(NEUTRAL_LIGHTING)
  })

  it('scales down for a dark scene and bounds the scale', () => {
    const estimate = estimateFromSample({ luminance: 0.05, horizontal: 0, vertical: 0, samples: 100 })
    expect(estimate.intensityScale).toBeLessThan(1)
    expect(estimate.intensityScale).toBeGreaterThanOrEqual(0.55)
  })

  it('scales up for a bright scene and bounds the scale', () => {
    const estimate = estimateFromSample({ luminance: 5, horizontal: 0, vertical: 0, samples: 100 })
    expect(estimate.intensityScale).toBeLessThanOrEqual(1.2)
  })

  it('bounds direction to [-1, 1] even for extreme bias', () => {
    const estimate = estimateFromSample({ luminance: 0.4, horizontal: 50, vertical: -50, samples: 100 })
    expect(estimate.directionX).toBeLessThanOrEqual(1)
    expect(estimate.directionX).toBeGreaterThanOrEqual(-1)
    expect(estimate.directionY).toBeLessThanOrEqual(1)
    expect(estimate.directionY).toBeGreaterThanOrEqual(-1)
  })
})

describe('LightingSmoother', () => {
  it('snaps to the first sample immediately', () => {
    const smoother = new LightingSmoother(0.5)
    const result = smoother.update({ intensityScale: 0.7, directionX: 0.3, directionY: -0.2 }, 1)
    expect(result).toEqual({ intensityScale: 0.7, directionX: 0.3, directionY: -0.2 })
  })

  it('approaches the target gradually on subsequent updates', () => {
    const smoother = new LightingSmoother(0.5)
    smoother.update(NEUTRAL_LIGHTING, 0)
    const target = { intensityScale: 0.5, directionX: 1, directionY: 0 }
    const first = smoother.update(target, 0.5) // one half-life
    expect(first.intensityScale).toBeGreaterThan(0.5)
    expect(first.intensityScale).toBeLessThan(1)
    const second = smoother.update(target, 100) // effectively fully converged
    expect(second.intensityScale).toBeCloseTo(0.5, 2)
  })

  it('reset() forgets state so the next update snaps again', () => {
    const smoother = new LightingSmoother(0.5)
    smoother.update({ intensityScale: 0.5, directionX: 0, directionY: 0 }, 1)
    smoother.reset()
    const result = smoother.update({ intensityScale: 0.9, directionX: 0.1, directionY: 0.1 }, 1)
    expect(result.intensityScale).toBe(0.9)
  })
})
