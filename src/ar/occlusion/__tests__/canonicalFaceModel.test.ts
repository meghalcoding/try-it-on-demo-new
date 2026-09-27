import { describe, expect, it } from 'vitest'
import {
  CANONICAL_TRIANGLES,
  CANONICAL_TRIANGLE_COUNT,
  CANONICAL_VERTEX_COUNT,
  CANONICAL_VERTICES,
  MIRRORED_CANONICAL_LATERAL_BIAS,
  MIRRORED_CANONICAL_VERTICES,
} from '../canonicalFaceModel'

describe('canonicalFaceModel', () => {
  it('has the documented MediaPipe vertex/triangle counts', () => {
    expect(CANONICAL_VERTEX_COUNT).toBe(468)
    expect(CANONICAL_TRIANGLE_COUNT).toBe(898)
    expect(CANONICAL_VERTICES.length).toBe(468 * 3)
    expect(CANONICAL_TRIANGLES.length).toBe(898 * 3)
  })

  it('every triangle index is in range', () => {
    for (let i = 0; i < CANONICAL_TRIANGLES.length; i += 1) {
      expect(CANONICAL_TRIANGLES[i]).toBeGreaterThanOrEqual(0)
      expect(CANONICAL_TRIANGLES[i]).toBeLessThan(CANONICAL_VERTEX_COUNT)
    }
  })

  it('nose tip (landmark 1) sits at the expected published coordinates', () => {
    expect(CANONICAL_VERTICES[1 * 3]).toBeCloseTo(0, 3)
    expect(CANONICAL_VERTICES[1 * 3 + 1]).toBeCloseTo(-1.126865, 3)
    expect(CANONICAL_VERTICES[1 * 3 + 2]).toBeCloseTo(7.475604, 3)
  })

  it('the model is roughly face-width by face-height by face-depth, in centimetres', () => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
    for (let i = 0; i < CANONICAL_VERTEX_COUNT; i += 1) {
      const x = CANONICAL_VERTICES[i * 3], y = CANONICAL_VERTICES[i * 3 + 1], z = CANONICAL_VERTICES[i * 3 + 2]
      minX = Math.min(minX, x); maxX = Math.max(maxX, x)
      minY = Math.min(minY, y); maxY = Math.max(maxY, y)
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z)
    }
    expect(maxX - minX).toBeGreaterThan(10)
    expect(maxX - minX).toBeLessThan(20)
    expect(maxY - minY).toBeGreaterThan(15)
    expect(maxY - minY).toBeLessThan(25)
    expect(maxZ - minZ).toBeGreaterThan(5)
    expect(maxZ - minZ).toBeLessThan(15)
  })

  it('MIRRORED_CANONICAL_VERTICES is CANONICAL_VERTICES with X negated', () => {
    expect(MIRRORED_CANONICAL_VERTICES.length).toBe(CANONICAL_VERTICES.length)
    for (let i = 0; i < CANONICAL_VERTEX_COUNT; i += 1) {
      expect(MIRRORED_CANONICAL_VERTICES[i * 3]).toBeCloseTo(-CANONICAL_VERTICES[i * 3], 6)
      expect(MIRRORED_CANONICAL_VERTICES[i * 3 + 1]).toBeCloseTo(CANONICAL_VERTICES[i * 3 + 1], 6)
      expect(MIRRORED_CANONICAL_VERTICES[i * 3 + 2]).toBeCloseTo(CANONICAL_VERTICES[i * 3 + 2], 6)
    }
  })

  it('lateral bias is ~0 at the sagittal centre and ~1 at the cheek/temple edge', () => {
    expect(MIRRORED_CANONICAL_LATERAL_BIAS.length).toBe(CANONICAL_VERTEX_COUNT)
    // Nose tip / nose bridge / chin: dead centre, must stay tight.
    for (const i of [1, 6, 152]) {
      expect(MIRRORED_CANONICAL_LATERAL_BIAS[i]).toBeLessThan(0.05)
    }
    // Face-oval side landmarks (right/left cheek edge, right/left temple): must be generous.
    for (const i of [234, 454, 127, 356]) {
      expect(MIRRORED_CANONICAL_LATERAL_BIAS[i]).toBeGreaterThan(0.9)
    }
    // Every value stays in [0, 1] and increases with |x| (monotonic in the aggregate).
    let previousBucketMax = 0
    const buckets = new Map<number, number>()
    for (let i = 0; i < CANONICAL_VERTEX_COUNT; i += 1) {
      const bias = MIRRORED_CANONICAL_LATERAL_BIAS[i]
      expect(bias).toBeGreaterThanOrEqual(0)
      expect(bias).toBeLessThanOrEqual(1)
      const bucket = Math.round((Math.abs(MIRRORED_CANONICAL_VERTICES[i * 3]) / 8) * 10)
      buckets.set(bucket, Math.max(buckets.get(bucket) ?? 0, bias))
    }
    for (const key of [...buckets.keys()].sort((a, b) => a - b)) {
      const value = buckets.get(key)!
      expect(value).toBeGreaterThanOrEqual(previousBucketMax - 1e-6)
      previousBucketMax = value
    }
  })

  it('is a genuinely closed-ish mesh (most edges shared by exactly two triangles)', () => {
    const edgeCount = new Map<string, number>()
    for (let t = 0; t < CANONICAL_TRIANGLES.length; t += 3) {
      const tri = [CANONICAL_TRIANGLES[t], CANONICAL_TRIANGLES[t + 1], CANONICAL_TRIANGLES[t + 2]]
      for (let e = 0; e < 3; e += 1) {
        const a = tri[e], b = tri[(e + 1) % 3]
        const key = a < b ? `${a}-${b}` : `${b}-${a}`
        edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1)
      }
    }
    let sharedByTwo = 0
    for (const count of edgeCount.values()) if (count === 2) sharedByTwo += 1
    // The face mask has a boundary (mouth interior, oval edge), so not literally
    // watertight, but the interior should be triangulated consistently.
    expect(sharedByTwo / edgeCount.size).toBeGreaterThan(0.85)
  })
})
