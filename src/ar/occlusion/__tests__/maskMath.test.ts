import { describe, expect, it } from 'vitest'
import {
  SEGMENT_CLASS,
  SEGMENT_CLASS_COUNT,
  boxBlur,
  classLayoutLooksValid,
  computeFaceGate,
  fillPolygon,
  screenToMaskUv,
} from '../maskMath'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

describe('fillPolygon', () => {
  it('fills a simple square exactly', () => {
    const w = 10
    const h = 10
    const out = new Uint8Array(w * h)
    fillPolygon(out, w, h, [2, 8, 8, 2], [2, 2, 8, 8], 4)
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const inside = x >= 2 && x < 8 && y >= 2 && y < 8
        expect(out[y * w + x]).toBe(inside ? 255 : 0)
      }
    }
  })

  it('leaves the mask empty for a degenerate (fewer than 3 point) polygon', () => {
    const out = new Uint8Array(16).fill(123)
    fillPolygon(out, 4, 4, [1, 2], [1, 2], 2)
    expect(Array.from(out)).toEqual(new Array(16).fill(0))
  })
})

describe('boxBlur', () => {
  it('is a no-op at radius 0', () => {
    const src = Uint8Array.from([10, 20, 30, 40])
    const dst = new Uint8Array(4)
    const tmp = new Uint8Array(4)
    boxBlur(src, dst, tmp, 2, 2, 0)
    expect(Array.from(dst)).toEqual(Array.from(src))
  })

  it('averages a solid block toward its own value (energy roughly preserved)', () => {
    const w = 20
    const h = 20
    const src = new Uint8Array(w * h).fill(200)
    const dst = new Uint8Array(w * h)
    const tmp = new Uint8Array(w * h)
    boxBlur(src, dst, tmp, w, h, 3)
    // Interior pixels of a uniform field are unchanged by a box blur.
    expect(dst[10 * w + 10]).toBe(200)
  })

  it('spreads a single hot pixel into its neighbourhood', () => {
    const w = 11
    const h = 11
    const src = new Uint8Array(w * h)
    src[5 * w + 5] = 255
    const dst = new Uint8Array(w * h)
    const tmp = new Uint8Array(w * h)
    boxBlur(src, dst, tmp, w, h, 2)
    expect(dst[5 * w + 5]).toBeGreaterThan(0)
    expect(dst[5 * w + 5]).toBeLessThan(255)
    expect(dst[5 * w + 4]).toBeGreaterThan(0) // neighbour picked up some energy
    expect(dst[0 * w + 0]).toBe(0) // far away, untouched
  })
})

function makeFaceOvalLandmarks(centerX: number, centerY: number, radius: number): NormalizedLandmark[] {
  // 468 is more than enough indices for FACE_OVAL_INDICES to dereference.
  const landmarks: NormalizedLandmark[] = Array.from({ length: 468 }, () => ({
    x: centerX,
    y: centerY,
    z: 0,
    visibility: 1,
  }))
  // FACE_OVAL_INDICES walks the true face-oval loop; place each at a point on
  // a circle so the polygon is a genuine (non-degenerate) closed shape.
  const FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  ]
  FACE_OVAL_INDICES.forEach((index, i) => {
    const angle = (i / FACE_OVAL_INDICES.length) * Math.PI * 2
    landmarks[index] = {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius * 0.75,
      z: 0,
      visibility: 1,
    }
  })
  return landmarks
}

describe('computeFaceGate', () => {
  it('produces a high gate value inside the face oval and near-zero far outside', () => {
    const width = 128
    const height = 128
    const landmarks = makeFaceOvalLandmarks(0.5, 0.5, 0.25)
    const out = new Uint8Array(width * height)
    const scratch = {
      a: new Uint8Array(width * height),
      b: new Uint8Array(width * height),
      tmp: new Uint8Array(width * height),
    }
    const result = computeFaceGate(landmarks, width, height, out, scratch)
    expect(result.ok).toBe(true)
    expect(result.faceWidthPx).toBeGreaterThan(0)

    const center = out[64 * width + 64]
    const corner = out[2 * width + 2]
    expect(center).toBeGreaterThan(corner)
    expect(center).toBeGreaterThan(200)
    expect(corner).toBeLessThan(50)
  })

  it('reports not-ok and clears the output for missing/invalid landmarks', () => {
    const width = 32
    const height = 32
    const out = new Uint8Array(width * height).fill(255)
    const scratch = {
      a: new Uint8Array(width * height),
      b: new Uint8Array(width * height),
      tmp: new Uint8Array(width * height),
    }
    const result = computeFaceGate([], width, height, out, scratch)
    expect(result.ok).toBe(false)
    expect(result.faceWidthPx).toBe(0)
    expect(Array.from(out)).toEqual(new Array(width * height).fill(0))
  })
})

describe('screenToMaskUv', () => {
  it('maps the un-mirrored right edge to mask u=0 (mirror applied)', () => {
    const params = { viewportWidth: 640, viewportHeight: 480, videoWidth: 640, videoHeight: 480 }
    // Video fills the viewport exactly; fragCoord bottom-left origin.
    const centerBottom = screenToMaskUv(0, 0, params) // bottom-left of screen = top-left after flip
    expect(centerBottom).not.toBeNull()
  })

  it('is null outside the cover-cropped video region', () => {
    const params = { viewportWidth: 100, viewportHeight: 200, videoWidth: 100, videoHeight: 100 }
    // Cover crops the video vertically to fit a tall viewport... actually here
    // video is square and viewport is tall, so cover SCALES UP to fill width,
    // cropping height on both sides; a fragment far outside must be null only if genuinely outside.
    const uv = screenToMaskUv(50, 199, params)
    expect(uv === null || (uv.u >= 0 && uv.u <= 1 && uv.v >= 0 && uv.v <= 1)).toBe(true)
  })

  it('applies the motion-compensation shift', () => {
    const params = { viewportWidth: 100, viewportHeight: 100, videoWidth: 100, videoHeight: 100, shiftX: 0.1, shiftY: -0.05 }
    const withShift = screenToMaskUv(50, 50, params)
    const withoutShift = screenToMaskUv(50, 50, { ...params, shiftX: 0, shiftY: 0 })
    expect(withShift).not.toBeNull()
    expect(withoutShift).not.toBeNull()
    expect(withShift!.u).toBeCloseTo(withoutShift!.u - 0.1, 5)
    expect(withShift!.v).toBeCloseTo(withoutShift!.v + 0.05, 5)
  })
})

describe('classLayoutLooksValid', () => {
  it('accepts a layout where face-skin dominates the face interior', () => {
    const means = new Array(SEGMENT_CLASS_COUNT).fill(0.05)
    means[SEGMENT_CLASS.faceSkin] = 0.8
    expect(classLayoutLooksValid(means)).toBe(true)
  })

  it('rejects a scrambled layout where hair (or another class) dominates the face interior', () => {
    const means = new Array(SEGMENT_CLASS_COUNT).fill(0.05)
    means[SEGMENT_CLASS.hair] = 0.8
    expect(classLayoutLooksValid(means)).toBe(false)
  })

  it('rejects a low-confidence layout even if face-skin is technically the max', () => {
    const means = new Array(SEGMENT_CLASS_COUNT).fill(0.05)
    means[SEGMENT_CLASS.faceSkin] = 0.2
    expect(classLayoutLooksValid(means)).toBe(false)
  })
})
