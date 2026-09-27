import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { computeCoverMapping } from './TrackerProjection'

/**
 * Pure, GPU-free and ML-free helpers for the foreground (hair / hand / object)
 * occlusion mask. Kept separate so the parts that can silently go wrong (UV
 * mirroring, cover cropping, gate shape, class-layout detection) are unit
 * tested.
 */

/** MediaPipe FACE_LANDMARKS_FACE_OVAL, in polygon order. */
export const FACE_OVAL_INDICES: readonly number[] = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

/** selfie_multiclass_256x256 class indices. */
export const SEGMENT_CLASS = Object.freeze({
  background: 0,
  hair: 1,
  bodySkin: 2,
  faceSkin: 3,
  clothes: 4,
  others: 5,
} as const)
export const SEGMENT_CLASS_COUNT = 6

/** Even-odd scanline fill of a polygon into a Uint8 mask (0 / 255). */
export function fillPolygon(
  out: Uint8Array,
  width: number,
  height: number,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  count: number,
): void {
  out.fill(0)
  if (count < 3) return

  const crossings: number[] = []
  for (let row = 0; row < height; row += 1) {
    const y = row + 0.5
    crossings.length = 0
    for (let i = 0; i < count; i += 1) {
      const j = (i + 1) % count
      const y0 = ys[i]
      const y1 = ys[j]
      if ((y0 <= y && y1 > y) || (y1 <= y && y0 > y)) {
        crossings.push(xs[i] + ((y - y0) / (y1 - y0)) * (xs[j] - xs[i]))
      }
    }
    crossings.sort((a, b) => a - b)
    for (let k = 0; k + 1 < crossings.length; k += 2) {
      const x0 = Math.max(0, Math.round(crossings[k]))
      const x1 = Math.min(width, Math.round(crossings[k + 1]))
      for (let x = x0; x < x1; x += 1) out[row * width + x] = 255
    }
  }
}

/** One separable box-blur pass (edge-clamped). `tmp` must be as large as `src`. */
export function boxBlur(
  src: Uint8Array,
  dst: Uint8Array,
  tmp: Uint8Array,
  width: number,
  height: number,
  radius: number,
): void {
  const r = Math.max(0, Math.floor(radius))
  if (r === 0) {
    dst.set(src)
    return
  }
  const span = 2 * r + 1

  for (let y = 0; y < height; y += 1) {
    const row = y * width
    let sum = 0
    for (let k = -r; k <= r; k += 1) sum += src[row + Math.min(width - 1, Math.max(0, k))]
    for (let x = 0; x < width; x += 1) {
      tmp[row + x] = sum / span
      sum += src[row + Math.min(width - 1, x + r + 1)] - src[row + Math.max(0, x - r)]
    }
  }

  for (let x = 0; x < width; x += 1) {
    let sum = 0
    for (let k = -r; k <= r; k += 1) sum += tmp[Math.min(height - 1, Math.max(0, k)) * width + x]
    for (let y = 0; y < height; y += 1) {
      dst[y * width + x] = sum / span
      sum +=
        tmp[Math.min(height - 1, y + r + 1) * width + x] - tmp[Math.max(0, y - r) * width + x]
    }
  }
}

export interface GateResult {
  /** Face-oval width in mask pixels (0 if the polygon was degenerate). */
  readonly faceWidthPx: number
  readonly ok: boolean
}

/**
 * Soft "where a face is" gate in un-mirrored mask space: ~1 inside the face
 * oval, ~0.5 at its border, decaying outward. Two thresholds of this one
 * channel give both an "inside the face" gate (bangs, strands) and a wider
 * ring gate (fingers on the temple, a phone beside the face).
 *
 * `scratch` needs 3 buffers the size of the mask.
 */
export function computeFaceGate(
  landmarks: readonly NormalizedLandmark[],
  width: number,
  height: number,
  out: Uint8Array,
  scratch: { a: Uint8Array; b: Uint8Array; tmp: Uint8Array },
): GateResult {
  const n = FACE_OVAL_INDICES.length
  const xs = new Float32Array(n)
  const ys = new Float32Array(n)
  let minX = Infinity
  let maxX = -Infinity

  for (let i = 0; i < n; i += 1) {
    const lm = landmarks[FACE_OVAL_INDICES[i]]
    if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) {
      out.fill(0)
      return { faceWidthPx: 0, ok: false }
    }
    xs[i] = lm.x * width
    ys[i] = lm.y * height
    minX = Math.min(minX, xs[i])
    maxX = Math.max(maxX, xs[i])
  }

  const faceWidthPx = maxX - minX
  if (!(faceWidthPx > 2)) {
    out.fill(0)
    return { faceWidthPx: 0, ok: false }
  }

  fillPolygon(scratch.a, width, height, xs, ys, n)
  const radius = Math.min(40, Math.max(2, 0.12 * faceWidthPx))
  // Two passes ~ a triangular kernel: smoother falloff than one box.
  boxBlur(scratch.a, scratch.b, scratch.tmp, width, height, radius)
  boxBlur(scratch.b, out, scratch.tmp, width, height, radius)
  return { faceWidthPx, ok: true }
}

export interface ScreenToMaskParams {
  /** Drawing-buffer size in device pixels. */
  readonly viewportWidth: number
  readonly viewportHeight: number
  /** Video source size (for the cover crop). */
  readonly videoWidth: number
  readonly videoHeight: number
  /** Motion compensation: face displacement (normalised, un-mirrored) since the mask was captured. */
  readonly shiftX?: number
  readonly shiftY?: number
}

/**
 * JS reference of the mapping the fragment shader performs.
 *
 * `fragX`/`fragY` are WebGL `gl_FragCoord` values (origin bottom-left, device
 * pixels). The result is a UV into the mask, whose row 0 is the TOP of the
 * un-mirrored camera frame. The display is mirrored, so u = 1 - u_display.
 * Returns null when the pixel is outside the video (cover crop).
 */
export function screenToMaskUv(
  fragX: number,
  fragY: number,
  p: ScreenToMaskParams,
): { u: number; v: number } | null {
  const cover = computeCoverMapping(p.viewportWidth, p.viewportHeight, p.videoWidth, p.videoHeight)
  const px = fragX
  const py = p.viewportHeight - fragY
  const uDisplay = (px - cover.offsetX) / cover.renderedWidth
  const vDisplay = (py - cover.offsetY) / cover.renderedHeight
  if (uDisplay < 0 || uDisplay > 1 || vDisplay < 0 || vDisplay > 1) return null
  return { u: 1 - uDisplay - (p.shiftX ?? 0), v: vDisplay - (p.shiftY ?? 0) }
}

/**
 * Does the class layout look like selfie_multiclass? A known MediaPipe issue
 * scrambles class indices on some GPU delegates (notably iOS Safari). Inside the
 * face interior the dominant class must be face-skin; if not, reading "hair" or
 * "body-skin" from the wrong channel would occlude the glasses with clothing or
 * background, so the caller must disable the mask.
 *
 * `classMeans` = mean confidence per class over the face interior.
 */
export function classLayoutLooksValid(classMeans: ArrayLike<number>): boolean {
  let best = -1
  let bestValue = -Infinity
  for (let c = 0; c < SEGMENT_CLASS_COUNT; c += 1) {
    if (classMeans[c] > bestValue) {
      bestValue = classMeans[c]
      best = c
    }
  }
  return best === SEGMENT_CLASS.faceSkin && bestValue > 0.4
}
