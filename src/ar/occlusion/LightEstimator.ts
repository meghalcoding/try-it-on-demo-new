/**
 * Cheap, deliberately conservative estimate of how the user's face is lit,
 * used to (optionally) make the glasses' lighting agree with the room.
 *
 * It is a HEURISTIC from a tiny downscaled frame, not physical light
 * estimation, so every output is smoothed and bounded and it is OFF by default.
 * What it can do: notice a dark scene (a face lit from behind is dark -> the
 * glasses, and their lens reflections, should dim instead of glowing) and
 * notice which side of the face is brighter (steer the key light and the
 * contact shadow accordingly).
 */

export interface LightingSample {
  /** Mean face luminance, 0..1. */
  readonly luminance: number
  /** (right - left) / mean, in DISPLAY (mirrored) orientation, roughly -1..1. */
  readonly horizontal: number
  /** (top - bottom) / mean, roughly -1..1. */
  readonly vertical: number
  /** Pixels that contributed; 0 means "no estimate". */
  readonly samples: number
}

export interface LightingEstimate {
  /** Multiplier for ambient/key intensity, bounded. */
  readonly intensityScale: number
  /** Key-light direction offset, camera space, bounded to [-1, 1]. */
  readonly directionX: number
  readonly directionY: number
}

export const NEUTRAL_LIGHTING: LightingEstimate = Object.freeze({
  intensityScale: 1,
  directionX: 0,
  directionY: 0,
})

const REFERENCE_LUMINANCE = 0.45
const MIN_SCALE = 0.55
const MAX_SCALE = 1.2
const MAX_DIRECTION = 0.8

/**
 * Luminance statistics over the central part of a face box.
 *
 * @param rgba RGBA8 pixels of the (un-mirrored) downscaled frame
 * @param box  face bounding box in normalised UN-MIRRORED frame coordinates
 */
export function sampleFaceLighting(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  box: { minX: number; minY: number; maxX: number; maxY: number },
): LightingSample {
  const w = box.maxX - box.minX
  const h = box.maxY - box.minY
  if (!(w > 0) || !(h > 0)) return { luminance: 0, horizontal: 0, vertical: 0, samples: 0 }

  // Central 60% x 70% of the box: skin, away from hair and background.
  const x0 = Math.max(0, Math.floor((box.minX + 0.2 * w) * width))
  const x1 = Math.min(width, Math.ceil((box.maxX - 0.2 * w) * width))
  const y0 = Math.max(0, Math.floor((box.minY + 0.15 * h) * height))
  const y1 = Math.min(height, Math.ceil((box.maxY - 0.15 * h) * height))
  if (x1 - x0 < 2 || y1 - y0 < 2) return { luminance: 0, horizontal: 0, vertical: 0, samples: 0 }

  const midX = (x0 + x1) / 2
  const midY = (y0 + y1) / 2
  let sum = 0
  let leftSum = 0
  let rightSum = 0
  let topSum = 0
  let bottomSum = 0
  let leftN = 0
  let rightN = 0
  let topN = 0
  let bottomN = 0
  let n = 0

  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const o = (y * width + x) * 4
      const lum = (0.2126 * rgba[o] + 0.7152 * rgba[o + 1] + 0.0722 * rgba[o + 2]) / 255
      sum += lum
      n += 1
      if (x < midX) { leftSum += lum; leftN += 1 } else { rightSum += lum; rightN += 1 }
      if (y < midY) { topSum += lum; topN += 1 } else { bottomSum += lum; bottomN += 1 }
    }
  }

  const mean = sum / n
  const eps = 1e-3
  // Frame x is un-mirrored; the display is mirrored, so frame-left is display-right.
  const displayRight = leftN > 0 ? leftSum / leftN : mean
  const displayLeft = rightN > 0 ? rightSum / rightN : mean
  const top = topN > 0 ? topSum / topN : mean
  const bottom = bottomN > 0 ? bottomSum / bottomN : mean

  return {
    luminance: mean,
    horizontal: (displayRight - displayLeft) / (mean + eps),
    vertical: (top - bottom) / (mean + eps),
    samples: n,
  }
}

/** Map a raw sample to a bounded estimate. */
export function estimateFromSample(sample: LightingSample): LightingEstimate {
  if (sample.samples === 0) return NEUTRAL_LIGHTING
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.pow(sample.luminance / REFERENCE_LUMINANCE, 0.6)))
  const clampDir = (v: number) => Math.max(-MAX_DIRECTION, Math.min(MAX_DIRECTION, v))
  return {
    intensityScale: scale,
    directionX: clampDir(sample.horizontal),
    directionY: clampDir(sample.vertical),
  }
}

/** Exponential smoothing that is independent of the sampling rate. */
export class LightingSmoother {
  private state: LightingEstimate = NEUTRAL_LIGHTING
  private initialised = false

  constructor(private readonly halfLifeSeconds = 0.6) {}

  update(target: LightingEstimate, dtSeconds: number): LightingEstimate {
    if (!this.initialised) {
      this.state = target
      this.initialised = true
      return this.state
    }
    const alpha = 1 - Math.pow(0.5, Math.max(0, dtSeconds) / this.halfLifeSeconds)
    this.state = {
      intensityScale: this.state.intensityScale + (target.intensityScale - this.state.intensityScale) * alpha,
      directionX: this.state.directionX + (target.directionX - this.state.directionX) * alpha,
      directionY: this.state.directionY + (target.directionY - this.state.directionY) * alpha,
    }
    return this.state
  }

  reset(): void {
    this.initialised = false
    this.state = NEUTRAL_LIGHTING
  }
}

const SAMPLER_WIDTH = 64
const SAMPLER_INTERVAL_MS = 200

/** DOM side: draws the video into a 64px canvas a few times a second and measures the face. */
export class LightingSampler {
  private canvas: HTMLCanvasElement | null = null
  private ctx: CanvasRenderingContext2D | null = null
  private lastMs = -Infinity

  /** Returns a sample at most every 200 ms, otherwise null. Never throws. */
  sample(
    video: HTMLVideoElement,
    landmarks: ReadonlyArray<{ x: number; y: number }>,
    nowMs: number,
  ): LightingSample | null {
    if (nowMs - this.lastMs < SAMPLER_INTERVAL_MS) return null
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !(video.videoWidth > 0)) return null
    this.lastMs = nowMs

    try {
      const height = Math.max(8, Math.round((SAMPLER_WIDTH * video.videoHeight) / video.videoWidth))
      if (!this.canvas || this.canvas.height !== height) {
        this.canvas = document.createElement('canvas')
        this.canvas.width = SAMPLER_WIDTH
        this.canvas.height = height
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })
      }
      if (!this.ctx) return null

      let minX = 1
      let minY = 1
      let maxX = 0
      let maxY = 0
      for (const lm of landmarks) {
        minX = Math.min(minX, lm.x)
        minY = Math.min(minY, lm.y)
        maxX = Math.max(maxX, lm.x)
        maxY = Math.max(maxY, lm.y)
      }

      this.ctx.drawImage(video, 0, 0, SAMPLER_WIDTH, height)
      const image = this.ctx.getImageData(0, 0, SAMPLER_WIDTH, height)
      return sampleFaceLighting(image.data, SAMPLER_WIDTH, height, { minX, minY, maxX, maxY })
    } catch {
      return null
    }
  }
}
