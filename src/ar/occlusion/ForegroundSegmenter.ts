import {
  FilesetResolver,
  ImageSegmenter,
  type ImageSegmenterResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import {
  SEGMENT_CLASS,
  SEGMENT_CLASS_COUNT,
  classLayoutLooksValid,
  computeFaceGate,
} from './maskMath'

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
const MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'

const INPUT_WIDTH = 256
const MIN_INTERVAL_MS = 33
const MAX_INTERVAL_MS = 250
/** Never spend more than this fraction of wall time inside segmentation. */
const MAX_DUTY_CYCLE = 0.25
const LAYOUT_FAILURES_BEFORE_FALLBACK = 6
const MAX_CONSECUTIVE_ERRORS = 5
const TEMPORAL_ALPHA = 0.6
const FACE_INTERIOR_GATE = 230

export type SegmenterStatus = 'idle' | 'loading' | 'ready' | 'unavailable'

/** A published mask. Row 0 is the TOP of the UN-MIRRORED camera frame. */
export interface ForegroundMaskFrame {
  readonly width: number
  readonly height: number
  /** RGBA8: R = hair, G = body-skin (hands/arms), B = clothes|others, A = face-skin. */
  readonly classes: Uint8Array
  /** Soft face-oval gate, 1 channel, same size (see computeFaceGate). */
  readonly gate: Uint8Array
  /** Face anchor at capture, normalised un-mirrored frame coords (for motion compensation). */
  readonly anchorX: number
  readonly anchorY: number
  /** Increments each time the mask content changes. */
  readonly version: number
}

/**
 * Turns per-class confidence maps into the packed, temporally smoothed mask,
 * and measures the class layout inside the face. No ML, no DOM: unit-testable.
 */
export class MaskAssembler {
  readonly width: number
  readonly height: number
  readonly classes: Uint8Array
  readonly gate: Uint8Array
  private readonly smooth: Float32Array[]
  private readonly scratch: { a: Uint8Array; b: Uint8Array; tmp: Uint8Array }
  private initialised = false
  readonly classMeans = new Float32Array(SEGMENT_CLASS_COUNT)

  constructor(width: number, height: number) {
    this.width = width
    this.height = height
    const n = width * height
    this.classes = new Uint8Array(n * 4)
    this.gate = new Uint8Array(n)
    this.smooth = Array.from({ length: 4 }, () => new Float32Array(n))
    this.scratch = { a: new Uint8Array(n), b: new Uint8Array(n), tmp: new Uint8Array(n) }
  }

  reset(): void {
    this.initialised = false
  }

  /**
   * @param confidences one Float32Array per class (length width*height), in
   *   selfie_multiclass order.
   * @returns false when the face gate could not be built (no usable landmarks).
   */
  assemble(confidences: readonly ArrayLike<number>[], landmarks: readonly NormalizedLandmark[]): boolean {
    const n = this.width * this.height
    if (confidences.length < SEGMENT_CLASS_COUNT) return false
    for (let c = 0; c < SEGMENT_CLASS_COUNT; c += 1) if (confidences[c].length < n) return false

    const gateResult = computeFaceGate(landmarks, this.width, this.height, this.gate, this.scratch)
    if (!gateResult.ok) return false

    const hair = confidences[SEGMENT_CLASS.hair]
    const body = confidences[SEGMENT_CLASS.bodySkin]
    const face = confidences[SEGMENT_CLASS.faceSkin]
    const clothes = confidences[SEGMENT_CLASS.clothes]
    const others = confidences[SEGMENT_CLASS.others]
    const alpha = this.initialised ? TEMPORAL_ALPHA : 1

    this.classMeans.fill(0)
    let interior = 0

    for (let i = 0; i < n; i += 1) {
      const packed = [hair[i], body[i], Math.max(clothes[i], others[i]), face[i]]
      for (let ch = 0; ch < 4; ch += 1) {
        const s = this.smooth[ch]
        s[i] += (packed[ch] - s[i]) * alpha
        this.classes[i * 4 + ch] = Math.max(0, Math.min(255, Math.round(s[i] * 255)))
      }
      if (this.gate[i] >= FACE_INTERIOR_GATE) {
        interior += 1
        for (let c = 0; c < SEGMENT_CLASS_COUNT; c += 1) this.classMeans[c] += confidences[c][i]
      }
    }

    if (interior > 0) for (let c = 0; c < SEGMENT_CLASS_COUNT; c += 1) this.classMeans[c] /= interior
    this.interiorPixels = interior
    this.initialised = true
    return true
  }

  interiorPixels = 0
}

export interface ForegroundSegmenterOptions {
  readonly wasmPath?: string
  readonly modelAssetPath?: string
  readonly delegate?: 'CPU' | 'GPU'
}

/**
 * Hair / hand / object segmentation for screen-space occlusion (Rules 2 & 3).
 *
 * Design constraints, all deliberate:
 *  - Lazy and non-blocking: face tracking never waits for this; until it is
 *    'ready' the renderer simply has no mask.
 *  - Bounded cost: it feeds a 256px-wide downscale (not the full video, whose
 *    mask would be returned at full resolution) and self-throttles to a duty
 *    cycle, so a slow CPU delegate degrades to a lower mask rate, not to a
 *    lower frame rate.
 *  - Fail closed: if the class layout is wrong (documented iOS/GPU scrambling)
 *    or inference keeps failing, it retries once on CPU and otherwise marks
 *    itself unavailable rather than occluding with the wrong class.
 *
 * NOTE: the inference path itself needs the model download and a browser, so it
 * is validated live, not in unit tests. Everything after inference is tested.
 */
export class ForegroundSegmenter {
  private readonly options: Required<ForegroundSegmenterOptions>
  private segmenter: ImageSegmenter | null = null
  private status: SegmenterStatus = 'idle'
  private statusMessage = ''
  private canvas: HTMLCanvasElement | OffscreenCanvas | null = null
  private ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null
  private assembler: MaskAssembler | null = null
  private lastRunMs = -Infinity
  private lastTimestamp = 0
  private minIntervalMs = MIN_INTERVAL_MS
  private avgDurationMs = 0
  private consecutiveErrors = 0
  private layoutFailures = 0
  private usedCpuFallback = false
  private version = 0
  private disposed = false
  private initPromise: Promise<void> | null = null

  constructor(options: ForegroundSegmenterOptions = {}) {
    this.options = {
      wasmPath: options.wasmPath ?? WASM_PATH,
      modelAssetPath: options.modelAssetPath ?? MODEL_PATH,
      delegate: options.delegate ?? 'GPU',
    }
  }

  getStatus(): SegmenterStatus {
    return this.status
  }

  getStatusMessage(): string {
    return this.statusMessage
  }

  getAverageDurationMs(): number {
    return this.avgDurationMs
  }

  /** Safe to call repeatedly; concurrent calls share one initialisation. */
  initialize(): Promise<void> {
    if (this.status === 'ready' || this.status === 'unavailable') return Promise.resolve()
    if (!this.initPromise) {
      this.initPromise = this.createSegmenter(this.options.delegate).finally(() => {
        this.initPromise = null
      })
    }
    return this.initPromise
  }

  /**
   * Call once per tracker frame. Returns a new mask frame only when inference
   * actually ran; null otherwise (throttled, not ready, or failed).
   */
  process(
    video: HTMLVideoElement,
    landmarks: readonly NormalizedLandmark[] | null,
    nowMs: number,
  ): ForegroundMaskFrame | null {
    if (this.disposed || this.status !== 'ready' || !this.segmenter || !landmarks) return null
    if (nowMs - this.lastRunMs < this.minIntervalMs) return null
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !(video.videoWidth > 0)) return null

    this.ensureInput(video)
    if (!this.ctx || !this.canvas || !this.assembler) return null

    const started = performance.now()
    let result: ImageSegmenterResult | null = null
    try {
      this.ctx.drawImage(video as CanvasImageSource, 0, 0, this.canvas.width, this.canvas.height)
      this.lastTimestamp = Math.max(this.lastTimestamp + 1, Math.floor(nowMs))
      result = this.segmenter.segmentForVideo(this.canvas as HTMLCanvasElement, this.lastTimestamp)

      const masks = result.confidenceMasks
      if (!masks || masks.length < SEGMENT_CLASS_COUNT) {
        throw new Error(`Segmenter returned ${masks?.length ?? 0} confidence masks, expected ${SEGMENT_CLASS_COUNT}.`)
      }

      const confidences = masks.map((mask) => mask.getAsFloat32Array())
      const built = this.assembler.assemble(confidences, landmarks)
      this.consecutiveErrors = 0

      const elapsed = performance.now() - started
      this.avgDurationMs = this.avgDurationMs === 0 ? elapsed : this.avgDurationMs * 0.8 + elapsed * 0.2
      this.minIntervalMs = Math.min(
        MAX_INTERVAL_MS,
        Math.max(MIN_INTERVAL_MS, this.avgDurationMs / MAX_DUTY_CYCLE),
      )
      this.lastRunMs = nowMs

      if (!built) return null
      if (!this.checkLayout()) return null

      this.version += 1
      return {
        width: this.assembler.width,
        height: this.assembler.height,
        classes: this.assembler.classes,
        gate: this.assembler.gate,
        anchorX: faceAnchor(landmarks, 0),
        anchorY: faceAnchor(landmarks, 1),
        version: this.version,
      }
    } catch (error) {
      this.consecutiveErrors += 1
      this.lastRunMs = nowMs
      if (this.consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        this.markUnavailable(error instanceof Error ? error.message : 'Segmentation failed repeatedly.')
      }
      return null
    } finally {
      result?.close()
    }
  }

  dispose(): void {
    this.disposed = true
    this.segmenter?.close()
    this.segmenter = null
    this.status = 'idle'
  }

  private checkLayout(): boolean {
    const assembler = this.assembler
    if (!assembler || assembler.interiorPixels < 50) return true // not enough evidence either way

    if (classLayoutLooksValid(assembler.classMeans)) {
      this.layoutFailures = 0
      return true
    }

    this.layoutFailures += 1
    if (this.layoutFailures >= LAYOUT_FAILURES_BEFORE_FALLBACK) {
      this.layoutFailures = 0
      if (!this.usedCpuFallback && this.options.delegate === 'GPU') {
        this.usedCpuFallback = true
        this.status = 'loading'
        this.statusMessage = 'Segmentation class layout invalid on GPU; retrying on CPU…'
        this.segmenter?.close()
        this.segmenter = null
        void this.createSegmenter('CPU')
      } else {
        this.markUnavailable('Segmentation class layout is invalid on this device (known GPU-delegate issue).')
      }
    }
    return false
  }

  private async createSegmenter(delegate: 'CPU' | 'GPU'): Promise<void> {
    this.status = 'loading'
    this.statusMessage = 'Loading hair/hand segmentation model…'
    try {
      const vision = await FilesetResolver.forVisionTasks(this.options.wasmPath)
      const make = (d: 'CPU' | 'GPU') =>
        ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: this.options.modelAssetPath, delegate: d },
          runningMode: 'VIDEO',
          outputConfidenceMasks: true,
          outputCategoryMask: false,
        })

      let segmenter: ImageSegmenter
      try {
        segmenter = await make(delegate)
      } catch (gpuError) {
        if (delegate !== 'GPU') throw gpuError
        this.usedCpuFallback = true
        segmenter = await make('CPU')
      }

      if (this.disposed) {
        segmenter.close()
        return
      }
      this.segmenter = segmenter
      this.assembler?.reset()
      this.status = 'ready'
      this.statusMessage = ''
    } catch (error) {
      this.markUnavailable(
        error instanceof Error ? `Segmentation unavailable: ${error.message}` : 'Segmentation unavailable.',
      )
    }
  }

  private markUnavailable(message: string): void {
    this.status = 'unavailable'
    this.statusMessage = message
    this.segmenter?.close()
    this.segmenter = null
  }

  private ensureInput(video: HTMLVideoElement): void {
    const height = Math.max(64, Math.min(256, Math.round((INPUT_WIDTH * video.videoHeight) / video.videoWidth)))
    if (this.canvas && this.canvas.width === INPUT_WIDTH && this.canvas.height === height) return

    if (typeof OffscreenCanvas !== 'undefined') {
      const canvas = new OffscreenCanvas(INPUT_WIDTH, height)
      this.canvas = canvas
      this.ctx = canvas.getContext('2d')
    } else {
      const canvas = document.createElement('canvas')
      canvas.width = INPUT_WIDTH
      canvas.height = height
      this.canvas = canvas
      this.ctx = canvas.getContext('2d')
    }
    this.assembler = new MaskAssembler(INPUT_WIDTH, height)
  }
}

/** Mean of a few rigid landmarks: nose bridge, nose tip, outer eye corners. */
function faceAnchor(landmarks: readonly NormalizedLandmark[], axis: 0 | 1): number {
  const indices = [168, 1, 33, 263]
  let sum = 0
  for (const i of indices) sum += axis === 0 ? landmarks[i].x : landmarks[i].y
  return sum / indices.length
}
