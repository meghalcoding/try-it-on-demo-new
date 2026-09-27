import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from '@mediapipe/tasks-vision'

export interface FaceTrackerOptions {
  /** Path or URL to the MediaPipe Vision Tasks WASM files. */
  readonly wasmPath?: string
  /** Path or URL to the Face Landmarker task model. */
  readonly modelAssetPath?: string
  /** MediaPipe delegate. CPU is the conservative browser default. */
  readonly delegate?: 'CPU' | 'GPU'
}

export interface FaceTrackerRuntimeResult {
  readonly result: FaceLandmarkerResult
  readonly timestampMs: number
  readonly hasFace: boolean
  readonly hasTransformationMatrix: boolean
}

const DEFAULT_OPTIONS: Required<FaceTrackerOptions> = {
  wasmPath:
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm',
  modelAssetPath:
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  delegate: 'GPU',
}

/**
 * Owns MediaPipe Face Landmarker initialization and synchronous VIDEO-mode
 * inference. It deliberately has no React dependency and stores its latest
 * result in mutable runtime state.
 *
 * Frame scheduling is intentionally not owned here. The later frame-scheduler
 * task will decide when processVideoFrame() is called.
 */
export class FaceTracker {
  private readonly options: Required<FaceTrackerOptions>
  private landmarker: FaceLandmarker | null = null
  private latestResult: FaceTrackerRuntimeResult | null = null
  private lastTimestampMs = -1

  constructor(options: FaceTrackerOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options }
  }

  async initialize(): Promise<void> {
    if (this.landmarker) {
      return
    }

    const vision = await FilesetResolver.forVisionTasks(this.options.wasmPath)

    try {
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: this.options.modelAssetPath,
          delegate: this.options.delegate,
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: true,
      })
    } catch (gpuError) {
      if (this.options.delegate === 'GPU') {
        console.warn('FaceTracker: GPU delegate initialization failed, falling back to CPU.', gpuError)
        this.landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: this.options.modelAssetPath,
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: true,
        })
      } else {
        throw gpuError
      }
    }
  }

  /**
   * Runs one VIDEO-mode inference against the supplied HTMLVideoElement.
   * The caller supplies a monotonically increasing timestamp in milliseconds.
   */
  processVideoFrame(
    video: HTMLVideoElement,
    timestampMs: number,
  ): FaceTrackerRuntimeResult {
    if (!this.landmarker) {
      throw new Error(
        'FaceTracker must be initialized before processing video frames.',
      )
    }

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      throw new Error(
        'FaceTracker cannot process a video frame before video data is available.',
      )
    }

    if (!Number.isFinite(timestampMs)) {
      throw new Error(
        'FaceTracker requires a finite frame timestamp in milliseconds.',
      )
    }

    if (timestampMs <= this.lastTimestampMs) {
      throw new Error(
        'FaceTracker frame timestamps must increase monotonically.',
      )
    }

    const result = this.landmarker.detectForVideo(video, timestampMs)
    const runtimeResult: FaceTrackerRuntimeResult = {
      result,
      timestampMs,
      hasFace: result.faceLandmarks.length > 0,
      hasTransformationMatrix:
        result.facialTransformationMatrixes.length > 0 &&
        result.facialTransformationMatrixes[0].data.length === 16,
    }

    this.lastTimestampMs = timestampMs
    this.latestResult = runtimeResult

    return runtimeResult
  }

  getLatestResult(): FaceTrackerRuntimeResult | null {
    return this.latestResult
  }

  getLandmarker(): FaceLandmarker | null {
    return this.landmarker
  }

  dispose(): void {
    this.landmarker?.close()
    this.landmarker = null
    this.latestResult = null
    this.lastTimestampMs = -1
  }
}
