export type CameraErrorCode =
  | 'permission-denied'
  | 'camera-not-found'
  | 'camera-not-readable'
  | 'constraints-unsupported'
  | 'unsupported'
  | 'unknown'

export class CameraError extends Error {
  readonly code: CameraErrorCode
  readonly cause: unknown

  constructor(code: CameraErrorCode, message: string, cause?: unknown) {
    super(message)
    this.name = 'CameraError'
    this.code = code
    this.cause = cause
  }
}

export interface CameraManagerOptions {
  readonly idealWidth?: number
  readonly idealHeight?: number
  readonly idealFrameRate?: number
}

const DEFAULT_OPTIONS: Required<CameraManagerOptions> = {
  idealWidth: 1280,
  idealHeight: 720,
  idealFrameRate: 30,
}

/**
 * Owns only browser camera acquisition and stream lifecycle.
 *
 * The manager deliberately does not start itself. The caller must invoke
 * start(), normally from an explicit user interaction.
 */
export class CameraManager {
  private readonly options: Required<CameraManagerOptions>
  private stream: MediaStream | null = null
  private videoElement: HTMLVideoElement | null = null
  private lastError: CameraError | null = null

  constructor(options: CameraManagerOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options }
  }

  /**
   * Configures the eventual video consumer for iOS Safari compatibility.
   * Attaching a video element does not start the camera or play the element.
   */
  attachVideoElement(videoElement: HTMLVideoElement): void {
    this.videoElement = videoElement
    videoElement.playsInline = true
    videoElement.muted = true
    videoElement.autoplay = true
  }

  /** Requests a user-facing camera stream. Safe to call repeatedly. */
  async start(): Promise<MediaStream> {
    if (this.stream) {
      return this.stream
    }

    if (!this.isSupported()) {
      const error = new CameraError(
        'unsupported',
        'Camera access is not supported by this browser.',
      )
      this.lastError = error
      throw error
    }

    this.lastError = null

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'user' },
          width: { ideal: this.options.idealWidth },
          height: { ideal: this.options.idealHeight },
          frameRate: { ideal: this.options.idealFrameRate },
        },
        audio: false,
      })
    } catch (error) {
      if (!this.isOverconstrainedError(error)) {
        const classified = this.classifyError(error)
        this.lastError = classified
        throw classified
      }

      // A browser may reject one or more requested constraints even though a
      // camera is available. Retry with only the front-camera preference.
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'user' } },
          audio: false,
        })
      } catch (fallbackError) {
        if (!this.isOverconstrainedError(fallbackError)) {
          const classified = this.classifyError(fallbackError)
          this.lastError = classified
          throw classified
        }

        // Final graceful fallback: let the browser choose any supported video
        // configuration. Audio remains explicitly disabled.
        try {
          this.stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          })
        } catch (finalError) {
          const classified = this.classifyError(finalError)
          this.lastError = classified
          throw classified
        }
      }
    }

    this.bindStreamToVideo()
    return this.stream
  }

  /** Returns the currently owned stream, if the camera is active. */
  getStream(): MediaStream | null {
    return this.stream
  }

  /** Returns the most recent classified camera error, if any. */
  getLastError(): CameraError | null {
    return this.lastError
  }

  /** Stops every track owned by this manager and releases the stream. */
  stop(): void {
    const stream = this.stream

    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop()
      }
    }

    this.stream = null

    if (this.videoElement) {
      this.videoElement.pause()
      this.videoElement.srcObject = null
    }
  }

  /** Releases all camera/video resources owned by the manager. */
  dispose(): void {
    this.stop()
    this.videoElement = null
    this.lastError = null
  }

  private isSupported(): boolean {
    return (
      typeof navigator !== 'undefined' &&
      Boolean(navigator.mediaDevices) &&
      typeof navigator.mediaDevices.getUserMedia === 'function'
    )
  }

  private bindStreamToVideo(): void {
    if (!this.videoElement || !this.stream) {
      return
    }

    this.videoElement.playsInline = true
    this.videoElement.muted = true
    this.videoElement.autoplay = true
    this.videoElement.srcObject = this.stream
  }

  private classifyError(error: unknown): CameraError {
    if (this.isDomException(error, 'NotAllowedError')) {
      return new CameraError(
        'permission-denied',
        'Camera permission was denied or blocked.',
        error,
      )
    }

    if (this.isDomException(error, 'NotFoundError')) {
      return new CameraError(
        'camera-not-found',
        'No camera was found on this device.',
        error,
      )
    }

    if (this.isDomException(error, 'NotReadableError')) {
      return new CameraError(
        'camera-not-readable',
        'The camera could not be accessed, possibly because it is already in use.',
        error,
      )
    }

    if (this.isDomException(error, 'OverconstrainedError')) {
      return new CameraError(
        'constraints-unsupported',
        'The requested camera constraints are not supported by this device.',
        error,
      )
    }

    return new CameraError(
      'unknown',
      'The camera could not be started.',
      error,
    )
  }

  private isOverconstrainedError(error: unknown): boolean {
    return this.isDomException(error, 'OverconstrainedError')
  }

  private isDomException(error: unknown, name: string): boolean {
    return (
      (typeof DOMException !== 'undefined' && error instanceof DOMException && error.name === name) ||
      (typeof error === 'object' && error !== null && 'name' in error && error.name === name)
    )
  }
}
