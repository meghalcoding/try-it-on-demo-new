import type { FaceTracker, FaceTrackerRuntimeResult } from './FaceTracker'

export interface VideoFrameSchedulerOptions {
  readonly onFrame: (runtimeResult: FaceTrackerRuntimeResult) => void
  readonly onError?: (error: unknown) => void
}

/**
 * Drives FaceTracker from the video's actual frame cadence without owning any
 * React state or Three.js rendering. The scheduler can use requestVideoFrameCallback
 * when available and falls back to requestAnimationFrame otherwise.
 */
export class VideoFrameScheduler {
  private readonly video: HTMLVideoElement
  private readonly tracker: FaceTracker
  private readonly options: VideoFrameSchedulerOptions
  private running = false
  private videoFrameCallbackId: number | null = null
  private animationFrameId: number | null = null
  private lastProcessedTimestampMs = -1

  constructor(
    video: HTMLVideoElement,
    tracker: FaceTracker,
    options: VideoFrameSchedulerOptions,
  ) {
    this.video = video
    this.tracker = tracker
    this.options = options
  }

  start(): void {
    if (this.running) {
      return
    }

    this.running = true
    this.lastProcessedTimestampMs = -1

    if (typeof this.video.requestVideoFrameCallback === 'function') {
      this.scheduleVideoFrameCallback()
      return
    }

    this.scheduleAnimationFrame()
  }

  stop(): void {
    this.running = false

    if (this.videoFrameCallbackId !== null && typeof this.video.cancelVideoFrameCallback === 'function') {
      this.video.cancelVideoFrameCallback(this.videoFrameCallbackId)
      this.videoFrameCallbackId = null
    }

    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
  }

  isRunning(): boolean {
    return this.running
  }

  dispose(): void {
    this.stop()
  }

  private scheduleVideoFrameCallback(): void {
    if (!this.running) {
      return
    }

    this.videoFrameCallbackId = this.video.requestVideoFrameCallback((now, metadata) => {
      this.videoFrameCallbackId = null

      if (!this.running) {
        return
      }

      // Use expectedDisplayTime or high-res presentation timestamp for live camera streams
      const frameTimestamp = Number.isFinite(metadata.expectedDisplayTime) && metadata.expectedDisplayTime > 0
        ? metadata.expectedDisplayTime
        : (Number.isFinite(now) && now > 0 ? now : performance.now())

      this.processFrame(frameTimestamp)
      this.scheduleVideoFrameCallback()
    })
  }

  private scheduleAnimationFrame(): void {
    if (!this.running) {
      return
    }

    this.animationFrameId = requestAnimationFrame((timestampMs) => {
      this.animationFrameId = null

      if (!this.running) {
        return
      }

      if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !this.video.paused && !this.video.ended) {
        this.processFrame(timestampMs)
      }

      this.scheduleAnimationFrame()
    })
  }

  private processFrame(timestampMs: number): void {
    if (!Number.isFinite(timestampMs)) {
      this.reportError(new Error('Video frame scheduler received an invalid frame timestamp.'))
      return
    }

    if (timestampMs <= this.lastProcessedTimestampMs) {
      return
    }

    try {
      const runtimeResult = this.tracker.processVideoFrame(this.video, timestampMs)
      this.lastProcessedTimestampMs = timestampMs
      this.options.onFrame(runtimeResult)
    } catch (error) {
      this.reportError(error)
    }
  }

  private reportError(error: unknown): void {
    this.options.onError?.(error)
  }
}
