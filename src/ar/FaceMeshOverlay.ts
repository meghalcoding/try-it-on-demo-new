import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import type { FaceTrackerRuntimeResult } from './FaceTracker'
import type { FaceBounds } from './FaceTrackingState'

export interface FaceMeshOverlayOptions {
  readonly lineWidth?: number
  readonly pointRadius?: number
}

const DEFAULT_OPTIONS: Required<FaceMeshOverlayOptions> = {
  lineWidth: 0.9,
  pointRadius: 1.1,
}

/**
 * Temporary 2D proof overlay for face tracking. It deliberately does not use
 * Three.js and consumes MediaPipe landmarks only for visualization.
 */
export class FaceMeshOverlay {
  private readonly options: Required<FaceMeshOverlayOptions>
  private canvas: HTMLCanvasElement | null = null
  private context: CanvasRenderingContext2D | null = null
  private resizeObserver: ResizeObserver | null = null

  constructor(options: FaceMeshOverlayOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options }
  }

  attach(canvas: HTMLCanvasElement): void {
    this.resizeObserver?.disconnect()
    this.resizeObserver = null

    this.canvas = canvas
    this.context = canvas.getContext('2d')

    const stage = canvas.parentElement
    if (stage && typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(([entry]) => {
        const { width, height } = entry.contentRect
        this.resize(width, height)
      })
      this.resizeObserver.observe(stage)
      this.resize(stage.clientWidth, stage.clientHeight)
    }
  }

  resize(width: number, height: number): void {
    if (!this.canvas || !this.context || width <= 0 || height <= 0) {
      return
    }

    const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1
    this.canvas.width = Math.round(width * dpr)
    this.canvas.height = Math.round(height * dpr)
    this.canvas.style.width = `${width}px`
    this.canvas.style.height = `${height}px`
    this.context.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  render(
    runtimeResult: FaceTrackerRuntimeResult | null,
    primaryFaceIndex: number | null,
    video: HTMLVideoElement,
    faceBounds: FaceBounds | null,
  ): void {
    if (!this.canvas || !this.context) {
      return
    }

    const width = this.canvas.clientWidth
    const height = this.canvas.clientHeight
    this.context.clearRect(0, 0, width, height)

    if (!runtimeResult || primaryFaceIndex === null || !faceBounds) {
      return
    }

    const landmarks = runtimeResult.result.faceLandmarks[primaryFaceIndex]
    if (!landmarks) {
      return
    }

    const points = landmarks.map((landmark) => this.mapToCoverSurface(landmark, video, width, height))

    this.context.lineWidth = this.options.lineWidth
    this.context.strokeStyle = 'rgba(91, 255, 196, 0.85)'
    this.context.beginPath()

    for (const connection of FaceLandmarker.FACE_LANDMARKS_TESSELATION) {
      const start = points[connection.start]
      const end = points[connection.end]
      if (!start || !end) continue
      this.context.moveTo(start.x, start.y)
      this.context.lineTo(end.x, end.y)
    }

    this.context.stroke()

    this.context.fillStyle = 'rgba(255, 255, 255, 0.9)'
    for (const point of points) {
      this.context.beginPath()
      this.context.arc(point.x, point.y, this.options.pointRadius, 0, Math.PI * 2)
      this.context.fill()
    }
  }

  clear(): void {
    if (!this.canvas || !this.context) {
      return
    }

    this.context.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight)
  }

  dispose(): void {
    this.clear()
    this.resizeObserver?.disconnect()
    this.resizeObserver = null
    this.canvas = null
    this.context = null
  }

  private mapToCoverSurface(
    landmark: NormalizedLandmark,
    video: HTMLVideoElement,
    surfaceWidth: number,
    surfaceHeight: number,
  ): { x: number; y: number } {
    const videoWidth = video.videoWidth || surfaceWidth
    const videoHeight = video.videoHeight || surfaceHeight
    const scale = Math.max(surfaceWidth / videoWidth, surfaceHeight / videoHeight)
    const renderedWidth = videoWidth * scale
    const renderedHeight = videoHeight * scale
    const offsetX = (surfaceWidth - renderedWidth) / 2
    const offsetY = (surfaceHeight - renderedHeight) / 2

    return {
      x: offsetX + landmark.x * renderedWidth,
      y: offsetY + landmark.y * renderedHeight,
    }
  }
}
