import { Euler, MathUtils, Quaternion } from 'three'
import type { FacePose } from './FacePose'
import type { FaceTrackerRuntimeResult } from './FaceTracker'
import type { FaceBounds } from './FaceTrackingState'

export interface FaceDebugOverlayOptions {
  readonly precision?: number
}

const DEFAULT_OPTIONS: Required<FaceDebugOverlayOptions> = {
  precision: 2,
}

/**
 * Temporary imperative diagnostics for Face Landmarker output.
 *
 * Per-frame values are written directly to the DOM rather than React state.
 * Pose values are consumed from the canonical FacePose produced by the single
 * coordinateTransform module; this diagnostic never interprets the raw
 * MediaPipe transformation matrix itself.
 */
export class FaceDebugOverlay {
  private readonly options: Required<FaceDebugOverlayOptions>
  private element: HTMLElement | null = null
  private readonly quaternion = new Quaternion()
  private readonly euler = new Euler(0, 0, 0, 'XYZ')

  constructor(options: FaceDebugOverlayOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options }
  }

  attach(element: HTMLElement): void {
    this.element = element
  }

  render(
    runtimeResult: FaceTrackerRuntimeResult | null,
    primaryFaceIndex: number | null,
    bounds: FaceBounds | null,
    facePose: FacePose | null,
  ): void {
    if (!this.element) return

    if (!runtimeResult || primaryFaceIndex === null || !bounds) {
      this.element.textContent = 'Face: searching\nPose: unavailable'
      return
    }

    const landmarks = runtimeResult.result.faceLandmarks[primaryFaceIndex]

    if (!landmarks || !facePose) {
      this.element.textContent = 'Face: detected\nPose: unavailable'
      return
    }

    this.quaternion.set(
      facePose.rotation.x,
      facePose.rotation.y,
      facePose.rotation.z,
      facePose.rotation.w,
    ).normalize()
    this.euler.setFromQuaternion(this.quaternion, 'XYZ')

    this.element.textContent = [
      'Face: detected',
      `Landmarks: ${landmarks.length}`,
      'Pose: canonical (mirrored)',
      `Position: x ${this.format(facePose.position.x)}  y ${this.format(facePose.position.y)}  z ${this.format(facePose.position.z)}`,
      `Rotation: X ${this.format(MathUtils.radToDeg(this.euler.x))}°  Y ${this.format(MathUtils.radToDeg(this.euler.y))}°  Z ${this.format(MathUtils.radToDeg(this.euler.z))}°`,
      `Scale: ${this.format(facePose.scale)}`,
      `Bounds: ${this.format(bounds.width)} × ${this.format(bounds.height)}`,
      `Frame: ${Math.round(facePose.timestampMs)} ms`,
    ].join('\n')
  }

  clear(): void {
    this.element?.replaceChildren(document.createTextNode('Face: searching\nPose: unavailable'))
  }

  dispose(): void {
    this.clear()
    this.element = null
  }

  private format(value: number | undefined): string {
    return typeof value === 'number' && Number.isFinite(value)
      ? value.toFixed(this.options.precision)
      : '—'
  }
}
