import * as THREE from 'three'
import type { FacePose } from './FacePose'

export interface PoseSmoothingOptions {
  /** Time in seconds for position/scale to move halfway toward the target. */
  readonly positionHalfLifeSeconds?: number
  /** Time in seconds for rotation to move halfway toward the target. */
  readonly rotationHalfLifeSeconds?: number
}

export interface PoseSmoothingSettings {
  readonly positionHalfLifeSeconds: number
  readonly rotationHalfLifeSeconds: number
}

export const DEFAULT_POSITION_HALF_LIFE_SECONDS = 0
export const DEFAULT_ROTATION_HALF_LIFE_SECONDS = 0
const MAX_DELTA_SECONDS = 0.25
const MIN_ADAPTIVE_HALF_LIFE_SECONDS = 0.012

/**
 * Deterministic, frame-rate-independent smoother for canonical FacePose data.
 *
 * A half-life of 0 (the default) means NO smoothing: the pose snaps fully to
 * the latest target every frame, so the glasses track the face with zero
 * added latency ("stuck to the face"). Raising a half-life trades some of
 * that immediacy for reduced jitter from landmark noise.
 *
 * Position and scale use exponential convergence. Rotation uses quaternion
 * slerp. Both support a velocity-adaptive half-life: during head rotation or
 * movement, half-life is dynamically reduced (never increased) to eliminate
 * tracking lag while allowing extra smoothing when nearly stationary.
 */
export class PoseSmoother {
  private positionHalfLifeSeconds: number
  private rotationHalfLifeSeconds: number
  private readonly position = new THREE.Vector3()
  private readonly rotation = new THREE.Quaternion()
  private scale = 1
  private initialized = false

  constructor(options: PoseSmoothingOptions = {}) {
    this.positionHalfLifeSeconds = validateHalfLife(
      options.positionHalfLifeSeconds ?? DEFAULT_POSITION_HALF_LIFE_SECONDS,
      'positionHalfLifeSeconds',
    )
    this.rotationHalfLifeSeconds = validateHalfLife(
      options.rotationHalfLifeSeconds ?? DEFAULT_ROTATION_HALF_LIFE_SECONDS,
      'rotationHalfLifeSeconds',
    )
  }

  getSettings(): PoseSmoothingSettings {
    return {
      positionHalfLifeSeconds: this.positionHalfLifeSeconds,
      rotationHalfLifeSeconds: this.rotationHalfLifeSeconds,
    }
  }

  setSettings(settings: PoseSmoothingSettings): void {
    this.positionHalfLifeSeconds = validateHalfLife(
      settings.positionHalfLifeSeconds,
      'positionHalfLifeSeconds',
    )
    this.rotationHalfLifeSeconds = validateHalfLife(
      settings.rotationHalfLifeSeconds,
      'rotationHalfLifeSeconds',
    )
  }

  reset(): void {
    this.initialized = false
    this.position.set(0, 0, 0)
    this.rotation.identity()
    this.scale = 1
  }

  isInitialized(): boolean {
    return this.initialized
  }

  update(target: FacePose, deltaSeconds: number): FacePose {
    if (!Number.isFinite(deltaSeconds) || deltaSeconds < 0) {
      throw new Error('PoseSmoother requires a finite, non-negative delta time.')
    }

    validatePose(target)

    const clampedDeltaSeconds = Math.min(deltaSeconds, MAX_DELTA_SECONDS)

    const targetPosition = new THREE.Vector3(
      target.position.x,
      target.position.y,
      target.position.z,
    )
    const targetRotation = new THREE.Quaternion(
      target.rotation.x,
      target.rotation.y,
      target.rotation.z,
      target.rotation.w,
    ).normalize()

    if (!this.initialized || clampedDeltaSeconds <= 0) {
      this.position.copy(targetPosition)
      this.rotation.copy(targetRotation)
      this.scale = target.scale
      this.initialized = true
    } else {
      // Compute rotational velocity (angular distance in radians per second)
      const angularDistance = this.rotation.angleTo(targetRotation)
      const angularVelocity = angularDistance / clampedDeltaSeconds

      // Compute linear velocity (cm per second)
      const linearDistance = this.position.distanceTo(targetPosition)
      const linearVelocity = linearDistance / clampedDeltaSeconds

      // Adaptively scale down half-life during rapid motion for instant response.
      // Wrapped in min() so this can only ever REDUCE latency relative to the
      // configured half-life, never increase it -- important when the
      // configured half-life is already at or below MIN_ADAPTIVE_HALF_LIFE_SECONDS
      // (e.g. 0, "instant"): without the min(), lerping toward the adaptive
      // floor from below would paradoxically ADD lag during fast motion.
      const rotFactor = Math.max(0, Math.min(1, (angularVelocity - 0.2) / 1.5))
      const posFactor = Math.max(0, Math.min(1, (linearVelocity - 2.0) / 10.0))

      const effectiveRotHalfLife = Math.min(
        this.rotationHalfLifeSeconds,
        THREE.MathUtils.lerp(this.rotationHalfLifeSeconds, MIN_ADAPTIVE_HALF_LIFE_SECONDS, rotFactor),
      )
      const effectivePosHalfLife = Math.min(
        this.positionHalfLifeSeconds,
        THREE.MathUtils.lerp(this.positionHalfLifeSeconds, MIN_ADAPTIVE_HALF_LIFE_SECONDS, posFactor),
      )

      const positionAlpha = halfLifeAlpha(clampedDeltaSeconds, effectivePosHalfLife)
      const rotationAlpha = halfLifeAlpha(clampedDeltaSeconds, effectiveRotHalfLife)

      this.position.lerp(targetPosition, positionAlpha)
      this.rotation.slerp(targetRotation, rotationAlpha).normalize()
      this.scale = THREE.MathUtils.lerp(this.scale, target.scale, positionAlpha)
    }

    return {
      position: {
        x: this.position.x,
        y: this.position.y,
        z: this.position.z,
      },
      rotation: {
        x: this.rotation.x,
        y: this.rotation.y,
        z: this.rotation.z,
        w: this.rotation.w,
      },
      scale: this.scale,
      trackingState: target.trackingState,
      timestampMs: target.timestampMs,
    }
  }
}

export function halfLifeAlpha(deltaSeconds: number, halfLifeSeconds: number): number {
  if (!Number.isFinite(deltaSeconds) || deltaSeconds < 0) {
    throw new Error('halfLifeAlpha requires a finite, non-negative delta time.')
  }

  const halfLife = validateHalfLife(halfLifeSeconds, 'halfLifeSeconds')
  if (halfLife === 0) {
    // Instant: fully converge in one step. Also sidesteps 0/0 = NaN when
    // deltaSeconds is also 0 (though PoseSmoother.update already short-circuits
    // that case before reaching here).
    return deltaSeconds > 0 ? 1 : 0
  }
  return 1 - Math.pow(0.5, deltaSeconds / halfLife)
}

function validateHalfLife(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a finite value greater than or equal to zero.`)
  }

  return value
}

function validatePose(pose: FacePose): void {
  const values = [
    pose.position.x,
    pose.position.y,
    pose.position.z,
    pose.rotation.x,
    pose.rotation.y,
    pose.rotation.z,
    pose.rotation.w,
    pose.scale,
    pose.timestampMs,
  ]

  if (!values.every(Number.isFinite) || pose.scale <= 0) {
    throw new Error('PoseSmoother received a FacePose with invalid numeric values.')
  }
}
