import type { Matrix } from '@mediapipe/tasks-vision'
import type { TrackingState } from './FaceTrackingState'

/**
 * Canonical application-space position. Coordinate conversion is intentionally
 * not performed here; Task 5.2 owns MediaPipe -> canonical/Three.js mapping.
 */
export interface FacePosePosition {
  readonly x: number
  readonly y: number
  readonly z: number
}

/** Canonical quaternion rotation used by the runtime pose pipeline. */
export interface FacePoseQuaternion {
  readonly x: number
  readonly y: number
  readonly z: number
  readonly w: number
}

/**
 * Canonical runtime pose consumed by later smoothing/rendering stages.
 *
 * `trackingState` is application tracking state, while `timestampMs` is the
 * frame timestamp carried by the FaceTracker runtime. MediaPipe's
 * FaceLandmarkerResult does not expose a per-face confidence value, so no
 * confidence field is invented here.
 */
export interface FacePose {
  readonly position: FacePosePosition
  readonly rotation: FacePoseQuaternion
  readonly scale: number
  readonly trackingState: TrackingState
  readonly timestampMs: number
}

/**
 * Exact MediaPipe matrix shape at the adapter boundary used by this repo.
 * FaceLandmarkerResult.facialTransformationMatrixes contains Matrix values;
 * the runtime FaceTracker verifies that the selected matrix contains 16 data
 * values before treating it as a 4x4 facial transformation matrix.
 */
export type MediaPipeFaceTransformationMatrix = Pick<
  Matrix,
  'rows' | 'columns' | 'data'
>

export function isMediaPipeFaceTransformationMatrix(
  matrix: Matrix | undefined,
): matrix is MediaPipeFaceTransformationMatrix {
  return Boolean(
    matrix &&
      matrix.rows === 4 &&
      matrix.columns === 4 &&
      matrix.data.length === 16 &&
      matrix.data.every(Number.isFinite),
  )
}
