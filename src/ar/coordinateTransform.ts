import * as THREE from 'three'
import type {
  FacePose,
  FacePosePosition,
  FacePoseQuaternion,
  MediaPipeFaceTransformationMatrix,
} from './FacePose'
import type { TrackingState } from './FaceTrackingState'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

/**
 * COORDINATE CONTRACT
 *
 * Source: MediaPipe Face Geometry metric 3D space.
 * - Right-handed orthonormal space.
 * - +X points to the subject's/right side of the source camera image.
 * - +Y points up.
 * - -Z points away from the camera, so +Z points toward the camera.
 * - The facial transformation matrix maps the canonical face model into this
 *   runtime metric space and contains uniform scale, rotation and translation.
 *
 * Canonical app space:
 * - Right-handed metric face space after the display-mirror basis correction.
 * - Units are retained as MediaPipe metric units (centimeter-scale), so no
 *   arbitrary scene-scale factor is introduced into pose conversion.
 * - +X points to the user's screen-right in the mirrored video.
 * - +Y points up on screen.
 * - -Z points away from the viewer.
 * - +Z therefore points toward the viewer.
 *
 * Three.js space:
 * - Right-handed, with +X right, +Y up and -Z forward/into the scene.
 * - One Three.js world unit is treated as one canonical metric unit
 *   (centimeter-scale) for the face-following POC.
 * - The canonical -> Three.js mapping is therefore an explicit
 *   representation conversion only; perspective projection is owned by
 *   ARRenderer's camera, not by this module.
 *
 * MIRROR CORRECTION
 * MediaPipe processes the underlying, unmirrored camera frame while the
 * <video> is displayed with CSS scaleX(-1). The display mirror is therefore
 * represented as a change of basis S = diag(-1, 1, 1).
 *
 * For a MediaPipe rigid transform T = [sR t; 0 1], the mirrored transform is:
 *
 *   T_mirrored = S * T * S
 *
 * This is deliberately a conjugation, not only an X translation flip:
 * - position X changes sign;
 * - yaw/roll directions change sign through the reflected basis;
 * - pitch direction is preserved;
 * - scale is preserved;
 * - the resulting rotation remains a proper rotation (determinant +1).
 *
 * No calibration or smoothing belongs in this module.
 */

const MIRROR_BASIS = new THREE.Matrix4().set(
  -1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
)

const MIN_SCALE = 1e-8
const UNIFORM_SCALE_RELATIVE_TOLERANCE = 1e-3
const CANONICAL_FACE_WIDTH = 14
const MIN_FACE_WIDTH = 1e-4

export interface ThreeJsFaceTransform {
  readonly position: THREE.Vector3
  readonly quaternion: THREE.Quaternion
  readonly scale: number
}

/**
 * Applies the display-mirror basis change to a MediaPipe transformation.
 *
 * This is the named, testable mirroring step required by the AR architecture.
 * The returned matrix remains a proper rigid transform with uniform scale.
 */
export function applyDisplayMirrorCorrection(
  mediaPipeMatrix: MediaPipeFaceTransformationMatrix,
): THREE.Matrix4 {
  const source = new THREE.Matrix4().fromArray(mediaPipeMatrix.data)

  // S * T * S reflects the coordinate basis on both sides of the transform.
  return MIRROR_BASIS.clone().multiply(source).multiply(MIRROR_BASIS)
}

/**
 * Converts a MediaPipe facial transformation matrix into the canonical
 * mirrored-display FacePose.
 *
 * `trackingState` and `timestampMs` are metadata supplied by the runtime; this
 * function performs no inference, smoothing, or calibration.
 */
export function mediaPipeTransformationMatrixToFacePose(
  mediaPipeMatrix: MediaPipeFaceTransformationMatrix,
  trackingState: TrackingState,
  timestampMs: number,
): FacePose {
  if (!Number.isFinite(timestampMs)) {
    throw new Error('coordinateTransform requires a finite pose timestamp.')
  }

  if (!Number.isFinite(mediaPipeMatrix.rows) || mediaPipeMatrix.rows !== 4) {
    throw new Error('coordinateTransform requires a 4x4 MediaPipe matrix.')
  }

  if (!Number.isFinite(mediaPipeMatrix.columns) || mediaPipeMatrix.columns !== 4) {
    throw new Error('coordinateTransform requires a 4x4 MediaPipe matrix.')
  }

  if (mediaPipeMatrix.data.length !== 16 || !mediaPipeMatrix.data.every(Number.isFinite)) {
    throw new Error('coordinateTransform requires 16 finite MediaPipe matrix values.')
  }

  const mirroredMatrix = applyDisplayMirrorCorrection(mediaPipeMatrix)

  const position = new THREE.Vector3()
  const quaternion = new THREE.Quaternion()
  const scaleVector = new THREE.Vector3()
  mirroredMatrix.decompose(position, quaternion, scaleVector)

  if (
    scaleVector.x < MIN_SCALE ||
    scaleVector.y < MIN_SCALE ||
    scaleVector.z < MIN_SCALE ||
    !Number.isFinite(scaleVector.x) ||
    !Number.isFinite(scaleVector.y) ||
    !Number.isFinite(scaleVector.z)
  ) {
    throw new Error('coordinateTransform received a matrix with an invalid scale.')
  }

  // Face-landmarker matrices contain small floating-point differences between
  // the three scale components even when the underlying transform is uniform.
  // Validate that difference relatively, then collapse it to one deterministic
  // uniform scale for the canonical pose.
  const scale = (scaleVector.x + scaleVector.y + scaleVector.z) / 3
  const scaleTolerance = Math.max(scale, MIN_SCALE) * UNIFORM_SCALE_RELATIVE_TOLERANCE

  if (
    Math.abs(scaleVector.x - scale) > scaleTolerance ||
    Math.abs(scaleVector.y - scale) > scaleTolerance ||
    Math.abs(scaleVector.z - scale) > scaleTolerance
  ) {
    throw new Error('coordinateTransform requires the MediaPipe transform to have uniform scale.')
  }

  const posePosition: FacePosePosition = {
    x: position.x,
    y: position.y,
    z: position.z,
  }

  const poseRotation: FacePoseQuaternion = {
    x: quaternion.x,
    y: quaternion.y,
    z: quaternion.z,
    w: quaternion.w,
  }

  return {
    position: posePosition,
    rotation: poseRotation,
    scale,
    trackingState,
    timestampMs,
  }
}

/**
 * Converts MediaPipe normalized face landmarks into canonical mirrored-display
 * face-local coordinates for depth-only rendering. The mirror correction lives
 * here with the transformation-matrix correction so no renderer or occluder
 * performs an implicit axis/sign conversion.
 *
 * MediaPipe landmark Z is relative to the face and uses smaller values for
 * points closer to the camera. Canonical +Z points toward the viewer, so the
 * sign is inverted during this conversion.
 */
export function writeMediaPipeLandmarksToCanonicalFaceMesh(
  landmarks: readonly NormalizedLandmark[],
  output: Float32Array,
): boolean {
  if (landmarks.length === 0 || output.length !== landmarks.length * 3) {
    return false
  }

  let minX = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY

  for (const landmark of landmarks) {
    if (!Number.isFinite(landmark.x) || !Number.isFinite(landmark.y) || !Number.isFinite(landmark.z)) {
      return false
    }

    minX = Math.min(minX, landmark.x)
    maxX = Math.max(maxX, landmark.x)
    minY = Math.min(minY, landmark.y)
    maxY = Math.max(maxY, landmark.y)
  }

  const width = maxX - minX
  if (width <= MIN_FACE_WIDTH) {
    return false
  }

  const centerX = (minX + maxX) / 2
  const centerY = (minY + maxY) / 2
  const scale = CANONICAL_FACE_WIDTH / width

  for (let index = 0; index < landmarks.length; index += 1) {
    const landmark = landmarks[index]
    const offset = index * 3

    // The X basis is reflected because MediaPipe sees the unmirrored camera
    // frame while the application displays the video mirrored.
    output[offset] = (centerX - landmark.x) * scale
    output[offset + 1] = (centerY - landmark.y) * scale
    output[offset + 2] = -landmark.z * scale
  }

  return true
}

/**
 * Converts the canonical FacePose into the exact Three.js runtime types.
 *
 * The canonical basis was deliberately defined to match Three.js, so this
 * function performs a representation conversion only; it does not introduce
 * another axis/sign mapping.
 */
export function facePoseToThreeJsTransform(
  pose: FacePose,
): ThreeJsFaceTransform {
  if (!Number.isFinite(pose.position.x) ||
      !Number.isFinite(pose.position.y) ||
      !Number.isFinite(pose.position.z) ||
      !Number.isFinite(pose.rotation.x) ||
      !Number.isFinite(pose.rotation.y) ||
      !Number.isFinite(pose.rotation.z) ||
      !Number.isFinite(pose.rotation.w) ||
      !Number.isFinite(pose.scale) ||
      pose.scale < MIN_SCALE) {
    throw new Error('coordinateTransform received a FacePose with non-finite values.')
  }

  return {
    position: new THREE.Vector3(
      pose.position.x,
      pose.position.y,
      pose.position.z,
    ),
    quaternion: new THREE.Quaternion(
      pose.rotation.x,
      pose.rotation.y,
      pose.rotation.z,
      pose.rotation.w,
    ).normalize(),
    scale: pose.scale,
  }
}
