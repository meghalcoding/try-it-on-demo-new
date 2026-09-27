import * as THREE from 'three'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import type { FaceReference } from './types'

// MediaPipe 468/478 Landmark Indices
const INDEX_NOSE_BRIDGE = 168
const INDEX_FOREHEAD = 10
const INDEX_NOSE_TIP = 1
const INDEX_LEFT_EYE_OUTER = 33
const INDEX_LEFT_EYE_INNER = 133
const INDEX_RIGHT_EYE_INNER = 362
const INDEX_RIGHT_EYE_OUTER = 263
const INDEX_LEFT_PUPIL = 468
const INDEX_RIGHT_PUPIL = 473

const CANONICAL_FACE_WIDTH_CM = 14.0
const AVERAGE_HUMAN_IPD_CM = 6.3

// MediaPipe's facial transformation matrix maps the canonical 468-point face
// model into runtime face space. Its transform origin is not the eye line: the
// canonical outer/inner eye-corner midpoint sits at Y=2.624618 cm in that
// canonical model. The fallback reference must use the same eye-line origin as
// the landmark-derived reference, otherwise automatic GLB calibration anchors
// lens-midpoint eyewear below the eyes by the canonical origin-to-eye offset.
const MEDIA_PIPE_CANONICAL_EYE_LINE_Y_CM = 2.624618

/**
 * Extracts reference face geometry (eye centers, interpupillary distance, nose bridge anchor,
 * face normal and up vectors) from MediaPipe face landmarks.
 */
export class FaceReferenceGeometryExtractor {
  /**
   * Computes a metric 3D FaceReference object from MediaPipe face landmarks.
   */
  public extractFromLandmarks(
    landmarks: readonly NormalizedLandmark[],
  ): FaceReference | null {
    if (!landmarks || landmarks.length < 468) {
      return this.createCanonicalDefaultFaceReference()
    }

    // Compute face-relative canonical scale from the outer eye landmarks.
    const rawWidth = Math.abs(
      landmarks[INDEX_RIGHT_EYE_OUTER].x - landmarks[INDEX_LEFT_EYE_OUTER].x,
    )
    const scaleFactor = rawWidth > 1e-4 ? CANONICAL_FACE_WIDTH_CM / rawWidth : 14.0
    const centerNormX =
      (landmarks[INDEX_LEFT_EYE_OUTER].x + landmarks[INDEX_RIGHT_EYE_OUTER].x) / 2
    const centerNormY =
      (landmarks[INDEX_LEFT_EYE_OUTER].y + landmarks[INDEX_RIGHT_EYE_OUTER].y) / 2

    // Map normalized MediaPipe landmarks into the canonical mirrored face-local
    // coordinate system used by the GLB calibration pipeline.
    const toCanonical = (lm: NormalizedLandmark) =>
      new THREE.Vector3(
        (centerNormX - lm.x) * scaleFactor,
        (centerNormY - lm.y) * scaleFactor,
        -lm.z * scaleFactor,
      )

    // Calculate left and right eye centers.
    const leftEyeCenter = isFiniteLandmark(landmarks[INDEX_LEFT_PUPIL])
      ? toCanonical(landmarks[INDEX_LEFT_PUPIL])
      : toCanonical(landmarks[INDEX_LEFT_EYE_INNER])
          .add(toCanonical(landmarks[INDEX_LEFT_EYE_OUTER]))
          .multiplyScalar(0.5)

    const rightEyeCenter = isFiniteLandmark(landmarks[INDEX_RIGHT_PUPIL])
      ? toCanonical(landmarks[INDEX_RIGHT_PUPIL])
      : toCanonical(landmarks[INDEX_RIGHT_EYE_INNER])
          .add(toCanonical(landmarks[INDEX_RIGHT_EYE_OUTER]))
          .multiplyScalar(0.5)

    // Compute the nose bridge anchor from upper and mid-bridge landmarks.
    const nbUpper = landmarks[INDEX_NOSE_BRIDGE]
    const nbMid = landmarks[6]
    const noseBridgeAnchor = new THREE.Vector3()

    if (isFiniteLandmark(nbUpper) && isFiniteLandmark(nbMid)) {
      const pUpper = toCanonical(nbUpper)
      const pMid = toCanonical(nbMid)
      noseBridgeAnchor.addVectors(pUpper.multiplyScalar(0.7), pMid.multiplyScalar(0.3))
    } else if (isFiniteLandmark(nbUpper)) {
      noseBridgeAnchor.copy(toCanonical(nbUpper))
    } else {
      noseBridgeAnchor.addVectors(leftEyeCenter, rightEyeCenter).multiplyScalar(0.5)
    }

    // Nose bridge rests ~3.2 cm forward (+Z) relative to head center.
    if (noseBridgeAnchor.z < 1.0) {
      noseBridgeAnchor.z += 3.2
    }

    const faceCenter = new THREE.Vector3()
      .addVectors(leftEyeCenter, rightEyeCenter)
      .multiplyScalar(0.5)
    const eyeSeparation = leftEyeCenter.distanceTo(rightEyeCenter)

    // Face orientation basis vectors.
    const forehead = landmarks[INDEX_FOREHEAD]
    const noseTip = landmarks[INDEX_NOSE_TIP]
    const faceUp = new THREE.Vector3(0, 1, 0)

    if (isFiniteLandmark(forehead) && isFiniteLandmark(noseTip)) {
      const pForehead = toCanonical(forehead)
      const pNoseTip = toCanonical(noseTip)
      faceUp.subVectors(pForehead, pNoseTip).normalize()
    }

    const eyeAxis = new THREE.Vector3()
      .subVectors(rightEyeCenter, leftEyeCenter)
      .normalize()
    const faceNormal = new THREE.Vector3()
      .crossVectors(eyeAxis, faceUp)
      .normalize()

    return {
      leftEyeCenter,
      rightEyeCenter,
      eyeSeparation: eyeSeparation > 0 ? eyeSeparation : AVERAGE_HUMAN_IPD_CM,
      faceCenter,
      noseBridgeAnchor,
      faceNormal,
      faceUp,
    }
  }

  /**
   * Provides a deterministic canonical default face reference based on standard human IPD (6.3 cm).
   */
  public createCanonicalDefaultFaceReference(): FaceReference {
    const halfIpd = AVERAGE_HUMAN_IPD_CM / 2
    return {
      leftEyeCenter: new THREE.Vector3(halfIpd, 0, 0),
      rightEyeCenter: new THREE.Vector3(-halfIpd, 0, 0),
      eyeSeparation: AVERAGE_HUMAN_IPD_CM,
      faceCenter: new THREE.Vector3(0, MEDIA_PIPE_CANONICAL_EYE_LINE_Y_CM, 0),
      noseBridgeAnchor: new THREE.Vector3(0, 0, 3.2),
      faceNormal: new THREE.Vector3(0, 0, 1),
      faceUp: new THREE.Vector3(0, 1, 0),
    }
  }
}

function isFiniteLandmark(lm: NormalizedLandmark | undefined): lm is NormalizedLandmark {
  return (
    lm !== undefined &&
    Number.isFinite(lm.x) &&
    Number.isFinite(lm.y) &&
    Number.isFinite(lm.z)
  )
}
