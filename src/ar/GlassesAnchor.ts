import * as THREE from 'three'
import type { FacePose } from './FacePose'
import { facePoseToThreeJsTransform, type ThreeJsFaceTransform } from './coordinateTransform'
import type { Calibration } from '../types/Calibration'

/**
 * Mathematical identity only. It is not a product calibration value and is
 * used until product data supplies an actual calibration record in later tasks.
 */
export const IDENTITY_GLASSES_CALIBRATION: Calibration = Object.freeze({
  scale: 1,
  x: 0,
  y: 0,
  z: 0,
  rotationX: 0,
  rotationY: 0,
  rotationZ: 0,
})

/**
 * Pure face-pose + calibration transform composer.
 *
 * Responsibilities are deliberately limited to combining the already
 * canonical face pose with product-relative placement. It does not know about
 * React, products, loading, Three.js scenes, camera state, coordinate-system
 * conversion, or smoothing.
 *
 * Composition order:
 * 1. Start with the canonical face position/rotation/scale.
 * 2. Express calibration translation in face-local coordinates.
 * 3. Rotate that local offset by the face rotation and scale it by face scale.
 * 4. Apply calibration rotation after the face rotation.
 * 5. Multiply face scale by calibration scale.
 */
export class GlassesAnchor {
  private readonly calibrationRotation = new THREE.Quaternion()
  private readonly calibratedOffset = new THREE.Vector3()
  private readonly position = new THREE.Vector3()
  private readonly rotation = new THREE.Quaternion()

  compose(
    facePose: FacePose,
    calibration: Calibration,
  ): ThreeJsFaceTransform {
    validateFacePose(facePose)
    validateCalibration(calibration)

    // Coordinate representation stays authoritative in coordinateTransform;
    // GlassesAnchor consumes the canonical pose through that adapter and only
    // composes calibration-relative placement afterward.
    const faceTransform = facePoseToThreeJsTransform(facePose)
    const faceRotation = faceTransform.quaternion

    this.calibrationRotation.setFromEuler(
      new THREE.Euler(
        calibration.rotationX,
        calibration.rotationY,
        calibration.rotationZ,
        'XYZ',
      ),
    ).normalize()

    this.rotation.copy(faceRotation).multiply(this.calibrationRotation).normalize()

    this.calibratedOffset.set(calibration.x, calibration.y, calibration.z)
    this.calibratedOffset
      .multiplyScalar(faceTransform.scale)
      .applyQuaternion(faceRotation)

    this.position.copy(faceTransform.position)
    this.position.add(this.calibratedOffset)

    return {
      position: this.position.clone(),
      quaternion: this.rotation.clone(),
      scale: faceTransform.scale * calibration.scale,
    }
  }
}

function validateFacePose(pose: FacePose): void {
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
    throw new Error('GlassesAnchor received a FacePose with invalid numeric values.')
  }
}

function validateCalibration(calibration: Calibration): void {
  const values = [
    calibration.scale,
    calibration.x,
    calibration.y,
    calibration.z,
    calibration.rotationX,
    calibration.rotationY,
    calibration.rotationZ,
  ]

  if (!values.every(Number.isFinite) || calibration.scale <= 0) {
    throw new Error('GlassesAnchor received calibration with invalid numeric values.')
  }
}
