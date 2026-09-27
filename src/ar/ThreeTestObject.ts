import * as THREE from 'three'
import type { ThreeJsFaceTransform } from './coordinateTransform'

/**
 * Temporary geometry (8 cm per side) used only to verify that the face-pose
 * pipeline drives the transparent Three.js layer correctly. This object contains no product
 * calibration or eyewear-specific behavior.
 */
export class ThreeTestObject {
  readonly object: THREE.Mesh<THREE.BoxGeometry, THREE.MeshNormalMaterial>

  constructor() {
    const geometry = new THREE.BoxGeometry(8, 8, 8)
    const material = new THREE.MeshNormalMaterial({
      transparent: true,
      opacity: 0.85,
      wireframe: false,
    })

    this.object = new THREE.Mesh(geometry, material)
    this.object.position.set(0, 0, 0)
    this.object.visible = false
  }

  applyPose(transform: ThreeJsFaceTransform): void {
    this.object.position.copy(transform.position)
    this.object.quaternion.copy(transform.quaternion)
    this.object.scale.setScalar(transform.scale)
    this.object.visible = true
  }

  clearPose(): void {
    this.object.visible = false
  }

  dispose(): void {
    this.object.geometry.dispose()
    this.object.material.dispose()
  }
}
