import * as THREE from 'three'
import type { NativeOrientation } from './types'

export interface NormalizationResult {
  /** Runtime wrapper container (EyewearRuntimeRoot) holding the native GLB asset */
  readonly runtimeRoot: THREE.Group
  /** Reference to the original native GLB scene inside the wrapper container */
  readonly nativeScene: THREE.Object3D
  /** 4x4 matrix mapping points from native asset space to canonical eyewear space */
  readonly normalizationMatrix: THREE.Matrix4
  /** Bounding box of the normalized asset in canonical eyewear space */
  readonly canonicalBounds: THREE.Box3
  /** Dimensions (width, height, depth) in canonical eyewear space */
  readonly canonicalDimensions: THREE.Vector3
  /** Selected canonical origin point expressed in native asset space */
  readonly nativeOrigin: THREE.Vector3
}

/**
 * Wraps raw GLB models in a runtime hierarchy container (`EyewearRuntimeRoot` -> `NativeGLB`),
 * re-centering the specified anchor point to (0,0,0) and aligning PCA orientation vectors
 * to Canonical Eyewear Space (+X width, +Y height, +Z depth) without mutating raw file data.
 */
export class CanonicalNormalizer {
  /**
   * Normalizes a native GLB scene graph into Canonical Eyewear Space.
   */
  public normalize(
    nativeScene: THREE.Object3D,
    orientation: NativeOrientation,
    targetNativeOrigin: THREE.Vector3,
  ): NormalizationResult {
    if (!nativeScene) {
      throw new Error('CanonicalNormalizer requires a valid nativeScene Object3D.')
    }

    // 1. Create outer runtime wrapper container (controlled by ARRenderer)
    const runtimeRoot = new THREE.Group()
    runtimeRoot.name = 'EyewearRuntimeRoot'

    // 2. Create inner normalized container (holds static canonical orientation transform)
    const normalizedContainer = new THREE.Group()
    normalizedContainer.name = 'NormalizedAssetContainer'

    // 3. Name the child native GLB object for identification
    if (!nativeScene.name) {
      nativeScene.name = 'NativeGLB'
    }

    // 4. Hierarchy assembly: runtimeRoot -> normalizedContainer -> nativeScene
    runtimeRoot.add(normalizedContainer)
    normalizedContainer.add(nativeScene)

    // 5. Re-center native asset so target origin sits at (0,0,0) relative to normalized container
    const originOffset = targetNativeOrigin.clone().negate()
    nativeScene.position.copy(originOffset)

    // 6. Apply canonical orientation rotation to the inner normalized container
    // This isolates the normalization rotation from runtimeRoot (which ARRenderer rotates)
    const rotationQuaternion = new THREE.Quaternion().setFromRotationMatrix(
      orientation.rotationMatrixToCanonical,
    )
    normalizedContainer.quaternion.copy(rotationQuaternion)

    // Force full hierarchy matrix update
    runtimeRoot.updateMatrixWorld(true)

    // 5. Compute full normalization matrix M_norm = R_canonical * T(-origin)
    const translationMatrix = new THREE.Matrix4().makeTranslation(
      originOffset.x,
      originOffset.y,
      originOffset.z,
    )
    const normalizationMatrix = new THREE.Matrix4().multiplyMatrices(
      orientation.rotationMatrixToCanonical,
      translationMatrix,
    )

    // 6. Calculate canonical bounding box and dimensions in canonical space
    const canonicalBounds = new THREE.Box3().setFromObject(runtimeRoot)
    const canonicalDimensions = new THREE.Vector3()
    canonicalBounds.getSize(canonicalDimensions)

    return {
      runtimeRoot,
      nativeScene,
      normalizationMatrix,
      canonicalBounds,
      canonicalDimensions,
      nativeOrigin: targetNativeOrigin.clone(),
    }
  }
}
