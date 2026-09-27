import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import type { GLTF } from 'three/addons/loaders/GLTFLoader.js'

export type GLBLoadStatus = 'idle' | 'loading' | 'success' | 'failure'

export interface GLBLoadState {
  readonly status: GLBLoadStatus
  readonly url: string | null
  readonly error: Error | null
}

export interface LoadedGLB {
  readonly scene: THREE.Group
  readonly animations: readonly THREE.AnimationClip[]
  readonly gltf: GLTF
}

export interface GLBLoaderOptions {
  readonly onStateChange?: (state: GLBLoadState) => void
}

/**
 * Owns asynchronous GLB loading and disposal of the resources belonging to a
 * loaded model. It deliberately has no product, calibration, React, camera,
 * or renderer responsibilities.
 *
 * Loading is asynchronous: callers can keep the camera, MediaPipe tracker,
 * and Three.js render loop running while a model is fetched and parsed.
 *
 * A monotonically increasing request id prevents a late result from an older
 * load from becoming the active result after a newer load or dispose() call.
 */
export class GLBLoader {
  private readonly loader: GLTFLoader
  private readonly onStateChange?: (state: GLBLoadState) => void
  private requestId = 0
  private activeModel: LoadedGLB | null = null
  private state: GLBLoadState = {
    status: 'idle',
    url: null,
    error: null,
  }

  constructor(options: GLBLoaderOptions = {}) {
    this.loader = new GLTFLoader()
    this.onStateChange = options.onStateChange
  }

  getState(): GLBLoadState {
    return this.state
  }

  getActiveModel(): LoadedGLB | null {
    return this.activeModel
  }

  /**
   * Start loading a GLB without blocking any other runtime subsystem.
   *
   * A successful load does not dispose the previous model. The caller owns
   * the active runtime model and must remove it from the renderer before
   * calling disposeLoadedGLB(). This is what lets the current model remain
   * visible while a replacement is loading.
   */
  async load(url: string): Promise<LoadedGLB> {
    if (!url.trim()) {
      const error = new Error('GLBLoader requires a non-empty model URL.')
      this.setState({ status: 'failure', url: null, error })
      throw error
    }

    const currentRequestId = ++this.requestId
    this.setState({ status: 'loading', url, error: null })

    try {
      const gltf = await this.loader.loadAsync(url)

      if (currentRequestId !== this.requestId) {
        disposeGLTF(gltf)
        throw new Error('GLB load was superseded by a newer load request.')
      }

      const loadedModel: LoadedGLB = {
        scene: gltf.scene,
        animations: gltf.animations,
        gltf,
      }

      this.activeModel = loadedModel
      this.setState({ status: 'success', url, error: null })
      return loadedModel
    } catch (unknownError) {
      const error = toError(unknownError)

      if (currentRequestId !== this.requestId) {
        throw error
      }

      this.setState({ status: 'failure', url, error })
      throw error
    }
  }

  /**
   * Invalidate in-flight requests and release the loader's reference.
   * Runtime model disposal is owned by the caller so a replacement can be
   * attached before the old GLTF resources are disposed.
   */
  clear(): void {
    this.requestId += 1
    this.activeModel = null

    this.setState({
      status: 'idle',
      url: null,
      error: null,
    })
  }

  dispose(): void {
    this.clear()
  }

  private setState(state: GLBLoadState): void {
    this.state = state
    this.onStateChange?.(state)
  }
}

/**
 * Dispose every GPU-backed resource owned by a GLTF scene.
 *
 * GLTFLoader can create shared textures/materials across meshes, so resources
 * are deduplicated before disposal. Object3D hierarchy itself is discarded by
 * the caller after this traversal; no renderer disposal is performed here.
 */
export function disposeLoadedGLB(model: LoadedGLB): void {
  disposeObject3D(model.scene)
}

function disposeGLTF(gltf: GLTF): void {
  disposeObject3D(gltf.scene)
}

function disposeObject3D(root: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  const textures = new Set<THREE.Texture>()

  root.traverse((object) => {
    const mesh = object as THREE.Mesh

    if (mesh.geometry instanceof THREE.BufferGeometry) {
      geometries.add(mesh.geometry)
    }

    if (!('material' in mesh) || !mesh.material) {
      return
    }

    const meshMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    for (const material of meshMaterials) {
      if (!(material instanceof THREE.Material)) {
        continue
      }

      materials.add(material)
      collectMaterialTextures(material, textures)
    }
  })

  for (const texture of textures) {
    texture.dispose()
  }

  for (const material of materials) {
    material.dispose()
  }

  for (const geometry of geometries) {
    geometry.dispose()
  }
}

function collectMaterialTextures(material: THREE.Material, textures: Set<THREE.Texture>): void {
  const materialRecord = material as unknown as Record<string, unknown>

  for (const value of Object.values(materialRecord)) {
    if (value instanceof THREE.Texture) {
      textures.add(value)
      continue
    }

    if (Array.isArray(value)) {
      for (const nestedValue of value) {
        if (nestedValue instanceof THREE.Texture) {
          textures.add(nestedValue)
        }
      }
    }
  }
}

function toError(error: unknown): Error {
  if (error instanceof Error) {
    return error
  }

  return new Error(typeof error === 'string' ? error : 'GLB model loading failed.')
}
