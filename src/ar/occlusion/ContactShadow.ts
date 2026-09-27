import * as THREE from 'three'
import type { FaceOcclusionSettings } from '../../types/FaceOcclusion'

const SHADOW_DISTANCE_CM = 40
const SHADOW_HALF_EXTENT_CM = 14
const DEFAULT_LIGHT_DIRECTION = new THREE.Vector3(-0.25, 0.9, 0.55).normalize()
const CATCHER_RENDER_ORDER = -5

/**
 * Soft contact shadows of the glasses on the face (Rule 5: nose bridge, upper
 * cheeks).
 *
 * A dedicated shadow-only DirectionalLight (intensity 0, so it does not change
 * how the glasses are lit) renders a shadow map of the glasses. The face
 * surface, drawn a second time with THREE.ShadowMaterial (transparent black
 * where shadowed, nothing elsewhere), receives it. The catcher is a child of
 * the pose-driven occluder root, so it follows the face exactly.
 *
 * The light is fixed in CAMERA space (a room light does not turn with the
 * head); only its target follows the face. Its direction can be steered by the
 * light estimate.
 */
export class ContactShadow {
  private readonly light = new THREE.DirectionalLight(0xffffff, 0)
  private readonly catcherMaterial = new THREE.ShadowMaterial({
    opacity: 0.25,
    transparent: true,
    depthWrite: false,
  })
  private readonly catcher: THREE.Mesh
  private readonly direction = DEFAULT_LIGHT_DIRECTION.clone()
  private enabled = true
  private casterRoot: THREE.Object3D | null = null

  constructor(faceGeometry: THREE.BufferGeometry, faceParent: THREE.Object3D) {
    this.light.name = 'ContactShadowLight'
    this.light.castShadow = true
    this.light.shadow.mapSize.set(1024, 1024)
    const camera = this.light.shadow.camera
    camera.left = -SHADOW_HALF_EXTENT_CM
    camera.right = SHADOW_HALF_EXTENT_CM
    camera.top = SHADOW_HALF_EXTENT_CM
    camera.bottom = -SHADOW_HALF_EXTENT_CM
    camera.near = SHADOW_DISTANCE_CM - 2 * SHADOW_HALF_EXTENT_CM
    camera.far = SHADOW_DISTANCE_CM + 2 * SHADOW_HALF_EXTENT_CM
    camera.updateProjectionMatrix()
    this.light.shadow.bias = -0.0003
    this.light.shadow.normalBias = 0.03
    this.light.shadow.radius = 4

    this.catcher = new THREE.Mesh(faceGeometry, this.catcherMaterial)
    this.catcher.name = 'FaceShadowCatcher'
    this.catcher.receiveShadow = true
    this.catcher.renderOrder = CATCHER_RENDER_ORDER
    this.catcher.frustumCulled = false
    this.catcher.visible = false
    faceParent.add(this.catcher)
  }

  /** Objects that must live directly in the scene (light + its target). */
  getSceneObjects(): THREE.Object3D[] {
    return [this.light, this.light.target]
  }

  setSettings(settings: FaceOcclusionSettings): void {
    this.enabled = settings.enabled && settings.contactShadowsEnabled && settings.contactShadowOpacity > 0
    this.catcherMaterial.opacity = settings.contactShadowOpacity
    this.applyEnabled()
  }

  /** Opaque glasses meshes cast; glass/lens (transparent) meshes do not. */
  setCasterRoot(root: THREE.Object3D | null): void {
    this.casterRoot = root
    root?.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return
      const materials = Array.isArray(object.material) ? object.material : [object.material]
      const opaque = materials.every(
        (m: THREE.Material) => !m.transparent && m.opacity >= 0.9,
      )
      object.castShadow = opaque
      object.receiveShadow = false
    })
    this.applyEnabled()
  }

  /** Steer the light. `x`/`y` are direction components in camera space (z is forward-facing). */
  setLightDirection(x: number, y: number, z: number): void {
    if (![x, y, z].every(Number.isFinite)) return
    this.direction.set(x, y, z)
    if (this.direction.lengthSq() < 1e-6) this.direction.copy(DEFAULT_LIGHT_DIRECTION)
    this.direction.normalize()
  }

  resetLightDirection(): void {
    this.direction.copy(DEFAULT_LIGHT_DIRECTION)
  }

  /** Per frame: follow the face; only active while there is a face and a model. */
  update(faceVisible: boolean, faceWorldPosition: THREE.Vector3): void {
    const active = this.enabled && faceVisible && this.casterRoot !== null
    this.catcher.visible = active
    this.light.castShadow = active
    if (!active) return

    this.light.target.position.copy(faceWorldPosition)
    this.light.position.copy(faceWorldPosition).addScaledVector(this.direction, SHADOW_DISTANCE_CM)
    this.light.target.updateMatrixWorld()
  }

  dispose(): void {
    this.catcherMaterial.dispose()
    this.light.shadow.map?.dispose()
    this.catcher.removeFromParent()
  }

  private applyEnabled(): void {
    this.light.castShadow = this.enabled && this.casterRoot !== null
    if (!this.enabled) this.catcher.visible = false
  }
}
