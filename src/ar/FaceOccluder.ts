import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { halfLifeAlpha } from './PoseSmoother'
import type { FacePose } from './FacePose'
import {
  DEFAULT_FACE_OCCLUSION_SETTINGS,
  normalizeFaceOcclusionSettings,
  type FaceOcclusionSettings,
} from '../types/FaceOcclusion'
import {
  CANONICAL_TRIANGLES,
  CANONICAL_VERTEX_COUNT,
  MIRRORED_CANONICAL_LATERAL_BIAS,
} from './occlusion/canonicalFaceModel'
import { buildEarGeometry, buildSkullGeometry } from './occlusion/HeadProxy'
import { FaceSurfaceReconstructor, type ReconstructionPose } from './occlusion/FaceSurfaceReconstructor'
import {
  createDebugMaterial,
  createDepthOnlyMaterial,
  createFaceDepthMaterial,
  setDepthOnlyPush,
  setFaceDepthPush,
} from './occlusion/DepthOnlyMaterial'
import { FaceOccluder as LegacyFaceOccluder } from './occlusion/LegacyFaceOccluder'

/** If landmarks keep failing for this long the surface is dropped rather than held. */
const MAX_HOLD_MS = 500
const MAX_SMOOTHING_DT_S = 0.25
const DEPTH_RENDER_ORDER = -10

export interface FaceOccluderDiagnostics {
  readonly mode: FaceOcclusionSettings['mode']
  readonly hasSurface: boolean
  /** 0..1 agreement between landmarks and pose; low => rigid canonical fallback. */
  readonly confidence: number
  /** RMS landmark-vs-pose disagreement, fraction of face width. */
  readonly fitRatio: number
  readonly ingested: number
  readonly rejected: number
  readonly lastReconstructMs: number
}

/**
 * Depth-only occlusion volumes for the face, head and ears.
 *
 * Hardened mode (default)
 * -----------------------
 *  - The face surface is rebuilt in FACE-LOCAL space from the pose matrix plus
 *    bounded landmark personalisation (see FaceSurfaceReconstructor), so its
 *    depth is metrically correct, not guessed from normalised landmark z.
 *  - It is smoothed in face-local space (independent of head motion, so it does
 *    not trail behind a moving head) and then placed by the SAME smoothed pose
 *    as the glasses: the transform of `poseGroup` is applied per render frame,
 *    the vertices only change when a new detection arrives. Glasses and
 *    occluder therefore cannot drift relative to each other.
 *  - A cranium and two ears (HeadProxy) supply the depth that the open face
 *    mask lacks behind the ear-front plane, so a temple arm on the far side of
 *    a turned head is hidden instead of being drawn across the hair.
 *
 * Legacy mode
 * -----------
 * The previous screen-space landmark occluder, retained unchanged for A/B
 * comparison while the new one is validated on a live camera.
 *
 * Nothing in here throws from the per-frame path: failures degrade to
 * "hold last good surface, then hide".
 */
export class FaceOccluder {
  private readonly group = new THREE.Group()
  private readonly poseGroup = new THREE.Group()

  private readonly faceGeometry = new THREE.BufferGeometry()
  private readonly skullGeometry = buildSkullGeometry()
  private readonly earGeometry = buildEarGeometry()

  private readonly faceMaterial = createFaceDepthMaterial(0, 0)
  private readonly skullMaterial = createDepthOnlyMaterial(0)
  private readonly earMaterial = createDepthOnlyMaterial(0)

  private readonly faceMesh: THREE.Mesh
  private readonly skullMesh: THREE.Mesh
  private readonly earMesh: THREE.Mesh
  private readonly debugMeshes: THREE.Mesh[]
  private readonly debugMaterials: THREE.MeshBasicMaterial[]

  private readonly legacy = new LegacyFaceOccluder()
  private readonly reconstructor = new FaceSurfaceReconstructor()
  private readonly smoothedLocal = new Float32Array(CANONICAL_VERTEX_COUNT * 3)
  private readonly positionAttribute: THREE.BufferAttribute

  private settings: FaceOcclusionSettings = DEFAULT_FACE_OCCLUSION_SETTINGS
  private hasSurface = false
  private previousIngestMs: number | null = null
  private lastGoodMs = -Infinity
  private videoWidth = 0
  private videoHeight = 0
  private legacySettingsApplied: FaceOcclusionSettings | null = null

  private confidence = 0
  private fitRatio = 0
  private ingested = 0
  private rejected = 0
  private lastReconstructMs = 0

  constructor() {
    this.positionAttribute = new THREE.BufferAttribute(this.smoothedLocal, 3)
    this.positionAttribute.setUsage(THREE.DynamicDrawUsage)
    this.faceGeometry.setAttribute('position', this.positionAttribute)
    this.faceGeometry.setAttribute(
      'aLateral',
      new THREE.BufferAttribute(new Float32Array(MIRRORED_CANONICAL_LATERAL_BIAS), 1),
    )
    this.faceGeometry.setIndex(new THREE.BufferAttribute(new Uint16Array(CANONICAL_TRIANGLES), 1))

    this.faceMesh = this.makeDepthMesh(this.faceGeometry, this.faceMaterial, 'FaceSurfaceOccluder')
    this.skullMesh = this.makeDepthMesh(this.skullGeometry, this.skullMaterial, 'HeadOccluder')
    this.earMesh = this.makeDepthMesh(this.earGeometry, this.earMaterial, 'EarOccluder')

    this.debugMaterials = [
      createDebugMaterial(0x2bd4ff, true),
      createDebugMaterial(0xffb020, false),
      createDebugMaterial(0xff4d8d, false),
    ]
    this.debugMeshes = [
      this.makeDebugMesh(this.faceGeometry, this.debugMaterials[0], 'FaceOccluderDebug'),
      this.makeDebugMesh(this.skullGeometry, this.debugMaterials[1], 'HeadOccluderDebug'),
      this.makeDebugMesh(this.earGeometry, this.debugMaterials[2], 'EarOccluderDebug'),
    ]

    this.poseGroup.name = 'FaceOccluderPoseRoot'
    this.poseGroup.visible = false
    this.poseGroup.add(this.faceMesh, this.skullMesh, this.earMesh, ...this.debugMeshes)

    const legacyObject = this.legacy.getObject3D()
    legacyObject.visible = false

    this.group.name = 'FaceOccluder'
    this.group.add(this.poseGroup, legacyObject)

    this.applySettingsToScene()
  }

  /** Root object to add to the scene. */
  getObject3D(): THREE.Object3D {
    return this.group
  }

  /** Pose-driven root; other face-attached objects (e.g. shadow catcher) can parent to it. */
  getPoseRoot(): THREE.Object3D {
    return this.poseGroup
  }

  /** Smoothed face-local vertex positions (xyz-interleaved); undefined content until the first detection. */
  getFaceLocalPositions(): Float32Array {
    return this.smoothedLocal
  }

  hasFaceSurface(): boolean {
    return this.hasSurface
  }

  /** Live face-local face geometry (shared, do not dispose). */
  getFaceGeometry(): THREE.BufferGeometry {
    return this.faceGeometry
  }

  setSettings(input: FaceOcclusionSettings): void {
    const next = normalizeFaceOcclusionSettings(input)
    const wasEnabled = this.settings.enabled
    const modeChanged = this.settings.mode !== next.mode
    this.settings = next

    if (!next.enabled || (!wasEnabled && next.enabled) || modeChanged) {
      this.reset()
    }

    this.applySettingsToScene()
  }

  getSettings(): FaceOcclusionSettings {
    return this.settings
  }

  setVideoSourceDimensions(width: number, height: number): void {
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return
    this.videoWidth = width
    this.videoHeight = height
    this.legacy.setVideoSourceDimensions(width, height)
  }

  setSurfaceDimensions(width: number, height: number): void {
    this.legacy.setSurfaceDimensions(width, height)
  }

  /**
   * Feed one detection. `rawPose` must be the UNSMOOTHED pose that belongs to
   * `landmarks` (same frame). Called once per detection, not per render frame.
   */
  ingest(
    landmarks: readonly NormalizedLandmark[] | null,
    rawPose: ReconstructionPose | null,
    timestampMs: number,
  ): void {
    if (!this.settings.enabled || this.settings.mode !== 'hardened') return
    if (!landmarks || !rawPose || !Number.isFinite(timestampMs)) return

    const started = typeof performance !== 'undefined' ? performance.now() : 0
    const result = this.reconstructor.reconstruct(landmarks, rawPose, {
      videoWidth: this.videoWidth,
      videoHeight: this.videoHeight,
      trackerVerticalFovDeg: this.settings.trackerVerticalFovDeg,
      landmarkDepthBlend: this.settings.landmarkDepthBlend,
      maxDeviationCm: this.settings.maxLandmarkDeviationCm,
    })
    this.lastReconstructMs = typeof performance !== 'undefined' ? performance.now() - started : 0

    if (!result.ok) {
      this.rejected += 1
      return
    }

    this.ingested += 1
    this.confidence = result.confidence
    this.fitRatio = result.fitRatio

    const target = this.reconstructor.local
    let alpha = 1
    if (this.hasSurface && this.previousIngestMs !== null) {
      const dt = Math.min(MAX_SMOOTHING_DT_S, Math.max(0, (timestampMs - this.previousIngestMs) / 1000))
      if (dt <= 0) return // same frame delivered twice
      alpha = halfLifeAlpha(dt, this.settings.landmarkSmoothingHalfLifeSeconds)
    }

    if (alpha >= 1) {
      this.smoothedLocal.set(target)
    } else {
      for (let i = 0; i < this.smoothedLocal.length; i += 1) {
        this.smoothedLocal[i] += (target[i] - this.smoothedLocal[i]) * alpha
      }
    }

    this.positionAttribute.needsUpdate = true
    this.hasSurface = true
    this.previousIngestMs = timestampMs
    this.lastGoodMs = timestampMs
  }

  /**
   * Per render frame. `facePose` is the SMOOTHED pose that positions the
   * glasses. `landmarks` is only consumed by legacy mode.
   */
  update(
    landmarks: readonly NormalizedLandmark[] | null,
    facePose: FacePose | null,
    camera: THREE.PerspectiveCamera,
    timestampMs: number,
  ): void {
    if (!this.settings.enabled || !facePose) {
      this.poseGroup.visible = false
      this.legacy.getObject3D().visible = false
      if (!facePose) this.reset()
      return
    }

    if (this.settings.mode === 'legacy') {
      this.poseGroup.visible = false
      if (this.legacySettingsApplied !== this.settings) {
        this.legacy.setSettings(this.settings)
        this.legacySettingsApplied = this.settings
      }
      try {
        this.legacy.update(landmarks, facePose, camera, timestampMs)
      } catch {
        this.legacy.reset()
      }
      return
    }

    this.legacy.getObject3D().visible = false

    const stale = timestampMs - this.lastGoodMs > MAX_HOLD_MS
    const p = facePose.position
    const r = facePose.rotation
    const valid =
      Number.isFinite(p.x + p.y + p.z + r.x + r.y + r.z + r.w + facePose.scale) && facePose.scale > 0
    if (!this.hasSurface || stale || !valid) {
      this.poseGroup.visible = false
      return
    }

    this.poseGroup.position.set(p.x, p.y, p.z)
    this.poseGroup.quaternion.set(r.x, r.y, r.z, r.w).normalize()
    this.poseGroup.scale.setScalar(facePose.scale)
    this.poseGroup.visible = true
  }

  reset(): void {
    this.hasSurface = false
    this.previousIngestMs = null
    this.lastGoodMs = -Infinity
    this.confidence = 0
    this.poseGroup.visible = false
    this.legacy.reset()
  }

  getDiagnostics(): FaceOccluderDiagnostics {
    return {
      mode: this.settings.mode,
      hasSurface: this.hasSurface,
      confidence: this.confidence,
      fitRatio: this.fitRatio,
      ingested: this.ingested,
      rejected: this.rejected,
      lastReconstructMs: this.lastReconstructMs,
    }
  }

  dispose(): void {
    this.legacy.dispose()
    this.faceGeometry.dispose()
    this.skullGeometry.dispose()
    this.earGeometry.dispose()
    this.faceMaterial.dispose()
    this.skullMaterial.dispose()
    this.earMaterial.dispose()
    for (const material of this.debugMaterials) material.dispose()
  }

  private applySettingsToScene(): void {
    const s = this.settings
    setFaceDepthPush(this.faceMaterial, s.faceSurfaceBiasCm, s.templeClearanceCm)
    setDepthOnlyPush(this.skullMaterial, s.headProxyPushCm)
    setDepthOnlyPush(this.earMaterial, s.earProxyPushCm)

    this.faceMesh.visible = true
    this.skullMesh.visible = s.headProxyEnabled
    this.earMesh.visible = s.earProxyEnabled
    this.debugMeshes[0].visible = s.debugShowOccluders
    this.debugMeshes[1].visible = s.debugShowOccluders && s.headProxyEnabled
    this.debugMeshes[2].visible = s.debugShowOccluders && s.earProxyEnabled
  }

  private makeDepthMesh(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    name: string,
  ): THREE.Mesh {
    const mesh = new THREE.Mesh(geometry, material)
    mesh.name = name
    mesh.renderOrder = DEPTH_RENDER_ORDER
    mesh.frustumCulled = false
    return mesh
  }

  private makeDebugMesh(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    name: string,
  ): THREE.Mesh {
    const mesh = new THREE.Mesh(geometry, material)
    mesh.name = name
    mesh.renderOrder = 100
    mesh.frustumCulled = false
    mesh.visible = false
    return mesh
  }
}
