import { FaceLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { halfLifeAlpha } from '../PoseSmoother'
import type { FacePose } from '../FacePose'
import { DEFAULT_FACE_OCCLUSION_SETTINGS, type FaceOcclusionSettings } from '../../types/FaceOcclusion'

const LANDMARK_COUNT = 478
const MIN_REFERENCE_FACE_WIDTH_CM = 8
const MAX_REFERENCE_FACE_WIDTH_CM = 24
const MIN_DEPTH_SCALE = 0.01
const MIN_SURFACE_SCALE = 0.5
const MAX_SURFACE_SCALE = 1.5
const MIN_SMOOTHING_HALF_LIFE = 0.001
const MAX_SMOOTHING_HALF_LIFE = 0.5
const MIN_CAMERA_DEPTH_CM = 0.25


/**
 * Depth-only face surface used to hide eyewear geometry that passes behind the
 * user's face. The surface is reconstructed from the same MediaPipe landmark
 * topology used by the existing debug mesh, but its vertices are projected
 * into the AR renderer's perspective camera space so the depth buffer is
 * comparable with the GLB.
 */
export class FaceOccluder {
  private readonly geometry: THREE.BufferGeometry
  private readonly material: THREE.MeshBasicMaterial
  private readonly mesh: THREE.Mesh
  private readonly positionAttribute: THREE.BufferAttribute
  private readonly trianglesByLandmarkCount: ReadonlyMap<number, number>
  private readonly smoothedLandmarks: THREE.Vector3[] = Array.from(
    { length: LANDMARK_COUNT },
    () => new THREE.Vector3(),
  )
  private hasSmoothedLandmarks = false
  private previousTimestampMs: number | null = null
  private settings: FaceOcclusionSettings = DEFAULT_FACE_OCCLUSION_SETTINGS
  private videoWidth = 0
  private videoHeight = 0
  private surfaceWidth = 1
  private surfaceHeight = 1

  constructor() {
    const triangles = buildTriangles()
    const indexArray = new Uint16Array(triangles.flat())

    this.geometry = new THREE.BufferGeometry()
    this.positionAttribute = new THREE.BufferAttribute(
      new Float32Array(LANDMARK_COUNT * 3),
      3,
    )
    this.positionAttribute.setUsage(THREE.DynamicDrawUsage)
    this.geometry.setAttribute('position', this.positionAttribute)
    this.geometry.setIndex(new THREE.BufferAttribute(indexArray, 1))

    const counts = new Map<number, number>()
    for (const count of [468, 478]) {
      let indexCount = 0
      for (const triangle of triangles) {
        if (triangle.every((index) => index < count)) {
          indexCount += 3
        }
      }
      counts.set(count, indexCount)
    }
    this.trianglesByLandmarkCount = counts

    this.material = new THREE.MeshBasicMaterial({
      colorWrite: false,
      depthTest: true,
      depthWrite: true,
      side: THREE.DoubleSide,
      transparent: false,
    })

    this.mesh = new THREE.Mesh(this.geometry, this.material)
    this.mesh.name = 'FaceOccluder'
    this.mesh.renderOrder = 0
    this.mesh.frustumCulled = false
    this.mesh.visible = false
  }

  getObject3D(): THREE.Object3D {
    return this.mesh
  }

  setSettings(settings: FaceOcclusionSettings): void {
    validateSettings(settings)
    const wasEnabled = this.settings.enabled
    this.settings = Object.freeze({ ...settings })

    if (!this.settings.enabled || (!wasEnabled && this.settings.enabled)) {
      this.reset()
      return
    }

    this.mesh.visible = this.hasSmoothedLandmarks
  }

  getSettings(): FaceOcclusionSettings {
    return this.settings
  }

  setVideoSourceDimensions(width: number, height: number): void {
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      return
    }

    this.videoWidth = width
    this.videoHeight = height
  }

  setSurfaceDimensions(width: number, height: number): void {
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      return
    }

    this.surfaceWidth = width
    this.surfaceHeight = height
  }

  update(
    landmarks: readonly NormalizedLandmark[] | null,
    facePose: FacePose | null,
    camera: THREE.PerspectiveCamera,
    timestampMs: number,
  ): void {
    if (!this.settings.enabled || !landmarks || !facePose || landmarks.length < 3) {
      this.mesh.visible = false
      if (!landmarks || !facePose) {
        this.reset()
      }
      return
    }

    validateTimestamp(timestampMs)

    const alpha = this.getSmoothingAlpha(timestampMs)
    const landmarkCount = Math.min(landmarks.length, LANDMARK_COUNT)
    const faceReferenceWidth = this.settings.referenceFaceWidthCm * facePose.scale

    for (let index = 0; index < landmarkCount; index += 1) {
      const landmark = landmarks[index]
      const target = this.projectLandmarkToCameraSpace(
        landmark,
        facePose,
        camera,
        faceReferenceWidth,
      )

      if (!this.hasSmoothedLandmarks) {
        this.smoothedLandmarks[index].copy(target)
      } else {
        this.smoothedLandmarks[index].lerp(target, alpha)
      }

      const offset = index * 3
      this.positionAttribute.array[offset] = this.smoothedLandmarks[index].x
      this.positionAttribute.array[offset + 1] = this.smoothedLandmarks[index].y
      this.positionAttribute.array[offset + 2] = this.smoothedLandmarks[index].z
    }

    for (let index = landmarkCount; index < LANDMARK_COUNT; index += 1) {
      const offset = index * 3
      this.positionAttribute.array[offset] = 0
      this.positionAttribute.array[offset + 1] = 0
      this.positionAttribute.array[offset + 2] = 10000
    }

    this.positionAttribute.needsUpdate = true
    this.geometry.setDrawRange(
      0,
      this.trianglesByLandmarkCount.get(landmarkCount) ?? this.trianglesByLandmarkCount.get(468) ?? 0,
    )
    this.hasSmoothedLandmarks = true
    this.previousTimestampMs = timestampMs
    this.mesh.visible = true
  }

  reset(): void {
    this.hasSmoothedLandmarks = false
    this.previousTimestampMs = null
    this.mesh.visible = false
  }

  dispose(): void {
    this.geometry.dispose()
    this.material.dispose()
  }

  private getSmoothingAlpha(timestampMs: number): number {
    if (this.previousTimestampMs === null) {
      return 1
    }

    const deltaSeconds = Math.min(
      0.25,
      Math.max(0, (timestampMs - this.previousTimestampMs) / 1000),
    )

    return halfLifeAlpha(
      deltaSeconds,
      this.settings.landmarkSmoothingHalfLifeSeconds,
    )
  }

  private projectLandmarkToCameraSpace(
    landmark: NormalizedLandmark,
    facePose: FacePose,
    camera: THREE.PerspectiveCamera,
    faceReferenceWidth: number,
  ): THREE.Vector3 {
    const mirroredX = 1 - landmark.x
    const cover = getCoverMapping(
      this.surfaceWidth,
      this.surfaceHeight,
      this.videoWidth,
      this.videoHeight,
    )

    const surfaceX = cover.offsetX + mirroredX * cover.renderedWidth
    const surfaceY = cover.offsetY + landmark.y * cover.renderedHeight
    const ndcX = (surfaceX / this.surfaceWidth) * 2 - 1
    const ndcY = 1 - (surfaceY / this.surfaceHeight) * 2

    // MediaPipe's face-landmark z is relative depth: smaller values are closer
    // to the camera. Its magnitude is approximately on the same scale as x,
    // so the current face scale converts it into the same centimeter-scale
    // runtime space as the existing face pose.
    const relativeDepth = -landmark.z * faceReferenceWidth * this.settings.depthScale
    const cameraZ = Math.min(
      -MIN_CAMERA_DEPTH_CM,
      facePose.position.z + relativeDepth + this.settings.depthBias,
    )

    const halfFovRadians = THREE.MathUtils.degToRad(camera.fov) / 2
    const halfHeight = Math.tan(halfFovRadians) * -cameraZ
    const halfWidth = halfHeight * camera.aspect
    const centerX = (cover.centerX / this.surfaceWidth) * 2 - 1
    const centerY = 1 - (cover.centerY / this.surfaceHeight) * 2
    const scaledX = centerX + (ndcX - centerX) * this.settings.surfaceScale
    const scaledY = centerY + (ndcY - centerY) * this.settings.surfaceScale

    return new THREE.Vector3(
      scaledX * halfWidth,
      scaledY * halfHeight,
      cameraZ,
    )
  }
}

function buildTriangles(): [number, number, number][] {
  const adjacency = Array.from({ length: LANDMARK_COUNT }, () => new Set<number>())

  for (const connection of FaceLandmarker.FACE_LANDMARKS_TESSELATION) {
    if (
      connection.start < 0 ||
      connection.end < 0 ||
      connection.start >= LANDMARK_COUNT ||
      connection.end >= LANDMARK_COUNT ||
      connection.start === connection.end
    ) {
      continue
    }

    adjacency[connection.start].add(connection.end)
    adjacency[connection.end].add(connection.start)
  }

  const triangles: [number, number, number][] = []

  for (let a = 0; a < LANDMARK_COUNT; a += 1) {
    for (const b of adjacency[a]) {
      if (b <= a) continue

      for (const c of adjacency[a]) {
        if (c <= b || !adjacency[b].has(c)) continue
        triangles.push([a, b, c])
      }
    }
  }

  return triangles
}

function getCoverMapping(
  surfaceWidth: number,
  surfaceHeight: number,
  videoWidth: number,
  videoHeight: number,
): { renderedWidth: number; renderedHeight: number; offsetX: number; offsetY: number; centerX: number; centerY: number } {
  if (videoWidth <= 0 || videoHeight <= 0) {
    return {
      renderedWidth: surfaceWidth,
      renderedHeight: surfaceHeight,
      offsetX: 0,
      offsetY: 0,
      centerX: surfaceWidth / 2,
      centerY: surfaceHeight / 2,
    }
  }

  const scale = Math.max(surfaceWidth / videoWidth, surfaceHeight / videoHeight)
  const renderedWidth = videoWidth * scale
  const renderedHeight = videoHeight * scale

  return {
    renderedWidth,
    renderedHeight,
    offsetX: (surfaceWidth - renderedWidth) / 2,
    offsetY: (surfaceHeight - renderedHeight) / 2,
    centerX: surfaceWidth / 2,
    centerY: surfaceHeight / 2,
  }
}

function validateTimestamp(timestampMs: number): void {
  if (!Number.isFinite(timestampMs)) {
    throw new Error('FaceOccluder requires a finite landmark timestamp.')
  }
}

function validateSettings(settings: FaceOcclusionSettings): void {
  if (
    typeof settings.enabled !== 'boolean' ||
    !Number.isFinite(settings.referenceFaceWidthCm) ||
    settings.referenceFaceWidthCm < MIN_REFERENCE_FACE_WIDTH_CM ||
    settings.referenceFaceWidthCm > MAX_REFERENCE_FACE_WIDTH_CM ||
    !Number.isFinite(settings.depthScale) ||
    settings.depthScale < MIN_DEPTH_SCALE ||
    !Number.isFinite(settings.depthBias) ||
    !Number.isFinite(settings.surfaceScale) ||
    settings.surfaceScale < MIN_SURFACE_SCALE ||
    settings.surfaceScale > MAX_SURFACE_SCALE ||
    !Number.isFinite(settings.landmarkSmoothingHalfLifeSeconds) ||
    settings.landmarkSmoothingHalfLifeSeconds < MIN_SMOOTHING_HALF_LIFE ||
    settings.landmarkSmoothingHalfLifeSeconds > MAX_SMOOTHING_HALF_LIFE
  ) {
    throw new Error('FaceOcclusionSettings contains an invalid value.')
  }
}
