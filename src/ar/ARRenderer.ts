import * as THREE from 'three'
import type { FacePose } from './FacePose'
import {
  PoseSmoother,
  halfLifeAlpha,
  type PoseSmoothingOptions,
  type PoseSmoothingSettings,
} from './PoseSmoother'
import { GlassesAnchor } from './GlassesAnchor'
import { FaceOccluder, type FaceOccluderDiagnostics } from './FaceOccluder'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import {
  DEFAULT_FACE_OCCLUSION_SETTINGS,
  normalizeFaceOcclusionSettings,
  type FaceOcclusionSettings,
} from '../types/FaceOcclusion'
import type { Calibration } from '../types/Calibration'
import { ForegroundOcclusion } from './occlusion/ForegroundOcclusion'
import type { ForegroundMaskFrame } from './occlusion/ForegroundSegmenter'
import { ContactShadow } from './occlusion/ContactShadow'
import { CANONICAL_TRIANGLES } from './occlusion/canonicalFaceModel'
import { matchedVerticalFovDeg } from './occlusion/TrackerProjection'
import {
  FaceHeightField,
  UNKNOWN_FIT,
  collectFrameSamples,
  computeFitReport,
  type FitReport,
} from './occlusion/FitDiagnostics'
import {
  LightingSmoother,
  NEUTRAL_LIGHTING,
  estimateFromSample,
  type LightingSample,
} from './occlusion/LightEstimator'

export interface ARRendererOptions extends PoseSmoothingOptions {
  readonly onContextError?: (message: string) => void
  /** Non-fatal errors from the per-frame update. The render loop keeps running. */
  readonly onRuntimeError?: (message: string) => void
}

/** Renderer camera vertical FOV when tracker-matching is off (original behaviour). */
const DEFAULT_CAMERA_FOV_DEG = 60
const BASE_HEMISPHERE_INTENSITY = 1.4
const BASE_KEY_INTENSITY = 1.8
const FIT_INTERVAL_MS = 400
const CLEARANCE_HALF_LIFE_S = 0.4
const CLEARANCE_DEADBAND_CM = 0.04
const RUNTIME_ERROR_INTERVAL_MS = 1000

export interface OcclusionStatus {
  readonly occluder: FaceOccluderDiagnostics
  readonly fit: FitReport
  /** Forward lift currently applied to the glasses, cm. */
  readonly appliedClearanceCm: number
  readonly foregroundMaskActive: boolean
  readonly cameraFovDeg: number
}

/**
 * Owns the transparent Three.js overlay lifecycle and render loop.
 *
 * Pose updates are mutable runtime data. The render loop consumes the latest
 * available pose every animation frame, independently from MediaPipe's
 * inference cadence.
 *
 * Occlusion is composed of four cooperating layers (see docs/OCCLUSION_HARDENING.md):
 *   1. depth volumes (face surface + head + ears)      -> FaceOccluder
 *   2. screen-space foreground mask (hair/hands/objects)-> ForegroundOcclusion
 *   3. contact shadows on the face                      -> ContactShadow
 *   4. anatomical clearance + fit diagnostics           -> FitDiagnostics
 */
export class ARRenderer {
  private readonly canvas: HTMLCanvasElement
  private readonly onContextError?: (message: string) => void
  private readonly onRuntimeError?: (message: string) => void
  private readonly scene: THREE.Scene
  private readonly camera: THREE.PerspectiveCamera
  private readonly renderer: THREE.WebGLRenderer
  private readonly poseSmoother: PoseSmoother
  private readonly glassesAnchor: GlassesAnchor
  private readonly faceOccluder: FaceOccluder
  private readonly foreground: ForegroundOcclusion
  private readonly contactShadow: ContactShadow
  private readonly heightField = new FaceHeightField()
  private readonly lightingSmoother = new LightingSmoother()
  private glassesCalibration: Calibration | null = null
  private readonly modelLighting: readonly THREE.Light[]
  private readonly hemisphereLight: THREE.HemisphereLight
  private readonly keyLight: THREE.DirectionalLight
  private animationFrameId: number | null = null
  private running = false
  private contextLost = false
  private resizeObserver: ResizeObserver | null = null
  private model: THREE.Object3D | null = null
  private frameSamples: Float32Array | null = null
  private latestFacePose: FacePose | null = null
  private latestFaceLandmarks: readonly NormalizedLandmark[] | null = null
  private previousRenderTimeMs: number | null = null
  private settings: FaceOcclusionSettings = DEFAULT_FACE_OCCLUSION_SETTINGS
  private surfaceWidth = 1
  private surfaceHeight = 1
  private videoWidth = 0
  private videoHeight = 0
  private cameraFovDeg = DEFAULT_CAMERA_FOV_DEG
  private lastIngestedTimestamp = Number.NaN
  private maskAnchor: { x: number; y: number } | null = null
  private fit: FitReport = UNKNOWN_FIT
  private lastFitMs = -Infinity
  private appliedClearanceCm = 0
  private lightingTarget = NEUTRAL_LIGHTING
  private lastRuntimeErrorMs = -Infinity
  private readonly tmpWorldPosition = new THREE.Vector3()

  constructor(canvas: HTMLCanvasElement, options: ARRendererOptions = {}) {
    this.canvas = canvas
    this.onContextError = options.onContextError
    this.onRuntimeError = options.onRuntimeError
    this.poseSmoother = new PoseSmoother(options)
    this.glassesAnchor = new GlassesAnchor()
    this.faceOccluder = new FaceOccluder()
    this.foreground = new ForegroundOcclusion()
    this.contactShadow = new ContactShadow(
      this.faceOccluder.getFaceGeometry(),
      this.faceOccluder.getPoseRoot(),
    )

    this.scene = new THREE.Scene()
    this.scene.add(this.faceOccluder.getObject3D())
    this.scene.add(this.foreground.getDebugObject())
    this.scene.add(...this.contactShadow.getSceneObjects())
    this.applyOcclusionSettings(DEFAULT_FACE_OCCLUSION_SETTINGS)

    this.camera = new THREE.PerspectiveCamera(DEFAULT_CAMERA_FOV_DEG, 1, 0.1, 2000)
    // MediaPipe's facial transformation matrix is expressed in metric face
    // space (centimeter-scale coordinates). The camera sits at the origin and
    // looks down -Z, matching that coordinate convention directly.
    this.camera.position.set(0, 0, 0)

    // GLB eyewear assets use physically based materials. These local lights
    // keep the supplied model visible without requiring an external
    // environment map or service.
    this.hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, BASE_HEMISPHERE_INTENSITY)
    this.keyLight = new THREE.DirectionalLight(0xffffff, BASE_KEY_INTENSITY)
    this.keyLight.position.set(0, 20, 40)
    this.modelLighting = [this.hemisphereLight, this.keyLight]
    this.scene.add(this.hemisphereLight, this.keyLight)

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: true,
    })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.setClearColor(0x000000, 0)
    // Shadow maps are only rendered while a shadow-casting light is active
    // (ContactShadow toggles that), so leaving this enabled costs nothing idle.
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFShadowMap

    this.handleContextLost = this.handleContextLost.bind(this)
    this.handleContextRestored = this.handleContextRestored.bind(this)

    canvas.addEventListener('webglcontextlost', this.handleContextLost, false)
    canvas.addEventListener('webglcontextrestored', this.handleContextRestored, false)

    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(canvas.parentElement ?? canvas)
    this.resize()
  }

  start(): void {
    if (this.running) {
      return
    }

    this.running = true
    this.previousRenderTimeMs = null
    this.scheduleRender()
  }

  stop(): void {
    this.running = false
    this.previousRenderTimeMs = null

    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
  }

  resize(): void {
    const host = this.canvas.parentElement
    const width = host?.clientWidth ?? this.canvas.clientWidth
    const height = host?.clientHeight ?? this.canvas.clientHeight

    if (width <= 0 || height <= 0) {
      return
    }

    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.setSize(width, height, false)
    this.surfaceWidth = width
    this.surfaceHeight = height
    this.faceOccluder.setSurfaceDimensions(width, height)

    this.camera.aspect = width / height
    this.applyCameraModel()
    this.syncForegroundGeometry()
  }

  getScene(): THREE.Scene {
    return this.scene
  }

  getCamera(): THREE.Camera {
    return this.camera
  }

  getRenderer(): THREE.WebGLRenderer {
    return this.renderer
  }

  isContextLost(): boolean {
    return this.contextLost
  }

  /**
   * Attach the loaded GLB scene to the AR scene. Face-relative placement is
   * delegated to GlassesAnchor; this renderer contains no product-specific
   * transform math.
   */
  setModel(model: THREE.Object3D): void {
    if (this.model === model) {
      return
    }

    this.clearModel()
    this.model = model
    this.model.visible = false
    this.model.renderOrder = 1
    this.model.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return
      const materials = Array.isArray(object.material) ? object.material : [object.material]
      for (const material of materials) {
        material.depthTest = true
        material.needsUpdate = true
      }
    })
    this.scene.add(this.model)

    this.foreground.attach(this.model)
    this.contactShadow.setCasterRoot(this.model)
    try {
      this.frameSamples = collectFrameSamples(this.model)
    } catch {
      this.frameSamples = null
    }
    this.fit = UNKNOWN_FIT
    this.lastFitMs = -Infinity
    this.appliedClearanceCm = 0
  }

  clearModel(): void {
    if (!this.model) {
      return
    }

    this.scene.remove(this.model)
    this.model.visible = false
    this.model = null
    this.frameSamples = null
    this.contactShadow.setCasterRoot(null)
    this.fit = UNKNOWN_FIT
    this.appliedClearanceCm = 0
  }

  setPoseSmoothingSettings(settings: PoseSmoothingSettings): void {
    this.poseSmoother.setSettings(settings)
  }

  setGlassesCalibration(calibration: Calibration): void {
    this.glassesCalibration = Object.freeze({ ...calibration })
    // A new calibration invalidates the fit measurement.
    this.lastFitMs = -Infinity
  }

  setFaceOcclusionSettings(input: FaceOcclusionSettings): void {
    this.applyOcclusionSettings(normalizeFaceOcclusionSettings(input))
  }

  setVideoSourceDimensions(width: number, height: number): void {
    this.faceOccluder.setVideoSourceDimensions(width, height)
    if (width === this.videoWidth && height === this.videoHeight) return
    if (!(width > 0) || !(height > 0)) return
    this.videoWidth = width
    this.videoHeight = height
    this.applyCameraModel()
    this.syncForegroundGeometry()
  }

  /** Latest foreground (hair/hand/object) mask, or null to drop it. */
  setForegroundMask(frame: ForegroundMaskFrame | null): void {
    this.foreground.setMask(frame)
    this.maskAnchor = frame ? { x: frame.anchorX, y: frame.anchorY } : null
    // A fresh mask is aligned with the face at capture time: no shift yet.
    if (frame) this.foreground.setMotionShift(0, 0)
  }

  /** Optional lighting sample from the video; only used when lightingMatchEnabled. */
  setLightingSample(sample: LightingSample | null): void {
    this.lightingTarget = sample ? estimateFromSample(sample) : NEUTRAL_LIGHTING
  }

  getOcclusionStatus(): OcclusionStatus {
    return {
      occluder: this.faceOccluder.getDiagnostics(),
      fit: this.fit,
      appliedClearanceCm: this.appliedClearanceCm,
      foregroundMaskActive: this.foreground.isActive(),
      cameraFovDeg: this.cameraFovDeg,
    }
  }

  /**
   * Paired detection input: the landmarks and the UNSMOOTHED pose of the same
   * frame. Preferred over the two separate setters because the occluder
   * personalises the face surface from landmarks relative to that exact pose.
   */
  setFaceObservation(
    landmarks: readonly NormalizedLandmark[] | null,
    pose: FacePose | null,
  ): void {
    this.setFacePose(pose)
    this.setFaceLandmarks(pose ? landmarks : null)
  }

  setFaceLandmarks(landmarks: readonly NormalizedLandmark[] | null): void {
    this.latestFaceLandmarks = landmarks

    if (landmarks === null) {
      this.faceOccluder.reset()
      this.lastIngestedTimestamp = Number.NaN
      return
    }

    const pose = this.latestFacePose
    if (pose && pose.timestampMs !== this.lastIngestedTimestamp) {
      this.lastIngestedTimestamp = pose.timestampMs
      try {
        this.faceOccluder.ingest(landmarks, pose, pose.timestampMs)
        this.updateMaskShift(landmarks)
      } catch (error) {
        this.reportRuntimeError('Face occluder ingest failed', error)
      }
    }
  }

  setFacePose(pose: FacePose | null): void {
    this.latestFacePose = pose

    if (pose === null) {
      this.poseSmoother.reset()
      this.faceOccluder.reset()
      this.lastIngestedTimestamp = Number.NaN
      this.model && (this.model.visible = false)
    }
  }

  dispose(): void {
    this.stop()
    this.resizeObserver?.disconnect()
    this.resizeObserver = null
    this.latestFacePose = null
    this.latestFaceLandmarks = null
    this.glassesCalibration = null
    this.poseSmoother.reset()

    this.canvas.removeEventListener('webglcontextlost', this.handleContextLost)
    this.canvas.removeEventListener('webglcontextrestored', this.handleContextRestored)

    this.clearModel()
    this.contactShadow.dispose()
    this.foreground.dispose()
    this.faceOccluder.dispose()

    for (const light of this.modelLighting) {
      this.scene.remove(light)
    }

    // preserveDrawingBuffer keeps the last rendered frame visible after the
    // render loop stops. Clear that frame before disposing the renderer so
    // stopping the camera also removes the transparent AR overlay pixels.
    if (!this.contextLost) {
      this.renderer.clear(true, true, true)
    }

    this.renderer.dispose()
    this.renderer.renderLists.dispose()
    this.scene.clear()
  }

  private applyOcclusionSettings(settings: FaceOcclusionSettings): void {
    const previous = this.settings
    this.settings = settings
    this.faceOccluder.setSettings(settings)
    this.foreground.setSettings(settings)
    this.contactShadow.setSettings(settings)

    if (previous.matchTrackerCamera !== settings.matchTrackerCamera ||
        previous.trackerVerticalFovDeg !== settings.trackerVerticalFovDeg) {
      // `camera` may not exist yet during construction.
      if (this.camera) this.applyCameraModel()
    }
    if (!settings.lightingMatchEnabled && this.hemisphereLight) this.applyLighting(NEUTRAL_LIGHTING)
    this.lastFitMs = -Infinity
  }

  /**
   * Camera vertical FOV. Off = the original 60 degrees. On = derived from the
   * tracker's virtual camera and the video cover-crop so that pose-space points
   * project onto the same pixels as the real face (see TrackerProjection).
   */
  private applyCameraModel(): void {
    const fov =
      this.settings.matchTrackerCamera && this.videoWidth > 0 && this.videoHeight > 0
        ? matchedVerticalFovDeg(
            this.surfaceWidth,
            this.surfaceHeight,
            this.videoWidth,
            this.videoHeight,
            this.settings.trackerVerticalFovDeg,
          )
        : DEFAULT_CAMERA_FOV_DEG

    if (fov !== this.cameraFovDeg || this.camera.fov !== fov) {
      this.cameraFovDeg = fov
      this.camera.fov = fov
    }
    this.camera.updateProjectionMatrix()
  }

  private syncForegroundGeometry(): void {
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2())
    this.foreground.setGeometry(size.x, size.y, this.videoWidth, this.videoHeight)
  }

  private updateMaskShift(landmarks: readonly NormalizedLandmark[]): void {
    if (!this.maskAnchor) return
    const indices = [168, 1, 33, 263]
    let x = 0
    let y = 0
    for (const i of indices) {
      x += landmarks[i].x
      y += landmarks[i].y
    }
    this.foreground.setMotionShift(x / indices.length - this.maskAnchor.x, y / indices.length - this.maskAnchor.y)
  }

  private scheduleRender(): void {
    if (!this.running) {
      return
    }

    this.animationFrameId = requestAnimationFrame((timestampMs) => {
      this.animationFrameId = null

      if (!this.running) {
        return
      }

      const deltaSeconds = this.previousRenderTimeMs === null
        ? 0
        : Math.max(0, (timestampMs - this.previousRenderTimeMs) / 1000)
      this.previousRenderTimeMs = timestampMs

      if (!this.contextLost) {
        // A failure in one frame's update must never stop the loop: previously
        // an exception here skipped scheduleRender() and froze the overlay.
        try {
          this.updateModel(deltaSeconds)
        } catch (error) {
          this.reportRuntimeError('AR update failed', error)
        }

        try {
          this.renderer.render(this.scene, this.camera)
        } catch (error) {
          this.reportRuntimeError('AR render failed', error)
        }
      }

      this.scheduleRender()
    })
  }

  private updateModel(deltaSeconds: number): void {
    if (!this.latestFacePose) {
      this.model && (this.model.visible = false)
      this.faceOccluder.reset()
      this.contactShadow.update(false, this.tmpWorldPosition)
      return
    }

    const smoothedPose = this.poseSmoother.update(this.latestFacePose, deltaSeconds)
    this.faceOccluder.update(
      this.latestFaceLandmarks,
      smoothedPose,
      this.camera,
      this.latestFacePose.timestampMs,
    )

    if (!this.model || !this.glassesCalibration) {
      this.model && (this.model.visible = false)
      this.contactShadow.update(false, this.tmpWorldPosition)
      return
    }

    this.updateClearance(deltaSeconds)
    this.updateLighting(deltaSeconds)

    const calibration: Calibration =
      this.appliedClearanceCm === 0
        ? this.glassesCalibration
        : { ...this.glassesCalibration, z: this.glassesCalibration.z + this.appliedClearanceCm }

    const anchoredTransform = this.glassesAnchor.compose(smoothedPose, calibration)

    this.model.position.copy(anchoredTransform.position)
    this.model.quaternion.copy(anchoredTransform.quaternion)
    this.model.scale.setScalar(anchoredTransform.scale)
    this.model.visible = true

    this.tmpWorldPosition.set(smoothedPose.position.x, smoothedPose.position.y, smoothedPose.position.z)
    this.contactShadow.update(true, this.tmpWorldPosition)
  }

  /**
   * Measures frame-vs-face embedding (in face space) a few times a second and
   * moves `appliedClearanceCm` toward the lift that un-embeds the frame. The
   * measurement excludes the current lift, so this is not a feedback loop.
   */
  private updateClearance(deltaSeconds: number): void {
    const s = this.settings
    const active =
      s.enabled && s.mode === 'hardened' && s.anatomicalClearanceEnabled && this.faceOccluder.hasFaceSurface()

    const now = performance.now()
    if (active && this.frameSamples && this.glassesCalibration && now - this.lastFitMs >= FIT_INTERVAL_MS) {
      this.lastFitMs = now
      this.heightField.rebuild(this.faceOccluder.getFaceLocalPositions(), CANONICAL_TRIANGLES)
      this.fit = computeFitReport(this.frameSamples, this.glassesCalibration, this.heightField)
    }

    const target = active ? Math.min(s.maxClearanceCm, this.fit.suggestedForwardCm) : 0
    const error = target - this.appliedClearanceCm
    if (Math.abs(error) <= CLEARANCE_DEADBAND_CM && target !== 0) return

    const alpha = deltaSeconds > 0 ? halfLifeAlpha(Math.min(deltaSeconds, 0.25), CLEARANCE_HALF_LIFE_S) : 1
    this.appliedClearanceCm += error * alpha
    if (Math.abs(this.appliedClearanceCm) < 1e-4) this.appliedClearanceCm = 0
  }

  private updateLighting(deltaSeconds: number): void {
    if (!this.settings.lightingMatchEnabled) return
    const estimate = this.lightingSmoother.update(this.lightingTarget, deltaSeconds)
    this.applyLighting(estimate)
  }

  private applyLighting(estimate: typeof NEUTRAL_LIGHTING): void {
    this.hemisphereLight.intensity = BASE_HEMISPHERE_INTENSITY * estimate.intensityScale
    this.keyLight.intensity = BASE_KEY_INTENSITY * estimate.intensityScale
    this.keyLight.position.set(estimate.directionX * 30, 20 + estimate.directionY * 15, 40)
    this.contactShadow.setLightDirection(-0.25 + estimate.directionX * 0.8, 0.9 + estimate.directionY * 0.3, 0.55)
  }

  private reportRuntimeError(context: string, error: unknown): void {
    const now = performance.now()
    if (now - this.lastRuntimeErrorMs < RUNTIME_ERROR_INTERVAL_MS) return
    this.lastRuntimeErrorMs = now
    const detail = error instanceof Error ? error.message : String(error)
    console.error(`[ARRenderer] ${context}: ${detail}`)
    this.onRuntimeError?.(`${context}: ${detail}`)
  }

  private handleContextLost(event: Event): void {
    event.preventDefault()
    this.contextLost = true
    this.onContextError?.('WebGL context was lost. Waiting for the browser to restore it.')
  }

  private handleContextRestored(): void {
    this.contextLost = false
    this.resize()
    this.renderer.resetState()
    this.onContextError?.('')

    if (this.running) {
      this.renderer.render(this.scene, this.camera)
    }
  }
}
