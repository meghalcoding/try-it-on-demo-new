import * as THREE from 'three'
import type { Calibration } from '../../types/Calibration'

/** Axis identifier for geometric analysis */
export type PrimaryAxis = 'X' | 'Y' | 'Z' | '-X' | '-Y' | '-Z'

/** Analysis of a single mesh node within the GLB scene graph */
export interface MeshAnalysis {
  readonly nodeName: string
  readonly parentName: string | null
  readonly vertexCount: number
  readonly triangleCount: number
  readonly localBounds: THREE.Box3
  readonly worldBounds: THREE.Box3
  readonly center: THREE.Vector3
  readonly dimensions: THREE.Vector3
  readonly materialNames: readonly string[]
  readonly materialCount: number
  readonly isTransparent: boolean
  readonly worldTransform: THREE.Matrix4
  readonly threeMesh?: THREE.Mesh
}

/** Node structure analysis for scene graph traversal */
export interface NodeAnalysis {
  readonly name: string
  readonly parentName: string | null
  readonly childrenNames: readonly string[]
  readonly isMesh: boolean
  readonly worldTransform: THREE.Matrix4
  readonly worldBounds: THREE.Box3
}

/** Axis mapping derived from Principal Component Analysis (PCA) */
export interface NativeOrientation {
  readonly widthAxis: PrimaryAxis
  readonly heightAxis: PrimaryAxis
  readonly depthAxis: PrimaryAxis
  readonly isRightHanded: boolean
  readonly pcaEigenvalues: readonly number[]
  readonly pcaEigenvectors: readonly THREE.Vector3[]
  readonly rotationMatrixToCanonical: THREE.Matrix4
}

/** Evidence gathered for a detected eyewear part */
export interface PartEvidence {
  readonly source: 'semantic' | 'geometric' | 'hybrid' | 'fallback'
  readonly confidence: number
  readonly description: string
}

/** Individual part candidates detected by EyewearPartDetector */
export interface DetectedPartCandidate {
  readonly name: string
  readonly center: THREE.Vector3
  readonly bounds: THREE.Box3
  readonly vertexCount: number
  readonly evidence: PartEvidence
}

/** Structural eyewear parts detected across the asset */
export interface EyewearPartDetection {
  readonly leftLensCenter: THREE.Vector3 | null
  readonly rightLensCenter: THREE.Vector3 | null
  readonly bridgeCenter: THREE.Vector3 | null
  readonly leftTempleStart: THREE.Vector3 | null
  readonly rightTempleStart: THREE.Vector3 | null
  readonly lensSeparation: number | null
  readonly frameWidth: number
  readonly frameHeight: number
  readonly frameDepth: number
  readonly detectedLenses: readonly DetectedPartCandidate[]
  readonly detectedBridge: DetectedPartCandidate | null
  readonly detectedTemples: readonly DetectedPartCandidate[]
  readonly anchorOriginSource: 'lens_midpoint' | 'bridge_center' | 'symmetrical_frame_center' | 'bounding_box_center'
}

/** Normalized profile of the eyewear asset in canonical eyewear space */
export interface EyewearAssetProfile {
  readonly canonicalWidth: number
  readonly canonicalHeight: number
  readonly canonicalDepth: number
  readonly leftLensCenter: THREE.Vector3 | null
  readonly rightLensCenter: THREE.Vector3 | null
  readonly bridgeCenter: THREE.Vector3 | null
  readonly lensSeparation: number | null
  readonly leftTempleStart: THREE.Vector3 | null
  readonly rightTempleStart: THREE.Vector3 | null
  readonly nativeBounds: THREE.Box3
  readonly nativeCenter: THREE.Vector3
  readonly normalizationTransform: THREE.Matrix4
  readonly parts: EyewearPartDetection
  readonly orientation: NativeOrientation
}

/** Complete analysis bundle produced by GLBAssetAnalyzer */
export interface GLBAssetAnalysis {
  readonly summary: import('./GLBAssetAnalyzer').AssetGeometrySummary
  readonly orientation: NativeOrientation
  readonly parts: EyewearPartDetection
  readonly profile: EyewearAssetProfile
  readonly normalization: import('./CanonicalNormalizer').NormalizationResult
}

/** Reference geometry derived from facial landmarks */
export interface FaceReference {
  readonly leftEyeCenter: THREE.Vector3
  readonly rightEyeCenter: THREE.Vector3
  readonly eyeSeparation: number
  readonly faceCenter: THREE.Vector3
  readonly noseBridgeAnchor: THREE.Vector3
  readonly faceNormal: THREE.Vector3
  readonly faceUp: THREE.Vector3
}

/** Multi-factor confidence score breakdown */
export interface ConfidenceBreakdown {
  readonly semantic: number
  readonly lensDetection: number
  readonly bridgeDetection: number
  readonly orientation: number
  readonly scaleEstimation: number
  readonly overall: number
  readonly tier: 'HIGH' | 'MEDIUM' | 'LOW'
}

/** Complete result produced by AutoCalibrationEngine */
export interface AutoCalibrationResult {
  readonly autoCalibration: Calibration
  readonly manualCorrection: Calibration
  readonly finalCalibration: Calibration
  readonly confidence: ConfidenceBreakdown
  readonly profile: EyewearAssetProfile
  readonly normalization: import('./CanonicalNormalizer').NormalizationResult
  readonly faceReference: FaceReference | null
  readonly isAutoApplied: boolean
  readonly warningMessages: readonly string[]
}
