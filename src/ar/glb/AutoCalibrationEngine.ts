import * as THREE from 'three'
import type { Calibration } from '../../types/Calibration'
import type {
  AutoCalibrationResult,
  ConfidenceBreakdown,
  FaceReference,
  GLBAssetAnalysis,
} from './types'
import { FaceReferenceGeometryExtractor } from './FaceReferenceGeometry'

export const IDENTITY_CALIBRATION_OVERLAY: Calibration = Object.freeze({
  scale: 1.0,
  x: 0,
  y: 0,
  z: 0,
  rotationX: 0,
  rotationY: 0,
  rotationZ: 0,
})

/**
 * AutoCalibrationEngine calculates geometry-derived automatic scale, translation offsets,
 * and rotation angles, composes them with manual correction overlays, and computes multi-factor confidence scores.
 */
export class AutoCalibrationEngine {
  private readonly faceExtractor = new FaceReferenceGeometryExtractor()

  /**
   * Computes automatic calibration, confidence breakdown, and manual correction composition.
   */
  public computeAutoCalibration(
    analysis: GLBAssetAnalysis,
    faceRef?: FaceReference | null,
    manualCorrectionOverlay?: Partial<Calibration>,
  ): AutoCalibrationResult {
    const face = faceRef || this.faceExtractor.createCanonicalDefaultFaceReference()
    const { profile, parts, orientation } = analysis

    const manualCorrection: Calibration = {
      scale: manualCorrectionOverlay?.scale ?? 1.0,
      x: manualCorrectionOverlay?.x ?? 0,
      y: manualCorrectionOverlay?.y ?? 0,
      z: manualCorrectionOverlay?.z ?? 0,
      rotationX: manualCorrectionOverlay?.rotationX ?? 0,
      rotationY: manualCorrectionOverlay?.rotationY ?? 0,
      rotationZ: manualCorrectionOverlay?.rotationZ ?? 0,
    }

    const warningMessages: string[] = []

    // --- 1. Auto Scale Derivation (AUTO-8) ---
    let autoScale = 1.0
    let scaleConfidence = 0.2

    const scaleFromLens = profile.lensSeparation && profile.lensSeparation > 0.01
      ? face.eyeSeparation / profile.lensSeparation
      : null

    const TARGET_FRAME_WIDTH_CM = 13.8
    const scaleFromWidth = profile.canonicalWidth > 0.01
      ? TARGET_FRAME_WIDTH_CM / profile.canonicalWidth
      : null

    if (scaleFromLens !== null && scaleFromWidth !== null) {
      // Balanced fusion: lens IPD match (60%) + overall frame width fit (40%)
      autoScale = scaleFromLens * 0.6 + scaleFromWidth * 0.4
      scaleConfidence = 0.95
    } else if (scaleFromLens !== null) {
      autoScale = scaleFromLens
      scaleConfidence = 0.90
    } else if (scaleFromWidth !== null) {
      autoScale = scaleFromWidth
      scaleConfidence = 0.70
      warningMessages.push('Lens separation could not be detected; using frame width scaling fallback.')
    } else {
      warningMessages.push('Could not determine GLB dimensions; falling back to unit scale.')
    }

    // Sanity check scale bounds
    autoScale = Math.max(0.01, Math.min(autoScale, 100.0))

    // --- 2. Auto Translation & Alignment Derivation (AUTO-9) ---
    // CanonicalNormalizer anchors the eyewear runtime root at the lens midpoint.
    // The correct face reference for that origin is therefore the eye midpoint,
    // not the lower nose bridge. Using the nose bridge for X/Y systematically placed
    // every frame below the eye line in the live test.
    //
    // Keep Z tied to the nose-bridge depth reference because the glasses must sit
    // slightly in front of the eye/face plane, while X/Y are anchored to the eye line.
    const autoX = face.faceCenter.x
    const autoY = face.faceCenter.y
    const autoZ = face.noseBridgeAnchor.z

    // --- 3. Auto Rotation Derivation (AUTO-9) ---
    // Pantoscopic tilt: slight forward tilt (~3-5 degrees = ~0.07 rad) standard in eyewear ergonomics
    const PANTOSCOPIC_TILT_RAD = 0.07
    let autoRotX = PANTOSCOPIC_TILT_RAD
    let autoRotY = 0
    let autoRotZ = 0

    const autoCalibration: Calibration = {
      scale: Number(autoScale.toFixed(4)),
      x: Number(autoX.toFixed(3)),
      y: Number(autoY.toFixed(3)),
      z: Number(autoZ.toFixed(3)),
      rotationX: Number(autoRotX.toFixed(3)),
      rotationY: Number(autoRotY.toFixed(3)),
      rotationZ: Number(autoRotZ.toFixed(3)),
    }

    // --- 4. Runtime Calibration Composition (AUTO-10) ---
    const finalCalibration: Calibration = {
      scale: autoCalibration.scale * manualCorrection.scale,
      x: autoCalibration.x + manualCorrection.x,
      y: autoCalibration.y + manualCorrection.y,
      z: autoCalibration.z + manualCorrection.z,
      rotationX: autoCalibration.rotationX + manualCorrection.rotationX,
      rotationY: autoCalibration.rotationY + manualCorrection.rotationY,
      rotationZ: autoCalibration.rotationZ + manualCorrection.rotationZ,
    }

    // --- 5. Multi-Factor Confidence Scoring (AUTO-11) ---
    const semanticConfidence = evaluateSemanticConfidence(parts)
    const lensConfidence = parts.leftLensCenter && parts.rightLensCenter ? 0.95 : parts.detectedLenses.length > 0 ? 0.5 : 0.2
    const bridgeConfidence = parts.detectedBridge ? parts.detectedBridge.evidence.confidence : parts.bridgeCenter ? 0.7 : 0.3
    const orientationConfidence = orientation.pcaEigenvalues.length === 3 && orientation.pcaEigenvalues[0] > 0 ? 0.9 : 0.5

    const overallScore =
      0.25 * semanticConfidence +
      0.35 * lensConfidence +
      0.15 * bridgeConfidence +
      0.15 * orientationConfidence +
      0.1 * scaleConfidence

    const tier: ConfidenceBreakdown['tier'] = overallScore >= 0.8 ? 'HIGH' : overallScore >= 0.55 ? 'MEDIUM' : 'LOW'

    if (tier === 'LOW') {
      warningMessages.push('Low geometry detection confidence; developer manual calibration recommended.')
    }

    const confidence: ConfidenceBreakdown = {
      semantic: Number(semanticConfidence.toFixed(2)),
      lensDetection: Number(lensConfidence.toFixed(2)),
      bridgeDetection: Number(bridgeConfidence.toFixed(2)),
      orientation: Number(orientationConfidence.toFixed(2)),
      scaleEstimation: Number(scaleConfidence.toFixed(2)),
      overall: Number(overallScore.toFixed(2)),
      tier,
    }

    return {
      autoCalibration,
      manualCorrection,
      finalCalibration,
      confidence,
      profile,
      normalization: analysis.normalization,
      faceReference: face,
      isAutoApplied: tier !== 'LOW',
      warningMessages: Object.freeze(warningMessages),
    }
  }
}

function evaluateSemanticConfidence(parts: GLBAssetAnalysis['parts']): number {
  let count = 0
  let totalScore = 0

  for (const lens of parts.detectedLenses) {
    count += 1
    totalScore += lens.evidence.source === 'semantic' || lens.evidence.source === 'hybrid' ? 0.9 : 0.4
  }

  if (parts.detectedBridge) {
    count += 1
    totalScore += parts.detectedBridge.evidence.source === 'semantic' ? 0.9 : 0.5
  }

  return count > 0 ? totalScore / count : 0.3
}
