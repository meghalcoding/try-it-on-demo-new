import type { PoseSmoothingSettings } from '../ar/PoseSmoother'
import type { Calibration } from '../types/Calibration'
import type { FaceOcclusionSettings } from '../types/FaceOcclusion'

export interface CalibrationDiagnosticContext {
  readonly source: 'init' | 'change' | 'reset' | 'export'
  readonly sliderValues?: Readonly<
    Partial<Record<keyof Calibration, number> & Partial<Record<keyof PoseSmoothingSettings, number>>>
  >
  readonly smoothing?: PoseSmoothingSettings
  readonly occlusion?: FaceOcclusionSettings
}

/**
 * Browser-console diagnostics intentionally use a stable, PowerShell-friendly
 * prefix and JSON payload. No server or backend is involved; calibration, smoothing, and occlusion data stay inside the browser.
 */
export function logCalibrationDiagnostics(
  calibration: Calibration,
  context: CalibrationDiagnosticContext,
): void {
  const payload = {
    source: context.source,
    calibration: {
      scale: calibration.scale,
      x: calibration.x,
      y: calibration.y,
      z: calibration.z,
      rotationX: calibration.rotationX,
      rotationY: calibration.rotationY,
      rotationZ: calibration.rotationZ,
    },
    smoothing: context.smoothing ?? null,
    occlusion: context.occlusion ?? null,
    slider: context.sliderValues ?? null,
  }

  console.info(`[AR-CALIBRATION] ${JSON.stringify(payload)}`)
}
