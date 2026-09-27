import type { AutoCalibrationResult } from '../ar/glb/types'

export interface GLBDiagnosticPanelProps {
  readonly result: AutoCalibrationResult | null
  readonly modelUrl?: string | null
  readonly onApplyAutoCalibration?: () => void
  readonly onResetManualCorrection?: () => void
}

export function GLBDiagnosticPanel({
  result,
  modelUrl,
  onApplyAutoCalibration,
  onResetManualCorrection,
}: GLBDiagnosticPanelProps) {
  if (!result) {
    return (
      <section className="calibration-panel calibration-panel--diagnostic" aria-label="GLB Analysis Diagnostics">
        <div className="calibration-panel__header">
          <h3>GLB Geometry Inspector</h3>
          <span className="badge badge--neutral">No Model Loaded</span>
        </div>
        <p className="calibration-panel__description">
          Select or upload an eyewear GLB model to inspect native coordinate geometry, PCA orientation, detected parts, and auto calibration.
        </p>
      </section>
    )
  }

  const { confidence, profile, autoCalibration, finalCalibration, warningMessages } = result
  const fileName = modelUrl ? modelUrl.split('/').pop() || modelUrl : 'Active GLB'
  const confidencePercent = Math.round(confidence.overall * 100)

  const confidenceBadgeClass =
    confidence.tier === 'HIGH'
      ? 'badge--success'
      : confidence.tier === 'MEDIUM'
        ? 'badge--warning'
        : 'badge--danger'

  return (
    <section className="calibration-panel calibration-panel--diagnostic" aria-label="GLB Analysis Diagnostics">
      <div className="calibration-panel__header">
        <div>
          <p className="eyebrow">Developer Diagnostics</p>
          <h3>GLB Geometry Inspector</h3>
        </div>
        <span className={`badge ${confidenceBadgeClass}`}>
          {confidencePercent}% {confidence.tier} Confidence
        </span>
      </div>

      <div className="diagnostic-summary">
        <p className="diagnostic-file">
          <strong>File:</strong> {fileName}
        </p>

        <div className="diagnostic-grid">
          <div className="diagnostic-card">
            <h4>Native Dimensions</h4>
            <ul>
              <li><strong>Width (X):</strong> {profile.canonicalWidth.toFixed(1)}</li>
              <li><strong>Height (Y):</strong> {profile.canonicalHeight.toFixed(1)}</li>
              <li><strong>Depth (Z):</strong> {profile.canonicalDepth.toFixed(1)}</li>
            </ul>
          </div>

          <div className="diagnostic-card">
            <h4>Native Orientation</h4>
            <ul>
              <li><strong>Width Axis:</strong> {profile.orientation.widthAxis}</li>
              <li><strong>Height Axis:</strong> {profile.orientation.heightAxis}</li>
              <li><strong>Depth Axis:</strong> {profile.orientation.depthAxis}</li>
            </ul>
          </div>

          <div className="diagnostic-card">
            <h4>Detected Parts</h4>
            <ul>
              <li>
                {profile.parts.leftLensCenter ? '✓' : '✗'} Left Lens
              </li>
              <li>
                {profile.parts.rightLensCenter ? '✓' : '✗'} Right Lens
              </li>
              <li>
                {profile.parts.bridgeCenter ? '✓' : '✗'} Bridge
              </li>
              <li>
                {profile.parts.detectedTemples.length > 0 ? '✓' : '—'} Temples ({profile.parts.detectedTemples.length})
              </li>
            </ul>
          </div>

          <div className="diagnostic-card">
            <h4>Measurements</h4>
            <ul>
              <li>
                <strong>Lens Separation:</strong>{' '}
                {profile.lensSeparation ? profile.lensSeparation.toFixed(1) : 'Not Detected'}
              </li>
              <li>
                <strong>Anchor Origin:</strong> {profile.parts.anchorOriginSource.replace(/_/g, ' ')}
              </li>
            </ul>
          </div>
        </div>

        <div className="diagnostic-calibration-values">
          <h4>Derived Auto Calibration</h4>
          <pre>
            Scale: {autoCalibration.scale.toFixed(4)}
            {'\n'}X: {autoCalibration.x.toFixed(3)} | Y: {autoCalibration.y.toFixed(3)} | Z: {autoCalibration.z.toFixed(3)}
            {'\n'}RotX: {autoCalibration.rotationX.toFixed(3)} | RotY: {autoCalibration.rotationY.toFixed(3)} | RotZ: {autoCalibration.rotationZ.toFixed(3)}
          </pre>
        </div>

        <div className="diagnostic-calibration-values">
          <h4>Final Composed Calibration (Auto + Manual Overlay)</h4>
          <pre>
            Scale: {finalCalibration.scale.toFixed(4)}
            {'\n'}X: {finalCalibration.x.toFixed(3)} | Y: {finalCalibration.y.toFixed(3)} | Z: {finalCalibration.z.toFixed(3)}
            {'\n'}RotX: {finalCalibration.rotationX.toFixed(3)} | RotY: {finalCalibration.rotationY.toFixed(3)} | RotZ: {finalCalibration.rotationZ.toFixed(3)}
          </pre>
        </div>

        {warningMessages.length > 0 && (
          <div className="diagnostic-warnings">
            {warningMessages.map((msg, idx) => (
              <p key={idx} className="diagnostic-warning-text">⚠️ {msg}</p>
            ))}
          </div>
        )}

        <div className="calibration-panel__actions">
          {onApplyAutoCalibration && (
            <button
              type="button"
              className="camera-permission__button camera-permission__button--secondary"
              onClick={onApplyAutoCalibration}
            >
              Re-Auto Calibrate
            </button>
          )}
          {onResetManualCorrection && (
            <button
              type="button"
              className="camera-permission__button camera-permission__button--secondary"
              onClick={onResetManualCorrection}
            >
              Reset Manual Overlay
            </button>
          )}
        </div>
      </div>
    </section>
  )
}
