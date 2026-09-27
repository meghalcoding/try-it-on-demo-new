import { useState } from 'react'
import {
  FACE_OCCLUSION_RANGES,
  type FaceOcclusionSettings,
  type NumericSettingKey,
} from '../types/FaceOcclusion'
import type { OcclusionStatus } from '../ar/ARRenderer'
import type { SegmenterStatus } from '../ar/occlusion/ForegroundSegmenter'

/** Live values shown in the panel; produced by App from the renderer + segmenter. */
export interface OcclusionStatusView {
  readonly occlusion: OcclusionStatus
  readonly segmenter: SegmenterStatus
  readonly segmenterMessage: string
  readonly segmenterMs: number
}

export interface FaceOcclusionPanelProps {
  readonly settings: FaceOcclusionSettings
  readonly status: OcclusionStatusView | null
  readonly onChange: (settings: FaceOcclusionSettings) => void
  readonly onReset: () => void
  readonly onExport: () => void
  readonly onBakeClearance: (clearanceCm: number) => void
}

type SliderSpec = {
  readonly key: NumericSettingKey
  readonly label: string
  readonly step: number
  readonly unit: 'cm' | '×' | 's' | '' | '°'
  readonly help: string
  /** Narrower range than the validation range, for a usable slider. */
  readonly uiMin?: number
  readonly uiMax?: number
}

type BooleanKey = {
  [K in keyof FaceOcclusionSettings]: FaceOcclusionSettings[K] extends boolean ? K : never
}[keyof FaceOcclusionSettings]

const SURFACE_SLIDERS: readonly SliderSpec[] = [
  { key: 'landmarkDepthBlend', label: 'Personalisation', step: 0.01, unit: '', help: '0 = rigid average face; 1 = fully from landmarks. Landmark noise grows toward 1.' },
  { key: 'maxLandmarkDeviationCm', label: 'Max deviation from average', step: 0.05, unit: 'cm', help: 'Caps how far landmarks may pull any vertex; bounds the effect of bad depth.' },
  { key: 'faceSurfaceBiasCm', label: 'Nose/brow clearance', step: 0.01, unit: 'cm', help: 'Kept tight so a badly fit frame is caught, not hidden.' },
  { key: 'templeClearanceCm', label: 'Cheek/temple clearance', step: 0.05, unit: 'cm', help: 'Kept generous: this is where arms run and reconstruction is noisiest. Raise this if arms cut off before the ear.' },
  { key: 'landmarkSmoothingHalfLifeSeconds', label: 'Surface smoothing', step: 0.005, unit: 's', uiMin: 0.01, uiMax: 0.15, help: 'Applied in face space, so it does not lag behind head motion.' },
]

const LEGACY_SLIDERS: readonly SliderSpec[] = [
  { key: 'referenceFaceWidthCm', label: 'Reference face width', step: 0.1, unit: 'cm', uiMin: 8, uiMax: 24, help: 'Metric scale used to convert landmark depth (legacy).' },
  { key: 'depthScale', label: 'Depth influence', step: 0.01, unit: '×', uiMin: 0.25, uiMax: 2, help: 'Exaggerates/flattens landmark depth (legacy).' },
  { key: 'depthBias', label: 'Occlusion depth bias', step: 0.01, unit: 'cm', uiMin: -0.5, uiMax: 0.5, help: 'Positive moves the surface toward the camera (legacy).' },
  { key: 'surfaceScale', label: 'Face surface coverage', step: 0.001, unit: '×', uiMin: 0.9, uiMax: 1.15, help: 'Expands the projected surface about the view centre (legacy).' },
  { key: 'landmarkSmoothingHalfLifeSeconds', label: 'Occlusion smoothing', step: 0.005, unit: 's', uiMin: 0.01, uiMax: 0.15, help: 'Lower responds faster; higher removes jitter (legacy).' },
]

const HEAD_SLIDERS: readonly SliderSpec[] = [
  { key: 'headProxyPushCm', label: 'Head push-back', step: 0.05, unit: 'cm', help: 'Larger keeps straight temple arms visible along the skin; smaller hides them sooner.' },
  { key: 'earProxyPushCm', label: 'Ear push-back', step: 0.05, unit: 'cm', help: 'Same, for the ear volumes.' },
]

const FIT_SLIDERS: readonly SliderSpec[] = [
  { key: 'maxClearanceCm', label: 'Max automatic lift', step: 0.1, unit: 'cm', help: 'Upper bound on the forward lift applied to un-embed the frame.' },
]

const MASK_SLIDERS: readonly SliderSpec[] = [
  { key: 'hairOcclusionStrength', label: 'Hair over frame', step: 0.01, unit: '', help: 'How strongly hair in front of the face hides the glasses.' },
  { key: 'hairReach', label: 'Hair reach', step: 0.01, unit: '', help: '0 = only inside the face outline (bangs); 1 = also the ring around it (temples).' },
  { key: 'handOcclusionStrength', label: 'Hands / fingers', step: 0.01, unit: '', help: 'How strongly a hand in front hides the glasses.' },
  { key: 'objectOcclusionStrength', label: 'Objects / clothing', step: 0.01, unit: '', help: 'Phones, sleeves and other things passing in front.' },
]

const SHADOW_SLIDERS: readonly SliderSpec[] = [
  { key: 'contactShadowOpacity', label: 'Contact shadow strength', step: 0.01, unit: '', help: 'Soft shadow of the frame on the nose and cheeks.' },
]

const CAMERA_SLIDERS: readonly SliderSpec[] = [
  { key: 'trackerVerticalFovDeg', label: 'Tracker vertical FOV', step: 0.5, unit: '°', help: 'MediaPipe Face Geometry assumes 63°. Only used when camera matching is on.' },
]

const SLIDER_MIN = 0
const SLIDER_MAX = 100

function bounds(spec: SliderSpec): { min: number; max: number } {
  const range = FACE_OCCLUSION_RANGES[spec.key]
  return { min: spec.uiMin ?? range.min, max: spec.uiMax ?? range.max }
}

const ALL_SLIDERS = [
  ...SURFACE_SLIDERS, ...LEGACY_SLIDERS, ...HEAD_SLIDERS, ...FIT_SLIDERS,
  ...MASK_SLIDERS, ...SHADOW_SLIDERS, ...CAMERA_SLIDERS,
]

/** Normalised 0..100 values used by the calibration diagnostics log. */
export function faceOcclusionToSliderValues(
  settings: FaceOcclusionSettings,
): Partial<Record<NumericSettingKey, number>> {
  const out: Partial<Record<NumericSettingKey, number>> = {}
  for (const spec of ALL_SLIDERS) {
    const { min, max } = bounds(spec)
    const normalized = (settings[spec.key] - min) / (max - min)
    out[spec.key] = Math.round(Math.min(1, Math.max(0, normalized)) * SLIDER_MAX)
  }
  return out
}

function fromSliderValue(sliderValue: number, spec: SliderSpec): number {
  const { min, max } = bounds(spec)
  const raw = min + (sliderValue / SLIDER_MAX) * (max - min)
  const precision = Math.max(0, Math.ceil(-Math.log10(spec.step)))
  const factor = 10 ** precision
  return Math.round(raw * factor) / factor
}

function formatValue(value: number, unit: SliderSpec['unit']): string {
  if (unit === '×') return `${value.toFixed(3)}×`
  if (unit === 'cm') return `${value.toFixed(2)} cm`
  if (unit === 's') return `${value.toFixed(3)} s`
  if (unit === '°') return `${value.toFixed(1)}°`
  return value.toFixed(2)
}

function verdictText(status: OcclusionStatus): string {
  const { fit, appliedClearanceCm } = status
  if (fit.verdict === 'unknown') return 'not measured yet'
  const pct = Math.round(fit.embeddedFraction * 100)
  if (fit.verdict === 'ok') return `frame clear of the face (${pct}% embedded)`
  return `${pct}% of the frame is inside the face (deepest ${fit.maxPenetrationCm.toFixed(1)} cm); lifted ${appliedClearanceCm.toFixed(2)} cm`
}

export function FaceOcclusionPanel({
  settings,
  status,
  onChange,
  onReset,
  onExport,
  onBakeClearance,
}: FaceOcclusionPanelProps) {
  const [activeKey, setActiveKey] = useState<string | null>(null)
  const legacy = settings.mode === 'legacy'

  const setSlider = (spec: SliderSpec, sliderValue: number) => {
    setActiveKey(spec.key)
    onChange({ ...settings, [spec.key]: fromSliderValue(sliderValue, spec) })
  }

  const renderSlider = (spec: SliderSpec) => {
    const { min, max } = bounds(spec)
    const value = settings[spec.key]
    const sliderValue = Math.round(Math.min(1, Math.max(0, (value - min) / (max - min))) * SLIDER_MAX)
    const inputId = `occlusion-${legacy ? 'legacy-' : ''}${spec.key}`

    return (
      <div className="calibration-control" key={inputId}>
        <div className="calibration-control__header">
          <label htmlFor={inputId}>{spec.label}</label>
          <output htmlFor={inputId}>{formatValue(value, spec.unit)}</output>
        </div>
        <input
          id={inputId}
          type="range"
          min={SLIDER_MIN}
          max={SLIDER_MAX}
          step={1}
          value={sliderValue}
          onChange={(event) => setSlider(spec, Number(event.target.value))}
          aria-valuetext={`${formatValue(value, spec.unit)}; slider ${sliderValue} of 100`}
        />
        <div className="calibration-control__meta">
          <span>{formatValue(min, spec.unit)}</span>
          <span>{spec.help}</span>
          <span>{formatValue(max, spec.unit)}</span>
        </div>
      </div>
    )
  }

  const renderToggle = (key: BooleanKey, title: string, hint: string) => (
    <label className="face-occlusion-toggle" key={key}>
      <input
        type="checkbox"
        checked={settings[key]}
        onChange={(event) => onChange({ ...settings, [key]: event.target.checked })}
      />
      <span>
        <strong>{title}</strong>
        <small>{hint}</small>
      </span>
    </label>
  )

  const occ = status?.occlusion
  const fit = occ?.fit
  const maskText = !settings.foregroundMaskEnabled
    ? 'off'
    : status?.segmenter === 'ready'
      ? occ?.foregroundMaskActive
        ? `active (${Math.round(status.segmenterMs)} ms / run)`
        : 'ready, waiting for a face'
      : status?.segmenter === 'loading'
        ? 'loading model…'
        : status?.segmenter === 'unavailable'
          ? `unavailable: ${status.segmenterMessage}`
          : 'starting…'

  return (
    <section className="smoothing-panel face-occlusion-panel" aria-label="Face occlusion controls">
      <div className="smoothing-panel__header">
        <div>
          <p className="eyebrow">Depth</p>
          <h2>Face occlusion</h2>
          <p className="smoothing-panel__description">
            Head, ear and face volumes hide temples behind the head; a segmentation mask puts hair
            and hands in front of the frame; contact shadows sit the frame on the nose.
          </p>
        </div>
        <span className="calibration-panel__active" aria-live="polite">
          {activeKey ? `${activeKey} adjusted` : settings.enabled ? (legacy ? 'Legacy' : 'Active') : 'Off'}
        </span>
      </div>

      {renderToggle('enabled', 'Enable face occlusion', 'Disable this to compare the raw GLB rendering against the occlusion passes.')}

      <div className="occlusion-mode" role="group" aria-label="Occluder version">
        <button
          type="button"
          className={`calibration-button ${legacy ? 'calibration-button--secondary' : ''}`}
          aria-pressed={!legacy}
          onClick={() => onChange({ ...settings, mode: 'hardened' })}
        >
          Hardened
        </button>
        <button
          type="button"
          className={`calibration-button ${legacy ? '' : 'calibration-button--secondary'}`}
          aria-pressed={legacy}
          onClick={() => onChange({ ...settings, mode: 'legacy' })}
        >
          Legacy (A/B)
        </button>
      </div>

      {status && occ && (
        <dl className="occlusion-status" aria-label="Occlusion status" aria-live="off">
          <div><dt>Face surface</dt><dd>{occ.occluder.hasSurface ? `tracking, agreement ${Math.round(occ.occluder.confidence * 100)}%` : 'waiting for a face'}</dd></div>
          {!legacy && <div><dt>Frame fit</dt><dd>{verdictText(occ)}</dd></div>}
          <div><dt>Hair / hands</dt><dd>{maskText}</dd></div>
          <div><dt>Camera FOV</dt><dd>{occ.cameraFovDeg.toFixed(1)}°{settings.matchTrackerCamera ? ' (tracker-matched)' : ' (original)'}</dd></div>
        </dl>
      )}

      {legacy ? (
        <div className="calibration-panel__controls">{LEGACY_SLIDERS.map(renderSlider)}</div>
      ) : (
        <>
          <h3 className="occlusion-section">Face surface</h3>
          <div className="calibration-panel__controls">{SURFACE_SLIDERS.map(renderSlider)}</div>

          <h3 className="occlusion-section">Head and ears</h3>
          {renderToggle('headProxyEnabled', 'Head volume', 'Hides the far temple arm behind a turned head.')}
          {renderToggle('earProxyEnabled', 'Ear volumes', 'Temples disappear behind the ear.')}
          <div className="calibration-panel__controls">{HEAD_SLIDERS.map(renderSlider)}</div>

          <h3 className="occlusion-section">Frame fit</h3>
          {renderToggle('anatomicalClearanceEnabled', 'Rest frame on the face', 'Lifts the frame forward until it is not embedded in the nose/brow.')}
          <div className="calibration-panel__controls">{FIT_SLIDERS.map(renderSlider)}</div>
          {fit && occ && occ.appliedClearanceCm > 0.05 && (
            <button
              type="button"
              className="calibration-button calibration-button--secondary"
              onClick={() => onBakeClearance(occ.appliedClearanceCm)}
            >
              Bake +{occ.appliedClearanceCm.toFixed(2)} cm into this product’s Z
            </button>
          )}

          <h3 className="occlusion-section">Hair, hands and objects</h3>
          {renderToggle('foregroundMaskEnabled', 'Foreground mask', 'Downloads a ~16 MB segmentation model on first use.')}
          <div className="calibration-panel__controls">{MASK_SLIDERS.map(renderSlider)}</div>
          {renderToggle('debugShowMask', 'Show mask overlay', 'Tints what will hide the glasses, exactly as applied.')}

          <h3 className="occlusion-section">Shadows and light</h3>
          {renderToggle('contactShadowsEnabled', 'Contact shadows', 'Adds a shadow pass; disable on slow devices.')}
          <div className="calibration-panel__controls">{SHADOW_SLIDERS.map(renderSlider)}</div>
          {renderToggle('lightingMatchEnabled', 'Match face lighting (heuristic)', 'Dims/steers the glasses’ lighting from the measured face brightness.')}

          <h3 className="occlusion-section">Camera model</h3>
          {renderToggle('matchTrackerCamera', 'Match tracker camera', 'Renderer FOV follows the tracker’s virtual camera. Changes on-screen glasses size by a few %.')}
          <div className="calibration-panel__controls">{CAMERA_SLIDERS.map(renderSlider)}</div>

          <h3 className="occlusion-section">Diagnostics</h3>
          {renderToggle('debugShowOccluders', 'Show occluder volumes', 'Outlines the face, head and ear depth volumes over the video.')}
        </>
      )}

      <div className="calibration-panel__actions">
        <button type="button" className="calibration-button calibration-button--secondary" onClick={onReset}>
          Reset
        </button>
        <button type="button" className="calibration-button" onClick={onExport}>
          Export Occlusion JSON
        </button>
      </div>
    </section>
  )
}
