import { useEffect, useMemo, useState } from 'react'
import type { Calibration } from '../types/Calibration'

export interface CalibrationPanelProps {
  readonly calibration: Calibration
  readonly onChange: (calibration: Calibration) => void
  readonly onReset: () => void
  readonly onExport: () => void
}

type CalibrationKey = keyof Calibration

type SliderSpec = {
  readonly key: CalibrationKey
  readonly label: string
  readonly min: number
  readonly max: number
  readonly step: number
  readonly unit: string
  readonly help: string
}

const ROTATION_LIMIT = 20 * Math.PI / 180
const SCALE_MIN = 0.1
const SCALE_MAX = 12
const TRANSLATION_LIMIT = 7

const SLIDER_SPECS: readonly SliderSpec[] = [
  {
    key: 'scale',
    label: 'Scale',
    min: SCALE_MIN,
    max: SCALE_MAX,
    step: 0.001,
    unit: '×',
    help: 'Overall eyewear size.',
  },
  {
    key: 'x',
    label: 'X offset',
    min: -TRANSLATION_LIMIT,
    max: TRANSLATION_LIMIT,
    step: 0.01,
    unit: 'cm',
    help: 'Move the frame left/right relative to the face.',
  },
  {
    key: 'y',
    label: 'Y offset',
    min: -TRANSLATION_LIMIT,
    max: TRANSLATION_LIMIT,
    step: 0.01,
    unit: 'cm',
    help: 'Move the frame down/up relative to the face.',
  },
  {
    key: 'z',
    label: 'Z offset',
    min: -TRANSLATION_LIMIT,
    max: TRANSLATION_LIMIT,
    step: 0.01,
    unit: 'cm',
    help: 'Move the frame toward/away from the face.',
  },
  {
    key: 'rotationX',
    label: 'Rotation X',
    min: -ROTATION_LIMIT,
    max: ROTATION_LIMIT,
    step: 0.001,
    unit: 'rad',
    help: 'Pitch correction.',
  },
  {
    key: 'rotationY',
    label: 'Rotation Y',
    min: -ROTATION_LIMIT,
    max: ROTATION_LIMIT,
    step: 0.001,
    unit: 'rad',
    help: 'Yaw correction.',
  },
  {
    key: 'rotationZ',
    label: 'Rotation Z',
    min: -ROTATION_LIMIT,
    max: ROTATION_LIMIT,
    step: 0.001,
    unit: 'rad',
    help: 'Roll correction.',
  },
]

const SLIDER_MIN = 0
const SLIDER_MAX = 100

export function identityCalibration(): Calibration {
  return {
    scale: 1,
    x: 0,
    y: 0,
    z: 0,
    rotationX: 0,
    rotationY: 0,
    rotationZ: 0,
  }
}

export function calibrationToSliderValues(
  calibration: Calibration,
): Partial<Record<CalibrationKey, number>> {
  return Object.fromEntries(
    SLIDER_SPECS.map((spec) => [spec.key, toSliderValue(calibration[spec.key], spec)]),
  ) as Partial<Record<CalibrationKey, number>>
}

function toSliderValue(value: number, spec: SliderSpec): number {
  const normalized = spec.key === 'scale'
    ? Math.log(Math.max(spec.min, value) / spec.min) / Math.log(spec.max / spec.min)
    : (value - spec.min) / (spec.max - spec.min)

  return Math.round(Math.min(1, Math.max(0, normalized)) * SLIDER_MAX)
}

function fromSliderValue(sliderValue: number, spec: SliderSpec): number {
  const normalized = sliderValue / SLIDER_MAX
  const raw = spec.key === 'scale'
    ? spec.min * (spec.max / spec.min) ** normalized
    : spec.min + normalized * (spec.max - spec.min)
  const precision = spec.key === 'scale'
    ? 4
    : spec.step >= 1 ? 0 : Math.max(0, Math.ceil(-Math.log10(spec.step)))
  const factor = 10 ** precision
  return Math.round(raw * factor) / factor
}

function formatValue(value: number, unit: string): string {
  if (unit === '×') return `${value.toFixed(3)}×`
  if (unit === 'cm') return `${value.toFixed(2)} cm`
  return `${value.toFixed(3)} rad`
}

export function CalibrationPanel({
  calibration,
  onChange,
  onReset,
  onExport,
}: CalibrationPanelProps) {
  const [activeKey, setActiveKey] = useState<CalibrationKey | null>(null)
  const specs = useMemo(() => SLIDER_SPECS, [])

  useEffect(() => {
    if (activeKey && !specs.some((spec) => spec.key === activeKey)) {
      setActiveKey(null)
    }
  }, [activeKey, specs])

  const updateValue = (spec: SliderSpec, sliderValue: number) => {
    setActiveKey(spec.key)
    onChange({
      ...calibration,
      [spec.key]: fromSliderValue(sliderValue, spec),
    })
  }

  return (
    <section className="calibration-panel" aria-label="Eyewear calibration controls">
      <div className="calibration-panel__header">
        <div>
          <p className="eyebrow">Calibration</p>
          <h2>Eyewear calibration</h2>
          <p className="calibration-panel__description">
            Use 0–100 sliders to position, size, and rotate the frame. Scale uses a wider range because the supplied GLB assets use different native units.
          </p>
        </div>
        <span className="calibration-panel__active" aria-live="polite">
          {activeKey ? `${activeKey} adjusted` : 'Live'}
        </span>
      </div>

      <div className="calibration-panel__controls">
        {specs.map((spec) => {
          const value = calibration[spec.key]
          const sliderValue = toSliderValue(value, spec)
          const inputId = `calibration-${spec.key}`

          return (
            <div className="calibration-control" key={spec.key}>
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
                onChange={(event) => updateValue(spec, Number(event.target.value))}
                aria-valuetext={`${formatValue(value, spec.unit)}; slider ${sliderValue} of 100`}
              />
              <div className="calibration-control__meta">
                <span>0</span>
                <span>{spec.help}</span>
                <span>100</span>
              </div>
            </div>
          )
        })}
      </div>

      <div className="calibration-panel__actions">
        <button type="button" className="calibration-button calibration-button--secondary" onClick={onReset}>
          Reset
        </button>
        <button type="button" className="calibration-button" onClick={onExport}>
          Export Calibration JSON
        </button>
      </div>
    </section>
  )
}
