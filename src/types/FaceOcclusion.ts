/**
 * Runtime controls for the occlusion system.
 *
 * Everything here tunes what is *hidden or shaded*; nothing in this file
 * changes the tracked face pose, the eyewear calibration, or the product
 * catalogue.
 *
 * The first block of keys is the original (v1) schema and is preserved so that
 * previously exported `face-occlusion.json` files still load. The legacy-only
 * keys are used only when `mode === 'legacy'`, which keeps the previous
 * flat-face occluder available for A/B comparison during live validation.
 */

export type FaceOcclusionMode = 'hardened' | 'legacy'

export interface FaceOcclusionSettings {
  // ---- v1 schema (kept) --------------------------------------------------
  readonly enabled: boolean
  /** LEGACY only. Approximate canonical face width used to convert landmark depth to cm. */
  readonly referenceFaceWidthCm: number
  /** LEGACY only. Multiplies the face landmark depth variation. */
  readonly depthScale: number
  /** LEGACY only. Moves the occlusion surface toward (+) / away from (-) the camera, cm. */
  readonly depthBias: number
  /** LEGACY only. Expands/contracts the projected surface around the viewport centre. */
  readonly surfaceScale: number
  /** Half-life used to smooth the face surface (face-local in hardened mode), seconds. */
  readonly landmarkSmoothingHalfLifeSeconds: number

  // ---- v2: mode -----------------------------------------------------------
  readonly mode: FaceOcclusionMode

  // ---- v2: face surface (Rules 1 & 4: depth, nose contour) ---------------
  /** 0 = rigid canonical face under the pose, 1 = fully landmark-derived relief. */
  readonly landmarkDepthBlend: number
  /** Cap on how far landmark depth may pull any vertex from the canonical prior, cm. */
  readonly maxLandmarkDeviationCm: number
  /**
   * Face-surface clearance AT THE SAGITTAL CENTRE (nose bridge/brow), cm.
   * Kept small: this is the region Rule 4 (anatomical alignment) needs tight,
   * so a badly calibrated frame is caught rather than hidden.
   */
  readonly faceSurfaceBiasCm: number
  /**
   * Face-surface clearance AT THE CHEEK/TEMPLE EDGE, cm. Kept larger than
   * faceSurfaceBiasCm: landmark reconstruction is noisiest here (steeper
   * viewing angle, more curvature, lower landmark confidence), and this is
   * exactly where a temple arm runs for several centimetres, so a shortfall
   * here reads as the arm being cut off right past the hinge instead of
   * fading in naturally near the ear. See MIRRORED_CANONICAL_LATERAL_BIAS.
   */
  readonly templeClearanceCm: number

  // ---- v2: anatomical clearance (Rule 4: frame rests ON the nose, not in it)
  /**
   * Lift the frame forward, in face space, just far enough that it is not
   * embedded in the face surface. Compensates calibrations that place the lens
   * plane at eye-corner depth (see docs). Only active in hardened mode.
   */
  readonly anatomicalClearanceEnabled: boolean
  /** Upper bound on the automatic forward lift, cm. */
  readonly maxClearanceCm: number

  // ---- v2: head + ear volumes (Rule 1: temples behind ears / head) -------
  readonly headProxyEnabled: boolean
  /** Pushes the skull volume away from the camera along the view ray, cm. */
  readonly headProxyPushCm: number
  readonly earProxyEnabled: boolean
  readonly earProxyPushCm: number

  // ---- v2: foreground mask (Rules 2 & 3: hair, hands, objects) -----------
  readonly foregroundMaskEnabled: boolean
  readonly hairOcclusionStrength: number
  /** 0 = only hair inside the face oval occludes; 1 = a wide ring around it too (temples). */
  readonly hairReach: number
  readonly handOcclusionStrength: number
  readonly objectOcclusionStrength: number

  // ---- v2: shadows & light (Rule 5) ---------------------------------------
  readonly contactShadowsEnabled: boolean
  readonly contactShadowOpacity: number
  /** Scale glasses lighting to the measured face brightness (heuristic). */
  readonly lightingMatchEnabled: boolean

  // ---- v2: camera model ----------------------------------------------------
  /** Make the renderer camera reproduce the tracker's projection (see TrackerProjection). */
  readonly matchTrackerCamera: boolean
  readonly trackerVerticalFovDeg: number

  // ---- v2: diagnostics -----------------------------------------------------
  readonly debugShowOccluders: boolean
  readonly debugShowMask: boolean
}

export interface Range {
  readonly min: number
  readonly max: number
}

/** Single source of truth for validation AND the control panel. */
export const FACE_OCCLUSION_RANGES = {
  referenceFaceWidthCm: { min: 8, max: 24 },
  depthScale: { min: 0.01, max: 4 },
  depthBias: { min: -1, max: 1 },
  surfaceScale: { min: 0.5, max: 1.5 },
  landmarkSmoothingHalfLifeSeconds: { min: 0.001, max: 0.5 },
  landmarkDepthBlend: { min: 0, max: 1 },
  maxLandmarkDeviationCm: { min: 0.1, max: 3 },
  faceSurfaceBiasCm: { min: 0, max: 1 },
  templeClearanceCm: { min: 0, max: 3 },
  maxClearanceCm: { min: 0, max: 4 },
  headProxyPushCm: { min: 0, max: 3 },
  earProxyPushCm: { min: 0, max: 2 },
  hairOcclusionStrength: { min: 0, max: 1 },
  hairReach: { min: 0, max: 1 },
  handOcclusionStrength: { min: 0, max: 1 },
  objectOcclusionStrength: { min: 0, max: 1 },
  contactShadowOpacity: { min: 0, max: 0.8 },
  trackerVerticalFovDeg: { min: 40, max: 90 },
} as const satisfies Record<string, Range>

export type NumericSettingKey = keyof typeof FACE_OCCLUSION_RANGES

export const DEFAULT_FACE_OCCLUSION_SETTINGS: FaceOcclusionSettings = Object.freeze({
  enabled: true,
  referenceFaceWidthCm: 14,
  depthScale: 1,
  depthBias: 0.08,
  surfaceScale: 1.015,
  // Kept fast (not 0) relative to the now-instant glasses pose smoothing:
  // literal 0 here would let raw per-frame landmark depth noise show up as
  // visible shimmer at the occlusion boundary (e.g. where a temple arm is
  // hidden behind the head), trading a lag artifact for a noise artifact.
  landmarkSmoothingHalfLifeSeconds: 0.02,

  mode: 'hardened',

  landmarkDepthBlend: 0.5,
  maxLandmarkDeviationCm: 0.8,
  faceSurfaceBiasCm: 0.1,
  templeClearanceCm: 1.4,

  anatomicalClearanceEnabled: true,
  maxClearanceCm: 3,

  headProxyEnabled: true,
  headProxyPushCm: 1.5,
  earProxyEnabled: true,
  earProxyPushCm: 0.7,

  foregroundMaskEnabled: false,
  hairOcclusionStrength: 0.85,
  hairReach: 0.35,
  handOcclusionStrength: 0.9,
  objectOcclusionStrength: 0.7,

  contactShadowsEnabled: false,
  contactShadowOpacity: 0.25,
  lightingMatchEnabled: false,

  // Match the renderer projection to the actual video crop by default. This is
  // essential on portrait phone viewports, where the 16:9 camera stream is
  // heavily cover-cropped and a fixed 60° FOV makes the glasses undersized.
  matchTrackerCamera: true,
  trackerVerticalFovDeg: 63,

  debugShowOccluders: false,
  debugShowMask: false,
})

const BOOLEAN_KEYS = [
  'enabled',
  'anatomicalClearanceEnabled',
  'headProxyEnabled',
  'earProxyEnabled',
  'foregroundMaskEnabled',
  'contactShadowsEnabled',
  'lightingMatchEnabled',
  'matchTrackerCamera',
  'debugShowOccluders',
  'debugShowMask',
] as const

/**
 * Merge an arbitrary (possibly partial, stale, or corrupt) payload onto the
 * defaults and clamp every number into its valid range. NEVER throws: an
 * invalid slider value or a hand-edited JSON file must not be able to stop the
 * render loop.
 */
export function normalizeFaceOcclusionSettings(
  input: Partial<Record<keyof FaceOcclusionSettings, unknown>> | null | undefined,
): FaceOcclusionSettings {
  const source = (input ?? {}) as Record<string, unknown>
  const result: Record<string, unknown> = { ...DEFAULT_FACE_OCCLUSION_SETTINGS }

  for (const key of BOOLEAN_KEYS) {
    if (typeof source[key] === 'boolean') result[key] = source[key]
  }

  if (source.mode === 'hardened' || source.mode === 'legacy') {
    result.mode = source.mode
  }

  for (const key of Object.keys(FACE_OCCLUSION_RANGES) as NumericSettingKey[]) {
    const value = source[key]
    if (typeof value === 'number' && Number.isFinite(value)) {
      const { min, max } = FACE_OCCLUSION_RANGES[key]
      result[key] = Math.min(max, Math.max(min, value))
    }
  }

  return Object.freeze(result) as unknown as FaceOcclusionSettings
}
