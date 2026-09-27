import { describe, expect, it } from 'vitest'
import {
  DEFAULT_FACE_OCCLUSION_SETTINGS,
  FACE_OCCLUSION_RANGES,
  normalizeFaceOcclusionSettings,
} from '../FaceOcclusion'

describe('normalizeFaceOcclusionSettings', () => {
  it('never throws and returns defaults for null/undefined', () => {
    expect(normalizeFaceOcclusionSettings(null)).toEqual(DEFAULT_FACE_OCCLUSION_SETTINGS)
    expect(normalizeFaceOcclusionSettings(undefined)).toEqual(DEFAULT_FACE_OCCLUSION_SETTINGS)
    expect(normalizeFaceOcclusionSettings({})).toEqual(DEFAULT_FACE_OCCLUSION_SETTINGS)
  })

  it('clamps every numeric key to its documented range', () => {
    const input: Record<string, unknown> = {}
    for (const key of Object.keys(FACE_OCCLUSION_RANGES)) input[key] = 1e9
    const result = normalizeFaceOcclusionSettings(input)
    for (const [key, range] of Object.entries(FACE_OCCLUSION_RANGES)) {
      expect(result[key as keyof typeof result]).toBe(range.max)
    }

    const negInput: Record<string, unknown> = {}
    for (const key of Object.keys(FACE_OCCLUSION_RANGES)) negInput[key] = -1e9
    const negResult = normalizeFaceOcclusionSettings(negInput)
    for (const [key, range] of Object.entries(FACE_OCCLUSION_RANGES)) {
      expect(negResult[key as keyof typeof negResult]).toBe(range.min)
    }
  })

  it('rejects non-finite numbers and falls back to the default', () => {
    const result = normalizeFaceOcclusionSettings({
      maxLandmarkDeviationCm: Number.NaN,
      headProxyPushCm: Number.POSITIVE_INFINITY,
    })
    expect(result.maxLandmarkDeviationCm).toBe(DEFAULT_FACE_OCCLUSION_SETTINGS.maxLandmarkDeviationCm)
    expect(result.headProxyPushCm).toBe(DEFAULT_FACE_OCCLUSION_SETTINGS.headProxyPushCm)
  })

  it('ignores garbage types instead of throwing', () => {
    const result = normalizeFaceOcclusionSettings({
      enabled: 'yes' as unknown as boolean,
      mode: 'quantum' as unknown as 'hardened',
      hairOcclusionStrength: 'lots' as unknown as number,
      debugShowMask: 1 as unknown as boolean,
    })
    expect(result.enabled).toBe(DEFAULT_FACE_OCCLUSION_SETTINGS.enabled)
    expect(result.mode).toBe(DEFAULT_FACE_OCCLUSION_SETTINGS.mode)
    expect(result.hairOcclusionStrength).toBe(DEFAULT_FACE_OCCLUSION_SETTINGS.hairOcclusionStrength)
    expect(result.debugShowMask).toBe(DEFAULT_FACE_OCCLUSION_SETTINGS.debugShowMask)
  })

  it('accepts a legitimate partial override (e.g. a v1 exported file)', () => {
    const result = normalizeFaceOcclusionSettings({
      enabled: true,
      referenceFaceWidthCm: 15,
      depthScale: 1.2,
    })
    expect(result.referenceFaceWidthCm).toBe(15)
    expect(result.depthScale).toBe(1.2)
    expect(result.mode).toBe('hardened')
    expect(result.headProxyEnabled).toBe(true)
  })

  it('round-trips the mode switch', () => {
    expect(normalizeFaceOcclusionSettings({ mode: 'legacy' }).mode).toBe('legacy')
    expect(normalizeFaceOcclusionSettings({ mode: 'hardened' }).mode).toBe('hardened')
  })

  it('returns a frozen object', () => {
    const result = normalizeFaceOcclusionSettings({})
    expect(Object.isFrozen(result)).toBe(true)
  })
})
