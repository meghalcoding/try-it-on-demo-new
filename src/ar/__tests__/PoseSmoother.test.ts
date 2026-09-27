import { describe, expect, it } from 'vitest'
import { PoseSmoother, halfLifeAlpha } from '../PoseSmoother'
import type { FacePose } from '../FacePose'

function pose(overrides: Partial<FacePose> = {}): FacePose {
  return {
    position: { x: 0, y: 0, z: -45 },
    rotation: { x: 0, y: 0, z: 0, w: 1 },
    scale: 1,
    trackingState: 'detected',
    timestampMs: 0,
    ...overrides,
  }
}

describe('halfLifeAlpha', () => {
  it('is 0 for zero elapsed time regardless of half-life', () => {
    expect(halfLifeAlpha(0, 0.08)).toBe(0)
    expect(halfLifeAlpha(0, 0)).toBe(0)
  })

  it('converges fully in one step when half-life is 0 (instant/no smoothing)', () => {
    expect(halfLifeAlpha(1 / 240, 0)).toBe(1) // even a 240fps frame fully converges
    expect(halfLifeAlpha(0.1, 0)).toBe(1)
  })

  it('is exactly 0.5 after exactly one half-life', () => {
    expect(halfLifeAlpha(0.08, 0.08)).toBeCloseTo(0.5, 6)
  })

  it('rejects negative half-life or negative delta', () => {
    expect(() => halfLifeAlpha(0.1, -0.01)).toThrow()
    expect(() => halfLifeAlpha(-0.1, 0.08)).toThrow()
  })

  it('accepts a half-life of exactly 0 as valid (not an error)', () => {
    expect(() => halfLifeAlpha(0.1, 0)).not.toThrow()
  })
})

describe('PoseSmoother with half-life 0 (default): zero added latency', () => {
  it('snaps fully to the target pose on the very first frame', () => {
    const smoother = new PoseSmoother({ positionHalfLifeSeconds: 0, rotationHalfLifeSeconds: 0 })
    const target = pose({ position: { x: 3, y: -1, z: -50 }, scale: 1.2 })
    const result = smoother.update(target, 0)
    expect(result.position).toEqual(target.position)
    expect(result.scale).toBe(target.scale)
  })

  it('tracks a moving target with no lag on every subsequent frame', () => {
    const smoother = new PoseSmoother({ positionHalfLifeSeconds: 0, rotationHalfLifeSeconds: 0 })
    smoother.update(pose({ position: { x: 0, y: 0, z: -45 } }), 0)

    // Simulate a fast, continuous head movement across several frames.
    let x = 0
    for (let frame = 0; frame < 10; frame += 1) {
      x += 2 // 2cm per frame of movement
      const result = smoother.update(pose({ position: { x, y: 0, z: -45 } }), 1 / 60)
      // With zero half-life the output must equal the target exactly, every frame.
      expect(result.position.x).toBeCloseTo(x, 9)
    }
  })

  it('tracks rotation with no lag', () => {
    const smoother = new PoseSmoother({ positionHalfLifeSeconds: 0, rotationHalfLifeSeconds: 0 })
    smoother.update(pose(), 0)

    const target = pose({ rotation: { x: 0, y: 0.3826834, z: 0, w: 0.9238795 } }) // ~45deg yaw
    const result = smoother.update(target, 1 / 60)
    expect(result.rotation.x).toBeCloseTo(target.rotation.x, 6)
    expect(result.rotation.y).toBeCloseTo(target.rotation.y, 6)
    expect(result.rotation.z).toBeCloseTo(target.rotation.z, 6)
    expect(result.rotation.w).toBeCloseTo(target.rotation.w, 6)
  })
})

describe('PoseSmoother adaptive half-life direction (regression: must never ADD lag)', () => {
  it('instant (0) configured half-life fully converges even during fast motion', () => {
    const smoother = new PoseSmoother({ positionHalfLifeSeconds: 0, rotationHalfLifeSeconds: 0 })
    smoother.update(pose({ position: { x: 0, y: 0, z: -45 } }), 0)

    // A large jump in one frame triggers the "fast motion" adaptive branch.
    // Bug being guarded against: lerping the configured half-life (0) UP
    // toward the 0.012s adaptive floor as velocity increases would leave the
    // pose short of the target instead of snapping fully.
    const result = smoother.update(pose({ position: { x: 50, y: 0, z: -45 } }), 1 / 60)
    expect(result.position.x).toBeCloseTo(50, 6)
  })

  it('a configured half-life already below the adaptive floor is unaffected by motion speed', () => {
    // 0.005s is below MIN_ADAPTIVE_HALF_LIFE_SECONDS (0.012s). Before the fix,
    // the adaptive lerp would blend this UP toward 0.012 as velocity
    // increased, making fast motion respond MORE slowly than slow motion --
    // backwards. After the fix, min(configured, lerp(...)) clamps the
    // effective half-life to never exceed configured, so the result must be
    // identical (within floating point) regardless of how fast the target moves.
    const dt = 1 / 60
    const configured = 0.005
    const staticAlpha = 1 - Math.pow(0.5, dt / configured)

    const slow = new PoseSmoother({ positionHalfLifeSeconds: configured, rotationHalfLifeSeconds: configured })
    slow.update(pose({ position: { x: 0, y: 0, z: -45 } }), 0)
    const slowResult = slow.update(pose({ position: { x: 0.1, y: 0, z: -45 } }), dt) // tiny motion, posFactor ~ 0

    const fast = new PoseSmoother({ positionHalfLifeSeconds: configured, rotationHalfLifeSeconds: configured })
    fast.update(pose({ position: { x: 0, y: 0, z: -45 } }), 0)
    const fastResult = fast.update(pose({ position: { x: 50, y: 0, z: -45 } }), dt) // large motion, posFactor -> 1

    expect(slowResult.position.x).toBeCloseTo(0.1 * staticAlpha, 9)
    expect(fastResult.position.x).toBeCloseTo(50 * staticAlpha, 9)
  })

  it('still smooths normally (does not become instant) for an ordinary configured half-life', () => {
    const smoother = new PoseSmoother({ positionHalfLifeSeconds: 0.08, rotationHalfLifeSeconds: 0.08 })
    smoother.update(pose({ position: { x: 0, y: 0, z: -45 } }), 0)

    // Slow, small motion: adaptive factor ~0, so this should behave like the
    // static half-life (not fully converge in one small frame).
    const result = smoother.update(pose({ position: { x: 0.5, y: 0, z: -45 } }), 1 / 60)
    expect(result.position.x).toBeGreaterThan(0)
    expect(result.position.x).toBeLessThan(0.5)
  })
})

describe('PoseSmoother validation', () => {
  it('accepts a half-life of exactly 0 at construction and via setSettings', () => {
    expect(() => new PoseSmoother({ positionHalfLifeSeconds: 0, rotationHalfLifeSeconds: 0 })).not.toThrow()
    const smoother = new PoseSmoother()
    expect(() => smoother.setSettings({ positionHalfLifeSeconds: 0, rotationHalfLifeSeconds: 0 })).not.toThrow()
    expect(smoother.getSettings()).toEqual({ positionHalfLifeSeconds: 0, rotationHalfLifeSeconds: 0 })
  })

  it('rejects negative or non-finite half-life', () => {
    expect(() => new PoseSmoother({ positionHalfLifeSeconds: -0.01 })).toThrow()
    expect(() => new PoseSmoother({ rotationHalfLifeSeconds: Number.NaN })).toThrow()
  })

  it('rejects a malformed target pose (non-finite values)', () => {
    const smoother = new PoseSmoother()
    expect(() => smoother.update(pose({ scale: Number.NaN }), 1 / 60)).toThrow()
    expect(() => smoother.update(pose({ scale: 0 }), 1 / 60)).toThrow()
  })

  it('reset() forgets state so the next update snaps again', () => {
    const smoother = new PoseSmoother({ positionHalfLifeSeconds: 0.2, rotationHalfLifeSeconds: 0.2 })
    smoother.update(pose({ position: { x: 0, y: 0, z: -45 } }), 0)
    smoother.reset()
    expect(smoother.isInitialized()).toBe(false)

    const result = smoother.update(pose({ position: { x: 10, y: 0, z: -45 } }), 1 / 60)
    // First update after reset always snaps, even with a non-zero half-life.
    expect(result.position.x).toBe(10)
  })
})

describe('The app default is zero-lag', () => {
  it('DEFAULT_POSITION_HALF_LIFE_SECONDS / DEFAULT_ROTATION_HALF_LIFE_SECONDS are 0', async () => {
    const { DEFAULT_POSITION_HALF_LIFE_SECONDS, DEFAULT_ROTATION_HALF_LIFE_SECONDS } = await import('../PoseSmoother')
    expect(DEFAULT_POSITION_HALF_LIFE_SECONDS).toBe(0)
    expect(DEFAULT_ROTATION_HALF_LIFE_SECONDS).toBe(0)
  })
})
