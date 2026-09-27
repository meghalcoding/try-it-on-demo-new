# Motion Latency: Zero-Lag Glasses Tracking

This documents making the glasses track the face with no added latency —
"stuck to the face," no perceptible smoothing delay between head movement and
glasses movement.

## Where latency could come from, and where it actually was

The pipeline from camera to rendered glasses is:

```
video frame -> FaceTracker (MediaPipe) -> FacePose -> PoseSmoother -> GlassesAnchor.compose -> model transform
```

Checking each stage:

- **`VideoFrameScheduler.ts`** drives detection from the video's own frame
  callback (`requestVideoFrameCallback`, falling back to
  `requestAnimationFrame`) with no throttling or frame-skipping — this already
  runs as fast as the browser delivers frames.
- **`coordinateTransform.ts`** and **`GlassesAnchor.ts`** are explicitly
  documented in their own source as performing "no calibration or smoothing" —
  pure, immediate matrix composition.
- **`PoseSmoother.ts`** is the one deliberate latency source: an exponential
  filter (half-life based) that trades some immediacy for reduced jitter from
  landmark noise.

So the only knob that matters is `PoseSmoother`, and — before this change —
it wasn't even set to its own fastest setting: the class's internal defaults
(0.035s position / 0.025s rotation) were immediately overridden by the
`SmoothingPanel` UI defaults of 0.08s / 0.08s the moment the app started,
because `App.tsx` calls `renderer.setPoseSmoothingSettings(smoothingSettings)`
right after construction. An 80ms half-life is a small but very much
perceptible delay — visible as the classic "glasses trailing behind a head
turn" effect.

## The fix: 0 half-life is now the actual default

`PoseSmoother` treats a half-life of `0` as a first-class value meaning
**instant, no smoothing**: the pose fully converges to the latest target every
single frame.

```ts
// PoseSmoother.ts
export function halfLifeAlpha(deltaSeconds: number, halfLifeSeconds: number): number {
  ...
  if (halfLife === 0) {
    return deltaSeconds > 0 ? 1 : 0 // explicit branch, not float underflow
  }
  return 1 - Math.pow(0.5, deltaSeconds / halfLife)
}
```

Both the class defaults (`DEFAULT_POSITION_HALF_LIFE_SECONDS`,
`DEFAULT_ROTATION_HALF_LIFE_SECONDS`) and the UI defaults
(`SmoothingPanel.DEFAULT_SMOOTHING_SETTINGS`) are now `0`, so there is no
window — even for a single frame — where a non-zero half-life is briefly
active before a later effect corrects it. The Motion panel's slider range now
extends down to `0` (was `0.02`) so this is reachable and visible in the UI,
not just a hidden default.

Smoothing is still available for anyone who wants a touch of damping over
raw landmark jitter (the slider goes up to 0.3s, unchanged), but instant
tracking is now what a person gets without touching a setting.

## A real bug found while making this the default

`PoseSmoother` has a velocity-adaptive mechanism: it reduces the effective
half-life during fast head motion, so a static (comfortable-when-still)
smoothing amount doesn't turn into visible tracking lag during a quick turn.
It computed:

```ts
const effectiveHalfLife = lerp(configuredHalfLife, MIN_ADAPTIVE_HALF_LIFE_SECONDS /* 0.012s */, velocityFactor)
```

This is only correct when `configuredHalfLife > 0.012s`. Once the configured
value is *already at or below* that floor — which, after this change, is
every user's default (`0`) — the lerp blends the effective half-life **up**
toward 0.012s as velocity increases. That's backwards: it would have made the
smoother respond *more slowly* during exactly the fast motion the mechanism
exists to keep fast, silently reintroducing a small amount of lag on quick
head turns even with "0" configured.

Fixed by clamping the result so it can only reduce latency, never add it:

```ts
const effectiveHalfLife = Math.min(
  configuredHalfLife,
  lerp(configuredHalfLife, MIN_ADAPTIVE_HALF_LIFE_SECONDS, velocityFactor),
)
```

For the normal case (`configuredHalfLife > 0.012s`, e.g. someone who dialed
in some smoothing) this is a no-op — the lerp result was already `<=`
configured, so `min()` doesn't change anything, and existing tuned behavior
is unaffected. For `configuredHalfLife <= 0.012s` (including `0`), it now
correctly stays at the configured value regardless of velocity.

## Testing

`src/ar/__tests__/PoseSmoother.test.ts` (16 tests) covers:

- `halfLifeAlpha(dt, 0) === 1` for any `dt > 0`, and `=== 0` for `dt === 0`.
- A `PoseSmoother` configured with `0`/`0` reproduces the exact target
  position, rotation and scale every frame — including across a sequence of
  frames simulating continuous fast motion (a `for` loop moving 2cm/frame for
  10 frames; every single output must equal the input target, not just the
  final one).
- The adaptive-direction regression specifically: a half-life configured at
  or below the `0.012s` floor (`0` and `0.005` are both tested) produces
  **identical results regardless of how fast the target is moving** — a slow,
  tiny motion and a fast, large motion both get exactly
  `1 - 0.5^(dt / configuredHalfLife)` convergence, proving the adaptive
  mechanism is correctly inert rather than silently adding lag.
- A normal (unaffected) configured half-life (`0.08`) still smooths as
  before, so this remains an available, correctly-functioning feature for
  anyone who wants it.
- Validation: `0` is accepted as a valid half-life at construction and via
  `setSettings`; negative or non-finite values are still rejected.

Together with the existing `FaceOccluder`/`ARRenderer` test suite (unaffected
by this change — the occlusion volumes are positioned by whatever pose
`PoseSmoother` outputs, regardless of its latency), the full suite is
100/100 passing after this change.

## What was deliberately left alone

The occlusion depth surface's own smoothing (`landmarkSmoothingHalfLifeSeconds`
in `FaceOcclusion` settings, see `docs/OCCLUSION_HARDENING.md`) was tightened
(45ms → 20ms) to keep pace with the now-instant glasses, but not zeroed: that
surface is reconstructed from noisy per-frame landmark depth, and removing
its smoothing entirely would trade a latency artifact for a worse one —
visible shimmer at the hidden/visible boundary (e.g. flickering exactly where
a temple arm should disappear behind the head). This is a different signal
(reconstructed 3D depth, not the pose transform) with different noise
characteristics, so it keeps its own, separately-tunable smoothing.

## Live-testing note

This has been verified with unit tests (deterministic, frame-rate-independent
math) but not yet on a live camera. If any residual perceived lag remains
after this change, the most likely remaining source is detector cadence —
how often MediaPipe actually produces a new landmark result, which is a
sampling-rate limit of the tracker itself, not smoothing, and can't be
eliminated the same way (though it could be mitigated with pose
extrapolation/prediction between detections, which does not exist yet).
