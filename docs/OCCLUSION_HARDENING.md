# Face Occlusion Hardening

This documents the hardened occlusion system in `src/ar/occlusion/` and how it
maps to the five occlusion rules for virtual glasses try-on. It assumes
familiarity with `DEVELOPMENT_STATUS.md` and the original architecture.

## TL;DR

The original `FaceOccluder` was a single flat mesh, screen-projected from
normalized MediaPipe landmarks using a fixed "reference face width" to guess
depth. That guess is only correct when the face fills the entire video frame
width — at a normal 40–70 cm chat distance it is off by **2–4 cm**, an order
of magnitude larger than the few millimetres of clearance that separate
"glasses sit on the nose" from "temple arm behind the ear." It also had no
representation of the head or ears (so a temple arm on the far side of a
turned head had nothing to hide behind it), no hair/hand/object occlusion, and
no shadows.

The hardened system replaces this with four cooperating layers, all
individually unit-tested, all independently toggleable, and none of which can
throw from the per-frame path:

| Layer | File(s) | Occlusion rule(s) |
|---|---|---|
| Metric face surface | `canonicalFaceModel.ts`, `TrackerProjection.ts`, `FaceSurfaceReconstructor.ts` | 1 (depth), 4 (nose contour) |
| Head + ear volumes | `HeadProxy.ts` | 1 (behind-the-ear) |
| Foreground segmentation mask | `maskMath.ts`, `ForegroundSegmenter.ts`, `ForegroundOcclusion.ts` | 2 (hair), 3 (hands/objects) |
| Anatomical clearance + fit diagnostics | `FitDiagnostics.ts` | 4 (rests on, not in, the nose) |
| Contact shadows + light estimate | `ContactShadow.ts`, `LightEstimator.ts` | 5 (shadow/light) |

`FaceOccluder.ts` and `ARRenderer.ts` wire these together. The old occluder is
kept, byte-for-byte, as `occlusion/LegacyFaceOccluder.ts`, selectable via
`settings.mode = 'legacy'` for A/B comparison on a live camera.

## Rule 1 — Depth & ear occlusion

### The core bug: fixed reference width

The original code did:

```ts
const depthCm = -landmark.z * referenceFaceWidthCm // referenceFaceWidthCm = 14
```

MediaPipe's landmark `z` is in "roughly the same units as x", and `x` is
normalized by image **width**, not by face width. The correct conversion is

```
cmPerNormalizedX(distance) = 2 * distance * tan(hfov/2)
```

i.e. depth scale depends on how much of the frame the face occupies, which
depends on distance — not on a constant. `TrackerProjection.ts` derives this
from MediaPipe Face Geometry's documented 63° vertical FOV
(`MEDIAPIPE_FACE_GEOMETRY_VFOV_DEG`, kept as one named, overridable constant)
and the tracked face distance.

### The fix: pose prior + bounded landmark residual

`FaceSurfaceReconstructor.ts` does not trust raw landmark depth at all.
Instead, for every one of the 468 canonical vertices it computes:

1. **Expected depth** from the tracker's own pose matrix
   (`facialTransformationMatrixes`) applied to the embedded MediaPipe
   canonical face model (`canonicalFaceModel.ts`, vertices fetched from
   MediaPipe's own `canonical_face_model.obj`, Apache-2.0). This is metrically
   exact (it's the same transform that positions the glasses) but generic — an
   average face, not the user's.
2. **Landmark relief**, i.e. the person's actual shape, from
   `-landmark.z * cmPerNormalizedX(distance)`, mean-centred over a stable
   upper-face vertex subset (jaw/mouth excluded — they move with expression).
3. A **bounded residual**: `clamp(relief - expected, ±maxDeviationCm)`, so a
   bad tracker FOV assumption or noisy z can only pull a vertex by a capped
   amount, never corrupt the whole surface.
4. A **confidence weight**: landmarks are re-projected through the pose and
   compared to their own rays; large disagreement (e.g. a hand covering half
   the face) fans the weight toward 0, i.e. toward the rigid canonical prior.

Result (synthetic ground truth, `FaceSurfaceReconstructor.test.ts`): mean
depth error ~1–2 mm vs. the legacy model's 3–4 cm at typical distances — see
`is far more accurate than the previous (legacy) screen-space occluder`.

The surface is smoothed **in face-local space** (before the pose is applied),
then placed every render frame with the *same* smoothed pose that positions
the glasses (`FaceOccluder.getPoseRoot()`), so the two can never drift apart.

### Graduated clearance: nose tight, temple generous

A face-surface push cannot be a single number. The nose bridge/brow region is
exactly where Rule 4 needs tight accuracy (a badly fit frame should be
visibly caught, not hidden by a generous push); the cheek/temple region needs
the opposite — landmark reconstruction is naturally noisiest there (steeper
viewing angle at any yaw, more surface curvature, lower MediaPipe landmark
confidence), and it's exactly where a temple arm runs for several
centimetres. Live-camera testing surfaced this directly: with one uniform
push, temple arms were cut off abruptly right past the hinge instead of
running along the head and fading in near the ear.

The fix (`MIRRORED_CANONICAL_LATERAL_BIAS` in `canonicalFaceModel.ts`,
`createFaceDepthMaterial` in `DepthOnlyMaterial.ts`) is a per-vertex bias, 0
at the sagittal centre rising to 1 at the cheek/temple edge, derived from the
mesh's own geometry (fraction of max |x|, smoothstepped — not a hand-picked
cm constant). The face-surface material mixes push by this attribute:
`push = mix(faceSurfaceBiasCm, templeClearanceCm, aLateral)`, so
`faceSurfaceBiasCm` (small, e.g. 0.1cm) still governs the nose/brow and
`templeClearanceCm` (larger, e.g. 1.4cm) governs the cheek/temple
independently. `HeadProxy`'s skull/ear pushes were also raised (0.7→1.5cm,
0.25→0.7cm): both volumes are proven to stay behind the true face surface
regardless of push amount (see the `HeadProxy.test.ts` invariant), so more
push there is free forgiveness with no downside.

Verified with a headless-Chromium pixel-readback test: a synthetic temple arm
placed with a deliberate 3–5mm inward offset (simulating realistic
calibration/reconstruction slack) was cut off exactly at the hinge (0%
visible past it) under the old uniform push; with the graduated push it
recovers to 13–18% visible past the hinge, matching or exceeding the
zero-error baseline. Not committed as a repo test (needs a GPU context), but
reproducible via the same headless-Chromium + SwiftShader approach documented
under `Testing` below.

### Head and ears

The face mesh is an open mask that stops at the ear-front plane
(z ≈ −2.4 cm in canonical space). Behind that, nothing represents the head, so
a temple arm on the far side of a turned head had nothing to occlude it.
`HeadProxy.ts` adds two more depth-only volumes, positioned by the *same*
pose:

- **Skull**: a closed ellipsoid fit to adult anthropometric means (head
  breadth ≈15 cm, glabella–opisthocranion ≈19 cm), kept ≥0.5 cm behind the
  canonical face surface everywhere in the face footprint (asserted by
  `HeadProxy.test.ts`) so it can never eat into eyewear resting on the face.
- **Ears**: an anatomically-sized mirror pair, anchored near the face-oval
  edge.

Both are depth-only (`DepthOnlyMaterial.ts`: colour writes off, and each
vertex is pushed a few millimetres **away from the camera along its own view
ray** — not shrunk or offset — so silhouettes stay full size while a temple
resting flush against the skin isn't clipped away). `HeadProxy.test.ts`
verifies that adding these volumes strictly increases (never decreases) how
much of a synthetic far-side temple arm is hidden as yaw increases, and that
the near-side arm becomes *more* visible as it swings toward the camera.

## Rule 2 & 3 — Hair, hand and object occlusion

The face depth mesh only knows about the face itself; it cannot know about
hair, fingers, or a phone in front of it. That needs actual video-frame
segmentation.

`ForegroundSegmenter.ts` runs MediaPipe's `selfie_multiclass_256x256`
`ImageSegmenter` (hair / body-skin / face-skin / clothes / others) on a
downscaled (256px) frame, temporally smoothed, and packs the classes into a
compact RGBA texture (`MaskAssembler` — pure, unit-tested independently of the
ML call in `maskMath.test.ts` / `ForegroundSegmenter.ts`). Design constraints:

- **Non-blocking**: face tracking and the depth occluder never wait on this;
  until `status === 'ready'` there is simply no mask.
- **Self-throttling**: it measures its own inference duration and caps itself
  to a duty cycle (`MAX_DUTY_CYCLE`), so a slow CPU delegate lowers the mask
  refresh rate, not the render frame rate.
- **Fails closed on a known class-scrambling bug**: some GPU delegates
  (documented MediaPipe issue, notably iOS Safari) return confidence masks
  with scrambled class order. `classLayoutLooksValid()` checks that face-skin
  actually dominates the face interior; if not, it retries once on CPU, then
  marks itself `unavailable` rather than occluding the glasses with the wrong
  class (e.g. "clothes").

`ForegroundOcclusion.ts` applies the mask on the GPU: it patches the glasses'
*own* material shaders (`onBeforeCompile`, keyed so PBR features like
normal/roughness maps are unaffected) to multiply fragment alpha by
`1 - hairOrHandOrObjectOcclusion(pixel)`. The mapping from `gl_FragCoord` to
mask UV mirrors the display exactly (unit-tested in `maskMath.test.ts`'s
`screenToMaskUv`, with a GPU shader test cross-checking the actual injected
GLSL against that same JS reference — see `Testing` below). Patching **fails
closed**: if a material's shader doesn't contain the expected anchor, it's
left alone (no mask occlusion for that one material) with a single console
warning, never a crash.

Hair is gated to the face interior (bangs/strands) with a `hairReach` control
to widen it toward the temples; hands/objects are gated to a wider ring around
the face and excluded wherever the segmenter thinks the pixel is the user's
own face-skin (protects against mislabeled skin cutting a hole in the frame).
A small motion-compensation shift (`setMotionShift`) corrects for the mask
lagging a few frames behind fast head motion.

## Rule 4 — Facial feature alignment (nose bridge / brow)

Two different problems live under this rule:

1. **The nose bridge should occlude the frame at extreme angles** — this is
   now just a consequence of the accurate face surface (Rule 1's fix) plus
   the skull volume; no separate code is needed.
2. **The frame should rest ON the face, not be clipped into it.** This is a
   *new, visible* consequence of fixing Rule 1: the old occluder sat ~2–4 cm
   too far from the camera (see above), which silently hid any calibration
   error that sank a frame's nose bridge or brow line into the skin. With an
   accurate surface, an over-inserted calibration now visibly loses geometry
   to the depth test — correct behaviour, but it means calibration quality
   now matters in a way it didn't before.

`FitDiagnostics.ts` makes this measurable and self-correcting:

- `FaceHeightField` rasterizes the canonical face surface into a 0.25 cm grid
  height field in face-local space (rotation/distance-independent).
- `collectFrameSamples` samples the model's **opaque** meshes only (lenses are
  meant to sit in front of the eye and are excluded by their
  `transparent`/`opacity` flag).
- `computeFitReport` transforms those samples by the calibration and checks
  penetration against the height field, returning a verdict
  (`ok` / `marginal` / `embedded`) and a `suggestedForwardCm` — the forward
  lift that would clear the 90th-percentile embedded vertex.

`ARRenderer` measures this a few times a second and smoothly applies up to
`maxClearanceCm` of forward lift (a view-space offset added to calibration Z,
not to the occluder) so miscalibrated products degrade to "slightly forward of
ideal" instead of "visibly eaten by the depth test." The panel surfaces the
verdict and offers a **"Bake into this product's calibration"** action, so the
correction becomes permanent (and the automatic lift decays back toward zero)
once someone reviews it — this is a diagnostic and a safety net, not a
replacement for calibrating the product asset correctly.

Measured on the current catalogue via a synthetic penetration test (headless
Chromium + the real `AutoCalibrationEngine`): most auto- and product-calibrated
assets have their frame front embedded 1.5–2.9 cm into the canonical face
surface under the *old* fixed-depth occluder's effective tolerance. This was
invisible before because the old occluder's own depth was off by a similar
amount in the same direction.

## Rule 5 — Lighting and shadow occlusion

- **Contact shadows** (`ContactShadow.ts`): a dedicated, intensity-0
  `DirectionalLight` (so it doesn't change how the glasses are *lit*, only
  what they *shadow*) renders a shadow map from the glasses onto a second
  draw of the face surface using `THREE.ShadowMaterial`. The catcher is a
  child of the same pose-driven root as the face occluder, so it always
  follows the face exactly. Only opaque meshes cast (lenses don't).
- **Light matching** (`LightEstimator.ts`) is explicitly a coarse heuristic,
  off by default: it samples mean/left-right/top-bottom luminance over the
  central face box from a tiny downscaled frame and derives a bounded
  intensity/direction adjustment, smoothed with a half-life. It is not
  physical light estimation — it cannot recover a real light direction from
  one 2D brightness gradient — so every output is clamped and it only ever
  *nudges* the existing local lighting rig, never replaces it outright.

## Camera model

A more subtle bug, independent of the four layers above: the renderer's
`PerspectiveCamera` used a fixed 60° FOV. MediaPipe's tracker uses a fixed
63° virtual camera over the *entire video frame*, and the video is displayed
under `object-fit: cover`. Two different FOVs (63° vs 60°) plus an
un-accounted-for crop can misalign anything positioned from pose data with
what's on screen by a small but real amount. `TrackerProjection.ts`
(`computeCoverMapping`, `matchedVerticalFovDeg`) derives the correct
renderer FOV so a tracker-space point projects onto the same screen pixel as
the real face; it's opt-in (`matchTrackerCamera`) because it changes on-screen
glasses size by a few percent versus the previous behaviour, which existing
calibrations may have been tuned against.

## Motion latency and the occlusion surface

Glasses position/rotation latency is a separate concern from occlusion,
covered in full in `docs/MOTION_LATENCY.md`. The one place it touches this
document: the occlusion depth surface has its own smoothing stage
(`landmarkSmoothingHalfLifeSeconds`), tightened from 45ms to 20ms alongside
the pose-latency work so the hidden/visible boundary it produces (e.g. where
a temple arm disappears behind the head) doesn't visibly lag behind the
glasses during a fast head turn. It was deliberately *not* set to 0 like the
pose smoother: that surface is rebuilt from noisy per-frame landmark depth,
and zero smoothing there trades a latency artifact for a worse one — visible
shimmer at the occlusion boundary. It remains independently tunable in the
occlusion panel.



`types/FaceOcclusion.ts` keeps the original (v1) settings keys verbatim —
`referenceFaceWidthCm`, `depthScale`, `depthBias`, `surfaceScale` — used only
when `mode === 'legacy'`, so previously exported `face-occlusion.json` files
still load and behave identically. `normalizeFaceOcclusionSettings()` clamps
every numeric field to a documented range and never throws, so a corrupt or
hand-edited settings file can't stop the render loop.

## Fail-safe design

Every layer is independently toggleable and defaults to a safe fallback:

- Reconstruction rejects malformed input (`FaceSurfaceReconstructor.reconstruct`
  returns `{ ok: false }`) rather than producing NaN geometry.
- `FaceOccluder` holds the last good surface for up to 500 ms
  (`MAX_HOLD_MS`) on a tracking gap, then hides rather than freezing a stale
  surface indefinitely.
- `ARRenderer`'s per-frame update and render calls are individually
  try/caught; a bug in one layer degrades that layer for a second (throttled
  via `onRuntimeError`) rather than freezing the whole AR overlay — the
  original renderer's single unguarded per-frame call could brick the entire
  view on one bad frame.
- The segmenter and shader-patching both fail closed: no mask, or no
  per-material occlusion, rather than a wrong or crashing result.

## Testing

All math-heavy modules are pure functions or classes with no DOM/WebGL/ML
dependency and are unit-tested with Vitest (`npx vitest run`,
`src/**/*.test.ts`, 82 tests across 9 files at time of writing):

- `TrackerProjection.test.ts` — cover mapping, FOV derivation, ray math.
- `FaceOcclusion.test.ts` (types) — settings clamping/robustness.
- `canonicalFaceModel.test.ts` — embedded MediaPipe data integrity, plus the
  `MIRRORED_CANONICAL_LATERAL_BIAS` gradient (tight at centre, generous at
  the cheek/temple edge).
- `HeadProxy.test.ts` — skull-behind-face invariant, yaw-dependent visibility.
- `FaceSurfaceReconstructor.test.ts` — accuracy vs. synthetic ground truth,
  accuracy vs. the real legacy occluder, bounded-deviation and
  confidence-degradation behaviour, robustness to a wrong tracker FOV.
- `maskMath.test.ts` — polygon fill, blur, screen↔mask UV mapping (the exact
  reference the injected GLSL is checked against), class-layout validity.
- `FitDiagnostics.test.ts` — height-field sampling, frame-sample collection,
  embedding detection and self-correction.
- `LightEstimator.test.ts` — luminance/direction sampling and smoothing.
- `FaceOccluder.test.ts` — integration: mode switching, hold/drop timing,
  malformed-input resilience, dispose safety, long randomized frame
  sequences.

Two things are validated outside the unit suite, in a headless Chromium +
SwiftShader sandbox (not checked in — GPU/WebGL rendering isn't something
Vitest can exercise), and should be re-run on a real device before shipping:

1. Ray-cast/pixel-readback comparisons of arm visibility with/without the
   head+ear volumes across yaw, and legacy vs. hardened mode.
2. A GPU shader test that renders the actual injected `ForegroundOcclusion`
   fragment shader against a synthetic mask and confirms occlusion appears
   only where, and to the degree, `maskMath`'s pure JS model predicts.

Live camera testing (real hair, real hands, real lighting, real head turns)
has not been performed as part of this change and is the natural next step
before shipping — see the panel's `mode` toggle and `debugShow*` flags, which
exist specifically to make that comparison and debugging easy.
