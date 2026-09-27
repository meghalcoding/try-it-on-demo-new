# Development Status

Current milestone: Phase 12 — Zero-Lag Glasses Tracking

Completed (Phase 12, see `docs/MOTION_LATENCY.md` for full detail):
- Made 0 half-life (instant, no smoothing) the actual default for glasses
  position/rotation tracking, not just an available setting: previously the
  UI's own default (0.08s) silently overrode the smoother's faster internal
  default (0.035s/0.025s) the moment the app started, so the shipped
  experience always had ~80ms of visible tracking lag ("glasses trailing a
  head turn").
- Found and fixed a real pre-existing bug in the velocity-adaptive smoothing:
  it could lerp the effective half-life UP toward its 0.012s floor as head
  motion sped up when the configured half-life was already at or below that
  floor -- the opposite of the mechanism's purpose, silently adding lag
  during fast motion. Fixed by clamping with `Math.min()` so the adaptive
  path can only ever reduce latency, never add it; a no-op for anyone using a
  non-zero configured half-life above the floor.
- Tightened the occlusion depth surface's own (separate) smoothing from 45ms
  to 20ms so its hidden/visible boundary doesn't visibly lag the now-instant
  glasses during a fast turn, without zeroing it outright (that surface is
  noisy per-frame reconstructed depth; zero smoothing there trades a latency
  artifact for a worse shimmer artifact).
- 16 new unit tests (`PoseSmoother.test.ts`), including a regression test
  that specifically catches the adaptive-direction bug by asserting identical
  output regardless of motion speed when configured half-life is at the
  floor. Full suite: 100/100 passing, clean `tsc -b` and `vite build`.

Current milestone: Phase 11 — Face Occlusion Hardening (depth, head/ears, hair/hand mask, fit, shadow)

Completed (Phase 11, see `docs/OCCLUSION_HARDENING.md` for full detail):
- Replaced the fixed-reference-width depth guess with a metrically correct face
  surface reconstructed from the pose matrix + bounded, confidence-weighted
  landmark personalisation (`FaceSurfaceReconstructor.ts`), closing TD-02.
  Synthetic ground-truth tests show ~1-2mm mean depth error vs. the legacy
  model's 3-4cm at typical chat distances.
- Added closed skull + ear depth volumes (`HeadProxy.ts`) so temple arms on
  the far side of a turned head are hidden instead of drawn over the hair.
- Added a real-time hair/hand/object occlusion mask via MediaPipe's
  `selfie_multiclass_256x256` segmenter, applied on the GPU by patching the
  glasses' own material shaders (`ForegroundSegmenter.ts`,
  `ForegroundOcclusion.ts`); fails closed on the known GPU class-scrambling
  issue and self-throttles to a duty cycle.
- Added frame/face fit diagnostics (`FitDiagnostics.ts`) that measure and
  auto-correct calibrations that embed the frame in the face -- a defect that
  was previously invisible because the old occluder's own depth error masked
  it. Exposed in the panel with a one-click "bake into calibration" action.
- Added contact shadows (dedicated shadow-only light + `ShadowMaterial`
  catcher) and an optional, bounded lighting-match heuristic.
- Added a tracker-camera-matched renderer FOV option
  (`TrackerProjection.ts`) so pose-space points project onto the same pixels
  as the tracked face under the video's `object-fit: cover` crop.
- Kept the previous occluder byte-identical as `occlusion/LegacyFaceOccluder.ts`,
  selectable via `settings.mode = 'legacy'` for on-camera A/B comparison.
  All v1 settings keys/exports still load unchanged.
- Hardened the render loop: per-frame update/render are individually
  try/caught so one bad frame degrades a layer instead of freezing the AR
  overlay; found and fixed a real pre-existing bug in a new diagnostic
  (`FaceHeightField.sample` returning 0 instead of null before first build).
- 82 new unit tests (Vitest) across 9 files covering every pure-math module;
  two additional GPU-level checks (ray-cast visibility, injected-shader
  pixel readback) run in a headless-Chromium sandbox outside the unit suite.
- `npx tsc -b` and `npx vite build` both pass clean; `npm run build` output
  unchanged in structure (single bundle, ~1.13MB before gzip).

Completed (Phase 11.1 — live-camera feedback fix):
- Live testing surfaced a real defect: temple arms were cut off abruptly right
  past the hinge instead of running along the head to the ear. Root cause: the
  face-surface occluder used one uniform push-back for the whole face, which
  can't be both tight at the nose/brow (needed for Rule 4) and generous at the
  cheek/temple (needed for arms, and exactly where landmark reconstruction is
  noisiest). Fixed with a per-vertex graduated push (`aLateral` attribute,
  `MIRRORED_CANONICAL_LATERAL_BIAS` in `canonicalFaceModel.ts`,
  `createFaceDepthMaterial` in `DepthOnlyMaterial.ts`): tight at the sagittal
  centre, generous toward the cheek/temple. New `templeClearanceCm` setting;
  `headProxyPushCm`/`earProxyPushCm` defaults also raised (proven safe by the
  existing "skull/ears stay behind the true face surface" invariant).
  Reproduced and verified fixed with a headless-Chromium pixel-readback test
  (synthetic arm with a deliberate 3-5mm inward offset: 0% visible past the
  hinge before the fix, 13-18% after, matching the zero-error baseline).
  84/84 unit tests pass; 2 new tests added for the lateral-bias gradient and
  the geometry attribute.

Completed (Phase 10):
- Fixed duplicate declarations in `FaceReferenceGeometry.ts`.
- Fixed GLB orientation normalization: removed the mathematically invalid centered-depth sign test that could invert Y/Z per model; canonical X/Y/Z snapping now prevents spurious PCA rotations for already-canonical assets.
- Fixed automatic eyewear eye-line placement: lens-midpoint-anchored models now derive X/Y from the canonical eye midpoint instead of the lower nose bridge.
- Fixed the remaining systematic Y offset at the calibration root: the fallback FaceReference now uses MediaPipe canonical eye-line Y=2.624618 cm instead of incorrectly treating the facial transformation-matrix origin as the eye line.
- Verified the five supplied catalog GLBs offline: the corrected orientation basis resolves to right-handed X/Y/Z for all five assets.
- Preserved camera, MediaPipe tracking, Three.js rendering, product schema, product switching, and asset files.

In progress:
- Live-camera empirical validation of the hardened occlusion system (real hair,
  hands, lighting, head turns) -- everything above has been validated with
  unit tests and a headless-GPU sandbox, not yet on a live device/browser.
- Full npm build verification requires the project's dependencies to be installed.

Next:
- Live camera validation of Phase 11, then production deployment.
- Consider baking the auto-clearance correction into each catalog product's
  stored calibration once live-validated, and code-splitting the bundle
  (currently a single >500KB chunk per the Vite build warning).

Blockers: None

## Tech Debt Register
- TD-01 - Full dependency installation/build verification requires registry access in the developer environment. - Severity: Low
- TD-02 - RESOLVED (Phase 11): the occluder now reconstructs metric depth from the pose matrix plus bounded landmark personalisation instead of a fixed reference width; see `FaceSurfaceReconstructor.ts`. The underlying constraint (approved browser task API does not expose the legacy Face Geometry runtime mesh directly) is unchanged, but is no longer a source of centimetre-scale error.
- TD-03 - The supplied catalog does not contain real pricing, so product price fields are set to 0 and are not surfaced as product pricing UI yet. - Severity: Low
- TD-04 - Product starting calibrations are asset-space baselines derived from GLB geometry; final camera validation is still required for production-quality fit. The new fit-diagnostics auto-clearance (Phase 11) masks most embedding in real time, but baking corrected Z values into the catalog is still open. - Severity: Low
- TD-05 - Full npm build verification in this execution environment is blocked by missing installed dependencies after the previous install timeout. - Severity: Low
- TD-06 - The hardened occlusion system (Phase 11) has had one round of live-camera
  feedback (temple-arm cutoff, fixed in Phase 11.1 — see above) but has not
  been systematically validated: real hair, hands, ambient lighting, and a
  full range of head motion/distance are still untested. - Severity: Med
- TD-07 - `selfie_multiclass_256x256` is fetched from a public Google Cloud Storage URL and the MediaPipe Tasks WASM runtime from a public jsDelivr CDN at runtime; production deployments may want to self-host both. - Severity: Low


## Latest iteration — portrait mobile projection correction (2026-09-27)

TASK
- Bug fix: glasses render too small in portrait phone orientation.

IMPLEMENTATION
- Enabled the existing tracker-camera-matched projection by default. The renderer already computes the appropriate vertical FOV from source video dimensions and the `object-fit: cover` crop; the feature was present but disabled in the shipped defaults. No pose, calibration, tracking, or occlusion algorithm changed.

FILES CHANGED
- `src/types/FaceOcclusion.ts`
- `DEVELOPMENT_STATUS.md`

VERIFICATION
- Source inspected; execution/build not run in this environment.

RESULT
- Default renderer now uses the existing cover-aware projection path on portrait and landscape surfaces. Must still be confirmed on a real phone.

ASSUMPTIONS
- The reported phone is using the app's existing `object-fit: cover` camera surface.

BLOCKERS
- Live-device verification not performed.


## Latest iteration — Occlusion defaults and persistent try-on viewport (2026-09-27)

TASK
- Set requested expensive/visual occlusion features off by default and keep the live try-on surface visible while navigating settings.

IMPLEMENTATION
- Changed startup defaults for foreground hair/hands/objects masking and contact shadows to false. Lighting match and both occlusion debug overlays were already false; preserved them. Existing exported JSON values continue to override defaults when explicitly loaded.
- Made the desktop settings column independently scrollable within the viewport and kept the try-on surface sticky. On narrow screens, the camera/product area remains sticky while the settings page scrolls.

FILES CHANGED
- `src/types/FaceOcclusion.ts`
- `src/App.css`
- `DEVELOPMENT_STATUS.md`

VERIFICATION
- Source inspection and archive extraction completed. Build/tests not executed.

RESULT
- The specified five controls are false at startup by default; `enabled`, anatomical clearance, head proxy, ear proxy, and tracker-camera matching retain their existing values.
- Settings scrolling is constrained to the right panel on desktop; the try-on surface remains in view.

ASSUMPTIONS
- “Hair, hands and objects” refers to the existing `foregroundMaskEnabled` master toggle; its per-class strength values remain intact.

BLOCKERS
- Build and live browser/device verification not performed.

NEXT TASK
- None

---

## Stakeholder Demo — Task 01 (2026-09-27)

Current milestone: Stakeholder Demo — Task 01 of 3 (immersive viewer foundation)

Completed this iteration:
- Reframed the app as a full-viewport try-on canvas with floating brand treatment and responsive overlay frame selector.
- Added premium dark glass surfaces, emerald accents, subtle grain, responsive safe-area handling, landscape adaptations, and reduced-motion support.
- Removed Circular 001 from the product catalog; the four other existing catalog entries and their GLB/calibration references remain unchanged.
- Existing calibration/developer control implementations remain in source for relocation into the diagnostics drawer in Task 02; temporarily hidden from the stakeholder-facing canvas.

Files changed:
- `src/App.css`
- `src/data/products.json`
- `DEVELOPMENT_STATUS.md`

Verification:
- Inspected App and carousel integration and retained the existing product selection handlers.
- Parsed product catalog JSON and checked the retained product IDs (see task execution verification).
- Build/lint and browser/device checks not executed; project dependencies are not installed in this workspace.

In progress: Task 02 — Collapsible diagnostics drawer and live developer HUD.
Next: Task 02
Blockers: None for Task 01. Automated build/device validation remains pending for Task 03.

Tech Debt Register:
- TD-01 - Run full build/lint and browser/device checks after dependency installation - Severity: Medium


## Latest iteration — Stakeholder diagnostics drawer (Task 02, 2026-09-27)

Current milestone: Stakeholder Demo — Task 02 of 3.

Completed:
- Replaced the always-visible calibration sidebar with a bottom-anchored expandable diagnostics drawer.
- Added live UI status cards sourced from existing camera, tracking, product-loading, calibration, and occlusion state.
- Relocated save-fit (local calibration JSON export) and face-mesh developer testing controls into the drawer.
- Added responsive, independently scrollable glassmorphism drawer styling.

In progress:
- Task 03 responsive and end-to-end verification.

Next:
- Task 03 — Responsive Demo Hardening & End-to-End Verification.

Blockers: Browser/device validation and build/lint remain to be run.

## Tech Debt Register
- TD-01 - Verify diagnostics drawer behavior with live camera on mobile Safari and Android browsers - Severity: Med

## Stakeholder Demo — Task 03 (2026-09-27)

Current milestone: Stakeholder Demo — Task 03 of 3 (responsive hardening and verification).

Completed:
- Hardened the collapsed diagnostics drawer for keyboard/accessibility behavior by making hidden drawer contents inert until expanded.
- Replaced placeholder smoothing HUD text with the live configured position and rotation half-life values.
- Confirmed the four active catalog products each reference an existing GLB and thumbnail, with the expected seven calibration keys; Circular 001 remains excluded from the active catalog.
- Reviewed camera permission/retry/stop branches, model loading error handling, model cleanup, and responsive/safe-area CSS for desktop, mobile, and short landscape layouts.

Verification:
- Catalog/asset integrity script executed successfully for all four active products.
- `npm ci --ignore-scripts --offline` attempted; dependency install failed because `zod-validation-error@4.0.2` is not present in the local npm cache and registry access is unavailable.
- Build, lint, automated tests, and real browser/device matrix not executed; dependencies unavailable and no browser/device harness in this environment.

Result:
- Task 03 source hardening and static asset checks completed. Responsive behavior and camera/model lifecycle were inspected, not live-device certified.

Assumptions:
- Smoothing HUD expresses the existing configured half-life values in milliseconds (seconds multiplied by 1000 for display only); runtime values are unchanged.

Blockers:
- Live browser/device verification (desktop, tablet, mobile portrait/landscape, permission denial, WebGL context loss/restore) remains outstanding.
- Dependency installation requires npm registry access or a populated cache to run build/lint/tests.

Next: Stakeholder demo tasks complete; provide the standalone project archive.

Tech Debt Register:
- TD-01 - Install dependencies and execute build/lint/test suite, then run the browser/device verification matrix - Severity: Medium
