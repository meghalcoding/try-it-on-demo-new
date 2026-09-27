/**
 * Camera model of the MediaPipe face tracker, plus the video "cover" mapping.
 *
 * WHY THIS EXISTS
 * ---------------
 * The FaceLandmarker's `facialTransformationMatrixes` are expressed in a
 * *virtual* pinhole camera space, not in the renderer's camera space. In the
 * MediaPipe Face Geometry pipeline that virtual camera has a 63 degree
 * VERTICAL field of view over the full video frame
 * (`perspective_camera { vertical_fov_degrees: 63.0 }`). The Tasks web API
 * does not expose this value, so it is an *assumption* here and is kept as a
 * single named constant that can be overridden if live testing shows the
 * pose scale drifting with a particular camera.
 *
 * Two things follow from it and are implemented below as pure functions so
 * they can be unit-tested without a GPU:
 *
 *  1. Landmark z is "the same scale as x", and x is normalised by the image
 *     WIDTH. Converting landmark depth to centimetres therefore needs the
 *     physical width of the image plane at the face distance
 *     (`cmPerNormalizedX`), not a fixed "reference face width".
 *
 *  2. To make a Three.js camera project a tracker-space point onto the same
 *     pixel as the real face in the (object-fit: cover) video, its vertical
 *     FOV must be derived from the tracker FOV and the crop
 *     (`matchedVerticalFovDeg`).
 */

/** Vertical FOV of MediaPipe's Face Geometry virtual camera (degrees). */
export const MEDIAPIPE_FACE_GEOMETRY_VFOV_DEG = 63

/** Bounds that keep the derived values numerically sane. */
const MIN_VFOV_DEG = 20
const MAX_VFOV_DEG = 120

export interface CoverMapping {
  /** Size the video occupies on the surface after `object-fit: cover`. */
  readonly renderedWidth: number
  readonly renderedHeight: number
  /** Top-left of the rendered video relative to the surface (<= 0 when cropped). */
  readonly offsetX: number
  readonly offsetY: number
  readonly scale: number
}

/**
 * Geometry of `object-fit: cover` for a video inside a surface. When either
 * video dimension is unknown the video is treated as filling the surface.
 */
export function computeCoverMapping(
  surfaceWidth: number,
  surfaceHeight: number,
  videoWidth: number,
  videoHeight: number,
): CoverMapping {
  if (!(videoWidth > 0) || !(videoHeight > 0) || !(surfaceWidth > 0) || !(surfaceHeight > 0)) {
    return {
      renderedWidth: Math.max(surfaceWidth, 0),
      renderedHeight: Math.max(surfaceHeight, 0),
      offsetX: 0,
      offsetY: 0,
      scale: 1,
    }
  }

  const scale = Math.max(surfaceWidth / videoWidth, surfaceHeight / videoHeight)
  const renderedWidth = videoWidth * scale
  const renderedHeight = videoHeight * scale

  return {
    renderedWidth,
    renderedHeight,
    offsetX: (surfaceWidth - renderedWidth) / 2,
    offsetY: (surfaceHeight - renderedHeight) / 2,
    scale,
  }
}

export interface TrackerFrustum {
  /** tan(vfov / 2) */
  readonly tanV: number
  /** tan(hfov / 2) over the *full video frame* */
  readonly tanH: number
}

export function clampVerticalFov(vfovDeg: number): number {
  if (!Number.isFinite(vfovDeg)) return MEDIAPIPE_FACE_GEOMETRY_VFOV_DEG
  return Math.min(MAX_VFOV_DEG, Math.max(MIN_VFOV_DEG, vfovDeg))
}

/** Tangent half-extents of the tracker's virtual camera over the full video frame. */
export function trackerFrustum(
  videoWidth: number,
  videoHeight: number,
  vfovDeg: number = MEDIAPIPE_FACE_GEOMETRY_VFOV_DEG,
): TrackerFrustum {
  const aspect = videoWidth > 0 && videoHeight > 0 ? videoWidth / videoHeight : 16 / 9
  const tanV = Math.tan((clampVerticalFov(vfovDeg) * Math.PI) / 360)
  return { tanV, tanH: tanV * aspect }
}

/**
 * Centimetres spanned by ONE normalised image-width unit at `distanceCm`.
 *
 * MediaPipe landmark z uses "roughly the same scale as x" and x is normalised
 * by image width, so `metricDepthCm = -landmark.z * cmPerNormalizedX(...)`.
 * (The previous occluder multiplied by a fixed ~14 cm instead, which is only
 * correct if the face filled the whole frame width; at a typical 30% face
 * width it flattens the face relief by ~3x.)
 */
export function cmPerNormalizedX(distanceCm: number, frustum: TrackerFrustum): number {
  return 2 * Math.max(distanceCm, 0) * frustum.tanH
}

/**
 * Vertical FOV (degrees) a Three.js camera must use, with aspect = surface
 * aspect, so that it projects tracker-space points onto the same surface
 * pixels as the tracker's image plane under `object-fit: cover`.
 *
 * Derivation: the surface is a centred window into the tracker's image plane.
 * Its vertical extent as a fraction of the (scaled) video height is
 * `surfaceHeight / renderedHeight`, so the visible tan(vfov/2) shrinks by that
 * factor. The horizontal FOV then follows automatically from the surface
 * aspect (see unit test).
 */
export function matchedVerticalFovDeg(
  surfaceWidth: number,
  surfaceHeight: number,
  videoWidth: number,
  videoHeight: number,
  trackerVfovDeg: number = MEDIAPIPE_FACE_GEOMETRY_VFOV_DEG,
): number {
  const cover = computeCoverMapping(surfaceWidth, surfaceHeight, videoWidth, videoHeight)
  if (!(cover.renderedHeight > 0)) return clampVerticalFov(trackerVfovDeg)

  const visibleFraction = surfaceHeight / cover.renderedHeight
  const tanV = Math.tan((clampVerticalFov(trackerVfovDeg) * Math.PI) / 360) * visibleFraction
  return (2 * Math.atan(tanV) * 180) / Math.PI
}

/**
 * Ray direction (z = -1 convention) through a landmark, in tracker camera space
 * in the app's MIRRORED display basis.
 *
 * `x`/`y` are the raw landmark values in the un-mirrored video frame, [0..1],
 * origin top-left. The display mirror is applied here (x -> 1 - x).
 */
export function landmarkRay(
  x: number,
  y: number,
  frustum: TrackerFrustum,
  out: { x: number; y: number },
): void {
  const mirroredX = 1 - x
  out.x = (mirroredX * 2 - 1) * frustum.tanH
  out.y = (1 - y * 2) * frustum.tanV
}
