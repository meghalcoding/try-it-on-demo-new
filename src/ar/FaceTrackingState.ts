import type { FaceLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision'

export type TrackingState = 'searching' | 'detected' | 'lost'

export interface FaceBounds {
  readonly minX: number
  readonly minY: number
  readonly maxX: number
  readonly maxY: number
  readonly width: number
  readonly height: number
  readonly centerX: number
  readonly centerY: number
  readonly area: number
}

export interface PrimaryFaceSelection {
  readonly state: TrackingState
  readonly primaryFaceIndex: number | null
  readonly bounds: FaceBounds | null
}

const MATCH_DISTANCE_THRESHOLD = 0.18
const MATCH_IOU_THRESHOLD = 0.2

/**
 * Keeps the selected face stable across result frames.
 *
 * Selection is based only on landmark-derived image bounds for identity of the
 * detected face within the current frame. It does not reconstruct head pose;
 * the MediaPipe transformation matrix remains the pose source.
 */
export class FaceTrackingState {
  private state: TrackingState = 'searching'
  private primaryFaceIndex: number | null = null
  private lastPrimaryBounds: FaceBounds | null = null

  update(result: FaceLandmarkerResult): PrimaryFaceSelection {
    if (result.faceLandmarks.length === 0) {
      this.state = this.primaryFaceIndex === null ? 'searching' : 'lost'
      return this.snapshot()
    }

    if (this.primaryFaceIndex !== null && this.lastPrimaryBounds) {
      const matchIndex = this.findMatchingFace(result.faceLandmarks, this.lastPrimaryBounds)

      if (matchIndex !== null) {
        this.primaryFaceIndex = matchIndex
        this.lastPrimaryBounds = getFaceBounds(result.faceLandmarks[matchIndex])
        this.state = 'detected'
        return this.snapshot()
      }

      this.state = 'lost'
      this.primaryFaceIndex = null
      this.lastPrimaryBounds = null
      return this.snapshot()
    }

    const newPrimaryIndex = selectLargestFace(result.faceLandmarks)
    this.primaryFaceIndex = newPrimaryIndex
    this.lastPrimaryBounds = getFaceBounds(result.faceLandmarks[newPrimaryIndex])
    this.state = 'detected'

    return this.snapshot()
  }

  reset(): void {
    this.state = 'searching'
    this.primaryFaceIndex = null
    this.lastPrimaryBounds = null
  }

  getState(): TrackingState {
    return this.state
  }

  getPrimaryFaceIndex(): number | null {
    return this.primaryFaceIndex
  }

  getLastPrimaryBounds(): FaceBounds | null {
    return this.lastPrimaryBounds
  }

  private findMatchingFace(
    faces: readonly NormalizedLandmark[][],
    previousBounds: FaceBounds,
  ): number | null {
    let bestIndex: number | null = null
    let bestScore = Number.NEGATIVE_INFINITY

    faces.forEach((landmarks, index) => {
      const bounds = getFaceBounds(landmarks)
      const iou = intersectionOverUnion(previousBounds, bounds)
      const distance = Math.hypot(
        previousBounds.centerX - bounds.centerX,
        previousBounds.centerY - bounds.centerY,
      )

      if (iou < MATCH_IOU_THRESHOLD && distance > MATCH_DISTANCE_THRESHOLD) {
        return
      }

      const score = iou * 2 + (1 - Math.min(distance / MATCH_DISTANCE_THRESHOLD, 1))

      if (score > bestScore) {
        bestScore = score
        bestIndex = index
      }
    })

    return bestIndex
  }

  private snapshot(): PrimaryFaceSelection {
    return {
      state: this.state,
      primaryFaceIndex: this.primaryFaceIndex,
      bounds: this.lastPrimaryBounds,
    }
  }
}

export function getFaceBounds(landmarks: readonly NormalizedLandmark[]): FaceBounds {
  if (landmarks.length === 0) {
    throw new Error('Cannot calculate face bounds without landmarks.')
  }

  let minX = 1
  let minY = 1
  let maxX = 0
  let maxY = 0

  for (const landmark of landmarks) {
    minX = Math.min(minX, landmark.x)
    minY = Math.min(minY, landmark.y)
    maxX = Math.max(maxX, landmark.x)
    maxY = Math.max(maxY, landmark.y)
  }

  const width = maxX - minX
  const height = maxY - minY

  return {
    minX,
    minY,
    maxX,
    maxY,
    width,
    height,
    centerX: minX + width / 2,
    centerY: minY + height / 2,
    area: width * height,
  }
}

function selectLargestFace(faces: readonly NormalizedLandmark[][]): number {
  let largestIndex = 0
  let largestArea = Number.NEGATIVE_INFINITY

  faces.forEach((landmarks, index) => {
    const area = getFaceBounds(landmarks).area

    if (area > largestArea) {
      largestArea = area
      largestIndex = index
    }
  })

  return largestIndex
}

function intersectionOverUnion(a: FaceBounds, b: FaceBounds): number {
  const intersectionMinX = Math.max(a.minX, b.minX)
  const intersectionMinY = Math.max(a.minY, b.minY)
  const intersectionMaxX = Math.min(a.maxX, b.maxX)
  const intersectionMaxY = Math.min(a.maxY, b.maxY)
  const intersectionWidth = Math.max(0, intersectionMaxX - intersectionMinX)
  const intersectionHeight = Math.max(0, intersectionMaxY - intersectionMinY)
  const intersectionArea = intersectionWidth * intersectionHeight
  const unionArea = a.area + b.area - intersectionArea

  return unionArea > 0 ? intersectionArea / unionArea : 0
}
