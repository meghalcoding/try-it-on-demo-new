import type * as THREE from 'three'
import type { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

export type ArDependencyCheck = {
  three: typeof THREE
  mediapipe: {
    FaceLandmarker: typeof FaceLandmarker
    FilesetResolver: typeof FilesetResolver
  }
}
