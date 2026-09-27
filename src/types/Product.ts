import type { Calibration } from './Calibration'

export interface Product {
  readonly id: string
  readonly name: string
  readonly price: number
  readonly currency: string
  readonly thumbnail: string
  readonly model: string
  readonly color: string
  readonly size: string
  readonly calibration: Calibration
}
