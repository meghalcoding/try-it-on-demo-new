/**
 * Product calibration contract.
 *
 * All values are numeric. Translation is expressed in canonical face-local
 * coordinates, scale is a positive multiplier, and rotation values are in
 * radians. No product-specific or UI-specific properties belong here.
 */
export interface Calibration {
  readonly scale: number
  readonly x: number
  readonly y: number
  readonly z: number
  readonly rotationX: number
  readonly rotationY: number
  readonly rotationZ: number
}
