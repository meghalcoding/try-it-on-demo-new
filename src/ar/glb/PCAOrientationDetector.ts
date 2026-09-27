import * as THREE from 'three'
import type { NativeOrientation, PrimaryAxis } from './types'

export interface PCAAnalysisResult {
  readonly eigenvalues: readonly [number, number, number]
  readonly eigenvectors: readonly [THREE.Vector3, THREE.Vector3, THREE.Vector3]
  readonly extents: readonly [number, number, number]
  readonly centroid: THREE.Vector3
}

/**
 * Performs Principal Component Analysis (PCA) over 3D vertex point clouds to determine
 * native coordinate axes (Width, Height, Depth), coordinate handedness, and the rotation
 * matrix required to orient the asset into Canonical Eyewear Space.
 */
export class PCAOrientationDetector {
  /**
   * Analyzes a contiguous 3D point cloud (x,y,z,x,y,z...) and overall bounding box
   * using hybrid PCA and Bilateral Reflection Symmetry to determine native orientation axes
   * and the exact transformation matrix to Canonical Eyewear Space.
   */
  public detectOrientation(
    pointCloud: Float32Array,
    overallBounds: THREE.Box3,
  ): NativeOrientation {
    if (pointCloud.length < 9) {
      return fallbackOrientationFromBounds(overallBounds)
    }

    const pca = computePointCloudPCA(pointCloud)
    const { eigenvectors, extents, centroid } = pca

    // Eyewear geometry invariant:
    // Eyeglasses possess strong Bilateral Reflection Symmetry across the sagittal plane (Width axis).
    // Temple arms (Depth axis) and frame height (Height axis) possess virtually zero reflection symmetry.
    // We compute the reflection symmetry error for each principal eigenvector:
    const symmetryScores = [0, 1, 2].map((idx) => {
      const vec = eigenvectors[idx].clone().normalize()
      const symError = computeBilateralSymmetryError(pointCloud, centroid, vec)
      return { idx, vec, symError, extent: extents[idx] }
    })

    // Sort by symmetry error: smallest error is the lateral (Width) axis
    symmetryScores.sort((a, b) => a.symError - b.symError)

    const widthIdx = symmetryScores[0].idx
    const remaining = [symmetryScores[1], symmetryScores[2]]

    // Between remaining two axes, Frame Height is consistently smaller than Temple Length (Depth)
    remaining.sort((a, b) => a.extent - b.extent)
    const heightIdx = remaining[0].idx
    let widthVec = eigenvectors[widthIdx].clone().normalize()
    let heightVec = eigenvectors[heightIdx].clone().normalize()

    // PCA eigenvectors are sign-ambiguous: v and -v are equally valid eigenvectors.
    // The previous implementation tried to resolve depth with a sum of centered
    // projections. That sum is mathematically ~0 by construction, so floating-point
    // noise randomly flipped the depth axis and, to preserve handedness, flipped the
    // height axis with it. This produced the exact per-model Y/Z inversions seen in
    // the live catalog.
    //
    // Establish the two semantic axes first: lateral width points toward +X and frame
    // height points toward +Y. Depth is then derived from their right-handed cross
    // product. This removes the invalid center-of-mass depth test entirely.
    if (widthVec.x < 0) {
      widthVec.negate()
    }

    // Keep the PCA axes orthogonal after any sign correction, then orient height upward.
    heightVec.addScaledVector(widthVec, -heightVec.dot(widthVec)).normalize()
    if (heightVec.y < 0) {
      heightVec.negate()
    }

    let depthVec = new THREE.Vector3().crossVectors(widthVec, heightVec).normalize()

    // Many supplied GLBs are already authored in the canonical X/Y/Z orientation.
    // PCA can still rotate those models several degrees because the mesh mass is not
    // an ellipsoid. When both semantic axes are clearly aligned to world X/Y, snap
    // them to the authored axes instead of introducing a spurious corrective rotation.
    // Non-native/rotated assets do not satisfy this threshold and continue through the
    // PCA-derived basis above.
    if (Math.abs(widthVec.x) >= 0.90 && Math.abs(heightVec.y) >= 0.90) {
      widthVec.set(1, 0, 0)
      heightVec.set(0, 1, 0)
      depthVec.set(0, 0, 1)
    }

    const widthAxis = vectorToPrimaryAxis(widthVec)
    const heightAxis = vectorToPrimaryAxis(heightVec)
    const depthAxis = vectorToPrimaryAxis(depthVec)

    // Construct rotation matrix mapping native principal axes to canonical basis:
    // Canonical basis: X_canonical = (1,0,0), Y_canonical = (0,1,0), Z_canonical = (0,0,1)
    const R_native = new THREE.Matrix4().makeBasis(widthVec, heightVec, depthVec)
    const rotationMatrixToCanonical = R_native.clone().invert()

    const det = R_native.determinant()
    const isRightHanded = det > 0

    return {
      widthAxis,
      heightAxis,
      depthAxis,
      isRightHanded,
      pcaEigenvalues: Object.freeze(pca.eigenvalues),
      pcaEigenvectors: Object.freeze([widthVec, heightVec, depthVec]),
      rotationMatrixToCanonical,
    }
  }
}

/**
 * Computes 3x3 Covariance matrix and solves exact eigenvalues/eigenvectors using Jacobi algorithm.
 */
function computePointCloudPCA(pointCloud: Float32Array): PCAAnalysisResult {
  const count = pointCloud.length / 3

  // 1. Centroid calculation
  let meanX = 0
  let meanY = 0
  let meanZ = 0

  for (let i = 0; i < pointCloud.length; i += 3) {
    meanX += pointCloud[i]
    meanY += pointCloud[i + 1]
    meanZ += pointCloud[i + 2]
  }

  meanX /= count
  meanY /= count
  meanZ /= count

  // 2. Covariance matrix elements (C_xx, C_xy, C_xz, C_yy, C_yz, C_zz)
  let cxx = 0
  let cxy = 0
  let cxz = 0
  let cyy = 0
  let cyz = 0
  let czz = 0

  for (let i = 0; i < pointCloud.length; i += 3) {
    const dx = pointCloud[i] - meanX
    const dy = pointCloud[i + 1] - meanY
    const dz = pointCloud[i + 2] - meanZ

    cxx += dx * dx
    cxy += dx * dy
    cxz += dx * dz
    cyy += dy * dy
    cyz += dy * dz
    czz += dz * dz
  }

  cxx /= count
  cxy /= count
  cxz /= count
  cyy /= count
  cyz /= count
  czz /= count

  const covariance = [
    [cxx, cxy, cxz],
    [cxy, cyy, cyz],
    [cxz, cyz, czz],
  ]

  // 3. Jacobi Eigenvalue Solver for 3x3 Symmetric Matrix
  const { eigenvalues, eigenvectors } = solveJacobiSymmetric3x3(covariance)

  // 4. Measure point cloud extent along each eigenvector
  const extents: [number, number, number] = [0, 0, 0]
  for (let k = 0; k < 3; k += 1) {
    const vec = eigenvectors[k]
    let minProj = Infinity
    let maxProj = -Infinity

    for (let i = 0; i < pointCloud.length; i += 3) {
      const proj = pointCloud[i] * vec.x + pointCloud[i + 1] * vec.y + pointCloud[i + 2] * vec.z
      minProj = Math.min(minProj, proj)
      maxProj = Math.max(maxProj, proj)
    }

    extents[k] = maxProj - minProj
  }

  const centroid = new THREE.Vector3(meanX, meanY, meanZ)
  return { eigenvalues, eigenvectors, extents, centroid }
}

/**
 * Measures bilateral reflection symmetry error of the point cloud across a mirror plane
 * passing through centroid with the given normal vector.
 * Returns a normalized error metric in [0, 1], where ~0 represents perfect mirror symmetry.
 */
function computeBilateralSymmetryError(
  pointCloud: Float32Array,
  centroid: THREE.Vector3,
  normal: THREE.Vector3,
): number {
  let posCount = 0
  let negCount = 0
  const totalCount = pointCloud.length / 3

  for (let i = 0; i < pointCloud.length; i += 3) {
    const dx = pointCloud[i] - centroid.x
    const dy = pointCloud[i + 1] - centroid.y
    const dz = pointCloud[i + 2] - centroid.z
    const proj = dx * normal.x + dy * normal.y + dz * normal.z
    if (proj > 0) {
      posCount += 1
    } else {
      negCount += 1
    }
  }

  return Math.abs(posCount - negCount) / (totalCount || 1)
}

/**
 * Jacobi Eigenvalue Algorithm for a 3x3 symmetric matrix.
 * Solves C * v = lambda * v with high numerical stability.
 */
function solveJacobiSymmetric3x3(matrix: number[][]): {
  eigenvalues: [number, number, number]
  eigenvectors: [THREE.Vector3, THREE.Vector3, THREE.Vector3]
} {
  const A = matrix.map((row) => [...row])
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]

  const MAX_ITERATIONS = 50
  const EPSILON = 1e-12

  for (let iter = 0; iter < MAX_ITERATIONS; iter += 1) {
    // Find largest off-diagonal element
    let p = 0
    let q = 1
    let maxOffDiag = Math.abs(A[0][1])

    if (Math.abs(A[0][2]) > maxOffDiag) {
      p = 0
      q = 2
      maxOffDiag = Math.abs(A[0][2])
    }
    if (Math.abs(A[1][2]) > maxOffDiag) {
      p = 1
      q = 2
      maxOffDiag = Math.abs(A[1][2])
    }

    if (maxOffDiag < EPSILON) {
      break
    }

    // Compute Jacobi rotation angle
    const app = A[p][p]
    const aqq = A[q][q]
    const apq = A[p][q]

    const phi = 0.5 * Math.atan2(2 * apq, aqq - app)
    const c = Math.cos(phi)
    const s = Math.sin(phi)

    // Rotate matrix A: A' = J^T * A * J
    A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
    A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
    A[p][q] = 0
    A[q][p] = 0

    for (let i = 0; i < 3; i += 1) {
      if (i !== p && i !== q) {
        const aip = A[i][p]
        const aiq = A[i][q]
        A[i][p] = c * aip - s * aiq
        A[p][i] = A[i][p]
        A[i][q] = s * aip + c * aiq
        A[q][i] = A[i][q]
      }
    }

    // Accumulate eigenvectors in V
    for (let i = 0; i < 3; i += 1) {
      const vip = V[i][p]
      const viq = V[i][q]
      V[i][p] = c * vip - s * viq
      V[i][q] = s * vip + c * viq
    }
  }

  const vec0 = new THREE.Vector3(V[0][0], V[1][0], V[2][0]).normalize()
  const vec1 = new THREE.Vector3(V[0][1], V[1][1], V[2][1]).normalize()
  const vec2 = new THREE.Vector3(V[0][2], V[1][2], V[2][2]).normalize()

  return {
    eigenvalues: [A[0][0], A[1][1], A[2][2]],
    eigenvectors: [vec0, vec1, vec2],
  }
}

/**
 * Maps a continuous 3D unit vector to its closest discrete signed axis ('X', 'Y', 'Z', '-X', '-Y', '-Z').
 */
function vectorToPrimaryAxis(v: THREE.Vector3): PrimaryAxis {
  const ax = Math.abs(v.x)
  const ay = Math.abs(v.y)
  const az = Math.abs(v.z)

  if (ax >= ay && ax >= az) {
    return v.x >= 0 ? 'X' : '-X'
  }
  if (ay >= ax && ay >= az) {
    return v.y >= 0 ? 'Y' : '-Y'
  }
  return v.z >= 0 ? 'Z' : '-Z'
}

/**
 * Fallback orientation provider when point cloud is empty or insufficient.
 */
function fallbackOrientationFromBounds(bounds: THREE.Box3): NativeOrientation {
  const size = new THREE.Vector3()
  bounds.getSize(size)

  const dims = [
    { axis: 'X' as PrimaryAxis, extent: size.x },
    { axis: 'Y' as PrimaryAxis, extent: size.y },
    { axis: 'Z' as PrimaryAxis, extent: size.z },
  ].sort((a, b) => b.extent - a.extent)

  return {
    widthAxis: dims[0].axis,
    heightAxis: dims[1].axis,
    depthAxis: dims[2].axis,
    isRightHanded: true,
    pcaEigenvalues: [dims[0].extent, dims[1].extent, dims[2].extent],
    pcaEigenvectors: [
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 1),
    ],
    rotationMatrixToCanonical: new THREE.Matrix4().identity(),
  }
}
