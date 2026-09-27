import * as THREE from 'three'
import type { AssetGeometrySummary } from './GLBAssetAnalyzer'
import type {
  DetectedPartCandidate,
  EyewearPartDetection,
  MeshAnalysis,
  NativeOrientation,
  PartEvidence,
} from './types'

const LENS_REGEX = /lens|glass|lenses|eye|glare|crystal|front_glass/i
const BRIDGE_REGEX = /bridge|nose|pad|link|middle/i
const TEMPLE_REGEX = /temple|arm|ear|leg|hinge|side|stems/i
const LEFT_TAG = /left|_l\b|\bl_|left_/i
const RIGHT_TAG = /right|_r\b|\br_|right_/i

/**
 * Multi-evidence detector that identifies structural eyewear parts (Lenses, Bridge, Temples)
 * using a hybrid combination of semantic naming keywords and spatial geometry heuristics.
 */
export class EyewearPartDetector {
  /**
   * Detects eyewear parts and computes the optimal anchor origin point in native asset space.
   */
  public detectParts(
    summary: AssetGeometrySummary,
    orientation: NativeOrientation,
  ): EyewearPartDetection & { targetNativeOrigin: THREE.Vector3 } {
    const { meshes, overallBounds, overallDimensions } = summary
    const R_canonical = orientation.rotationMatrixToCanonical
    const R_native = R_canonical.clone().invert()

    const canonicalMeshes = meshes.map((mesh) => {
      const canonicalCenter = mesh.center.clone().applyMatrix4(R_canonical)
      const canonicalBounds = mesh.worldBounds.clone().applyMatrix4(R_canonical)
      return { mesh, canonicalCenter, canonicalBounds }
    })

    const canonicalOverallCenter = summary.overallCenter.clone().applyMatrix4(R_canonical)
    const frameWidth = overallDimensions.x > 0 ? overallDimensions.x : 10
    const frameHeight = overallDimensions.y > 0 ? overallDimensions.y : 5
    const frameDepth = overallDimensions.z > 0 ? overallDimensions.z : 5

    // --- 1. Detect Lens Candidates ---
    const detectedLenses: DetectedPartCandidate[] = []
    let leftLensMesh: (typeof canonicalMeshes)[0] | null = null
    let rightLensMesh: (typeof canonicalMeshes)[0] | null = null

    for (const item of canonicalMeshes) {
      const semanticScore = evaluateSemanticScore(item.mesh, LENS_REGEX)
      const dx = item.canonicalCenter.x - canonicalOverallCenter.x
      const isLeftCandidate = dx > 0.1 * (frameWidth / 2) || LEFT_TAG.test(item.mesh.nodeName)
      const isRightCandidate = dx < -0.1 * (frameWidth / 2) || RIGHT_TAG.test(item.mesh.nodeName)

      let confidence = 0
      let source: PartEvidence['source'] = 'geometric'
      let description = ''

      if (semanticScore > 0) {
        confidence += 0.5 + semanticScore * 0.3
        source = 'semantic'
        description = `Matched lens semantic keywords in '${item.mesh.nodeName}'`
      }

      if (item.mesh.isTransparent) {
        confidence += 0.3
        source = source === 'semantic' ? 'hybrid' : 'geometric'
        description += (description ? ' + ' : '') + 'Material has transparency'
      }

      const lateralOffset = Math.abs(dx) / (frameWidth / 2 || 1)
      if (lateralOffset > 0.15 && lateralOffset < 0.85) {
        confidence += 0.2
      }

      if (confidence >= 0.4) {
        const candidate: DetectedPartCandidate = {
          name: item.mesh.nodeName,
          center: item.mesh.center.clone(),
          bounds: item.mesh.worldBounds.clone(),
          vertexCount: item.mesh.vertexCount,
          evidence: {
            source,
            confidence: Math.min(confidence, 1.0),
            description: description || 'Spatial placement candidate',
          },
        }
        detectedLenses.push(candidate)

        if (isLeftCandidate && (!leftLensMesh || confidence > 0.5)) {
          leftLensMesh = item
        }
        if (isRightCandidate && (!rightLensMesh || confidence > 0.5)) {
          rightLensMesh = item
        }
      }
    }

    // Determine Lens Centers & Separation
    let leftLensCenter: THREE.Vector3 | null = null
    let rightLensCenter: THREE.Vector3 | null = null
    let lensSeparation: number | null = null

    if (leftLensMesh && rightLensMesh) {
      leftLensCenter = leftLensMesh.mesh.center.clone()
      rightLensCenter = rightLensMesh.mesh.center.clone()
      lensSeparation = leftLensCenter.distanceTo(rightLensCenter)
    } else {
      // Single-Mesh Lens Clustering:
      // In many commercial 3D eyewear models, both lenses are modeled as a single merged mesh
      // centered at X ≈ 0. Find the best candidate glass/lens mesh and partition its vertices
      // across the sagittal bilateral symmetry plane (X = canonicalOverallCenter.x).
      const singleMeshCandidate = findMergedLensMeshCandidate(canonicalMeshes, canonicalOverallCenter, frameWidth, frameDepth)
      if (singleMeshCandidate && singleMeshCandidate.mesh.threeMesh?.geometry) {
        const cluster = clusterMergedLensVertices(
          singleMeshCandidate.mesh.threeMesh,
          R_canonical,
          canonicalOverallCenter.x,
        )
        if (cluster) {
          leftLensCenter = cluster.leftCenter.clone().applyMatrix4(R_native)
          rightLensCenter = cluster.rightCenter.clone().applyMatrix4(R_native)
          lensSeparation = leftLensCenter.distanceTo(rightLensCenter)

          detectedLenses.push({
            name: `${singleMeshCandidate.mesh.nodeName} [Bilateral Lenses]`,
            center: singleMeshCandidate.mesh.center.clone(),
            bounds: singleMeshCandidate.mesh.worldBounds.clone(),
            vertexCount: singleMeshCandidate.mesh.vertexCount,
            evidence: {
              source: 'hybrid',
              confidence: 0.95,
              description: 'Single-mesh bilateral vertex clustering',
            },
          })
        }
      }
    }

    // --- 2. Detect Bridge Candidate ---
    let detectedBridge: DetectedPartCandidate | null = null
    let bridgeCenter: THREE.Vector3 | null = null

    for (const item of canonicalMeshes) {
      const semanticScore = evaluateSemanticScore(item.mesh, BRIDGE_REGEX)
      const dx = Math.abs(item.canonicalCenter.x - canonicalOverallCenter.x)
      const isCentral = dx < 0.25 * (frameWidth / 2)

      if (semanticScore > 0 || isCentral) {
        const confidence = semanticScore > 0 ? 0.8 : 0.4
        detectedBridge = {
          name: item.mesh.nodeName,
          center: item.mesh.center.clone(),
          bounds: item.mesh.worldBounds.clone(),
          vertexCount: item.mesh.vertexCount,
          evidence: {
            source: semanticScore > 0 ? 'semantic' : 'geometric',
            confidence,
            description: semanticScore > 0 ? 'Matched bridge semantic keyword' : 'Central bridge geometry',
          },
        }
        bridgeCenter = item.mesh.center.clone()
        break
      }
    }

    // Fallback for bridge center: midpoint between detected lenses
    if (!bridgeCenter && leftLensCenter && rightLensCenter) {
      bridgeCenter = new THREE.Vector3().addVectors(leftLensCenter, rightLensCenter).multiplyScalar(0.5)
    }

    // --- 3. Detect Temple Candidates (Optional / Soft Requirement) ---
    const detectedTemples: DetectedPartCandidate[] = []
    let leftTempleStart: THREE.Vector3 | null = null
    let rightTempleStart: THREE.Vector3 | null = null

    for (const item of canonicalMeshes) {
      const semanticScore = evaluateSemanticScore(item.mesh, TEMPLE_REGEX)
      if (semanticScore > 0) {
        const candidate: DetectedPartCandidate = {
          name: item.mesh.nodeName,
          center: item.mesh.center.clone(),
          bounds: item.mesh.worldBounds.clone(),
          vertexCount: item.mesh.vertexCount,
          evidence: {
            source: 'semantic',
            confidence: 0.75,
            description: `Matched temple keyword in '${item.mesh.nodeName}'`,
          },
        }
        detectedTemples.push(candidate)

        const dx = item.canonicalCenter.x - canonicalOverallCenter.x
        if (dx > 0 && !leftTempleStart) {
          leftTempleStart = item.mesh.center.clone()
        } else if (dx < 0 && !rightTempleStart) {
          rightTempleStart = item.mesh.center.clone()
        }
      }
    }

    // --- 4. Fallback Chain for Native Anchor Origin ---
    let targetNativeOrigin = new THREE.Vector3()
    let anchorOriginSource: EyewearPartDetection['anchorOriginSource'] = 'bounding_box_center'

    if (leftLensCenter && rightLensCenter) {
      targetNativeOrigin.addVectors(leftLensCenter, rightLensCenter).multiplyScalar(0.5)
      anchorOriginSource = 'lens_midpoint'
    } else if (bridgeCenter) {
      targetNativeOrigin.copy(bridgeCenter)
      anchorOriginSource = 'bridge_center'
    } else if (summary.overallCenter) {
      targetNativeOrigin.copy(summary.overallCenter)
      anchorOriginSource = 'bounding_box_center'
    }

    return {
      leftLensCenter,
      rightLensCenter,
      bridgeCenter,
      leftTempleStart,
      rightTempleStart,
      lensSeparation,
      frameWidth,
      frameHeight,
      frameDepth,
      detectedLenses: Object.freeze(detectedLenses),
      detectedBridge,
      detectedTemples: Object.freeze(detectedTemples),
      anchorOriginSource,
      targetNativeOrigin,
    }
  }
}

/**
 * Helper to compute semantic match score against keywords in node and material names.
 */
function evaluateSemanticScore(mesh: MeshAnalysis, regex: RegExp): number {
  let score = 0
  if (regex.test(mesh.nodeName)) score += 1.0
  if (mesh.parentName && regex.test(mesh.parentName)) score += 0.5
  for (const matName of mesh.materialNames) {
    if (regex.test(matName)) score += 0.7
  }
  return score
}

function findMergedLensMeshCandidate(
  canonicalMeshes: Array<{ mesh: MeshAnalysis; canonicalCenter: THREE.Vector3; canonicalBounds: THREE.Box3 }>,
  canonicalOverallCenter: THREE.Vector3,
  frameWidth: number,
  frameDepth: number,
) {
  let bestCandidate: (typeof canonicalMeshes)[0] | null = null
  let highestScore = -1

  for (const item of canonicalMeshes) {
    const semanticScore = evaluateSemanticScore(item.mesh, LENS_REGEX)
    const bounds = item.canonicalBounds
    const spansCenter = bounds.min.x < canonicalOverallCenter.x && bounds.max.x > canonicalOverallCenter.x
    const width = bounds.max.x - bounds.min.x
    const depth = bounds.max.z - bounds.min.z

    let score = 0
    if (semanticScore > 0) score += semanticScore * 2.0
    if (item.mesh.isTransparent) score += 1.5
    if (spansCenter) score += 1.0
    if (width > 0.4 * frameWidth) score += 1.0
    if (depth < 0.4 * frameDepth) score += 0.8
    if (item.canonicalCenter.z > canonicalOverallCenter.z - 0.2 * frameDepth) score += 0.5

    if (spansCenter && score > highestScore) {
      highestScore = score
      bestCandidate = item
    }
  }

  return highestScore >= 1.5 ? bestCandidate : null
}

function clusterMergedLensVertices(
  mesh: THREE.Mesh,
  R_canonical: THREE.Matrix4,
  centerX: number,
): { leftCenter: THREE.Vector3; rightCenter: THREE.Vector3 } | null {
  const geom = mesh.geometry
  if (!geom) return null
  const pos = geom.getAttribute('position') as THREE.BufferAttribute | undefined
  if (!pos || pos.count === 0) return null

  mesh.updateMatrixWorld(true)
  const fullCanonicalMatrix = new THREE.Matrix4().multiplyMatrices(R_canonical, mesh.matrixWorld)

  const v = new THREE.Vector3()
  const leftSum = new THREE.Vector3()
  let leftCount = 0
  const rightSum = new THREE.Vector3()
  let rightCount = 0

  const step = Math.max(1, Math.floor(pos.count / 1000))
  for (let i = 0; i < pos.count; i += step) {
    v.fromBufferAttribute(pos, i).applyMatrix4(fullCanonicalMatrix)
    if (v.x < centerX) {
      leftSum.add(v)
      leftCount += 1
    } else {
      rightSum.add(v)
      rightCount += 1
    }
  }

  if (leftCount === 0 || rightCount === 0) return null

  return {
    leftCenter: leftSum.divideScalar(leftCount),
    rightCenter: rightSum.divideScalar(rightCount),
  }
}

