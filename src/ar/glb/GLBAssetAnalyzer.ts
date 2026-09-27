import * as THREE from 'three'
import type { EyewearAssetProfile, GLBAssetAnalysis, MeshAnalysis, NodeAnalysis } from './types'
import { PCAOrientationDetector } from './PCAOrientationDetector'
import { EyewearPartDetector } from './EyewearPartDetector'
import { CanonicalNormalizer } from './CanonicalNormalizer'

export interface AssetGeometrySummary {
  readonly meshes: readonly MeshAnalysis[]
  readonly nodes: readonly NodeAnalysis[]
  readonly overallBounds: THREE.Box3
  readonly overallCenter: THREE.Vector3
  readonly overallDimensions: THREE.Vector3
  readonly totalVertices: number
  readonly totalTriangles: number
}

/**
 * Performs complete, non-destructive world-space geometry traversal over a GLB scene graph.
 * Computes exact world-space bounding boxes, centroids, vertex metrics, and material properties.
 */
export class GLBAssetAnalyzer {
  private readonly pcaDetector = new PCAOrientationDetector()
  private readonly partDetector = new EyewearPartDetector()
  private readonly normalizer = new CanonicalNormalizer()

  /**
   * Orchestrates the complete GLB analysis pipeline:
   * 1. Geometry & Bounds Analysis
   * 2. PCA Orientation Detection
   * 3. Multi-evidence Part & Anchor Origin Detection
   * 4. Canonical Space Normalization
   * 5. EyewearAssetProfile Generation
   */
  public analyzeAsset(root: THREE.Object3D): GLBAssetAnalysis {
    const summary = this.analyzeGeometry(root)
    const pointCloud = this.extractWorldPointCloud(root)
    const orientation = this.pcaDetector.detectOrientation(pointCloud, summary.overallBounds)
    const partsWithOrigin = this.partDetector.detectParts(summary, orientation)
    const { targetNativeOrigin, ...parts } = partsWithOrigin

    const normalization = this.normalizer.normalize(root, orientation, targetNativeOrigin)

    const profile: EyewearAssetProfile = {
      canonicalWidth: normalization.canonicalDimensions.x,
      canonicalHeight: normalization.canonicalDimensions.y,
      canonicalDepth: normalization.canonicalDimensions.z,
      leftLensCenter: parts.leftLensCenter ? parts.leftLensCenter.clone() : null,
      rightLensCenter: parts.rightLensCenter ? parts.rightLensCenter.clone() : null,
      bridgeCenter: parts.bridgeCenter ? parts.bridgeCenter.clone() : null,
      lensSeparation: parts.lensSeparation,
      leftTempleStart: parts.leftTempleStart ? parts.leftTempleStart.clone() : null,
      rightTempleStart: parts.rightTempleStart ? parts.rightTempleStart.clone() : null,
      nativeBounds: summary.overallBounds.clone(),
      nativeCenter: summary.overallCenter.clone(),
      normalizationTransform: normalization.normalizationMatrix.clone(),
      parts,
      orientation,
    }

    return {
      summary,
      orientation,
      parts,
      profile,
      normalization,
    }
  }

  /**
   * Analyzes every Object3D node and Mesh within the provided GLTF scene root.
   */
  public analyzeGeometry(root: THREE.Object3D): AssetGeometrySummary {
    if (!root) {
      throw new Error('GLBAssetAnalyzer requires a valid THREE.Object3D root.')
    }

    // Force complete hierarchy matrix update so world transforms reflect true asset state
    root.updateMatrixWorld(true)

    const meshes: MeshAnalysis[] = []
    const nodes: NodeAnalysis[] = []
    const overallBounds = new THREE.Box3()
    let totalVertices = 0
    let totalTriangles = 0

    root.traverse((object) => {
      const parentName = object.parent ? object.parent.name || object.parent.type : null

      // Record Node Analysis for every scene graph object
      const childrenNames = object.children.map((child) => child.name || child.type)
      const nodeWorldBounds = computeObjectWorldBounds(object)

      nodes.push({
        name: object.name || object.type,
        parentName,
        childrenNames,
        isMesh: object instanceof THREE.Mesh,
        worldTransform: object.matrixWorld.clone(),
        worldBounds: nodeWorldBounds,
      })

      if (object instanceof THREE.Mesh) {
        const meshAnalysis = analyzeMesh(object, parentName)
        if (meshAnalysis) {
          meshes.push(meshAnalysis)
          overallBounds.union(meshAnalysis.worldBounds)
          totalVertices += meshAnalysis.vertexCount
          totalTriangles += meshAnalysis.triangleCount
        }
      }
    })

    const overallCenter = new THREE.Vector3()
    const overallDimensions = new THREE.Vector3()

    if (!overallBounds.isEmpty()) {
      overallBounds.getCenter(overallCenter)
      overallBounds.getSize(overallDimensions)
    }

    return {
      meshes: Object.freeze(meshes),
      nodes: Object.freeze(nodes),
      overallBounds,
      overallCenter,
      overallDimensions,
      totalVertices,
      totalTriangles,
    }
  }

  /**
   * Extracts a single contiguous Float32Array containing all world-space vertex coordinates
   * across all meshes in the scene. Useful for PCA orientation and spatial point-cloud analysis.
   */
  public extractWorldPointCloud(root: THREE.Object3D): Float32Array {
    root.updateMatrixWorld(true)

    const meshVertices: THREE.Vector3[] = []
    const tempVertex = new THREE.Vector3()

    root.traverse((object) => {
      if (!(object instanceof THREE.Mesh) || !object.geometry) {
        return
      }

      const positionAttr = object.geometry.getAttribute('position') as THREE.BufferAttribute | undefined
      if (!positionAttr) {
        return
      }

      const matrixWorld = object.matrixWorld

      for (let i = 0; i < positionAttr.count; i += 1) {
        tempVertex.fromBufferAttribute(positionAttr, i)
        tempVertex.applyMatrix4(matrixWorld)
        meshVertices.push(tempVertex.clone())
      }
    })

    const pointCloud = new Float32Array(meshVertices.length * 3)
    for (let i = 0; i < meshVertices.length; i += 1) {
      const v = meshVertices[i]
      pointCloud[i * 3] = v.x
      pointCloud[i * 3 + 1] = v.y
      pointCloud[i * 3 + 2] = v.z
    }

    return pointCloud
  }
}

/**
 * Analyzes an individual Three.js Mesh.
 */
function analyzeMesh(mesh: THREE.Mesh, parentName: string | null): MeshAnalysis | null {
  const geometry = mesh.geometry
  if (!geometry) {
    return null
  }

  const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute | undefined
  if (!positionAttr || positionAttr.count === 0) {
    return null
  }

  // Ensure local bounding box exists
  if (!geometry.boundingBox) {
    geometry.computeBoundingBox()
  }

  const localBounds = geometry.boundingBox ? geometry.boundingBox.clone() : new THREE.Box3()
  const worldBounds = computeObjectWorldBounds(mesh)

  const center = new THREE.Vector3()
  const dimensions = new THREE.Vector3()
  worldBounds.getCenter(center)
  worldBounds.getSize(dimensions)

  const vertexCount = positionAttr.count
  const indexAttr = geometry.getIndex()
  const triangleCount = indexAttr ? indexAttr.count / 3 : vertexCount / 3

  const materialNames: string[] = []
  let isTransparent = false

  const meshMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
  for (const mat of meshMaterials) {
    if (mat) {
      if (mat.name) materialNames.push(mat.name)
      if (mat.transparent || mat.opacity < 0.99) {
        isTransparent = true
      }
    }
  }

  return {
    nodeName: mesh.name || 'UnnamedMesh',
    parentName,
    vertexCount,
    triangleCount: Math.floor(triangleCount),
    localBounds,
    worldBounds,
    center,
    dimensions,
    materialNames: Object.freeze(materialNames),
    materialCount: meshMaterials.length,
    isTransparent,
    worldTransform: mesh.matrixWorld.clone(),
    threeMesh: mesh,
  }
}

/**
 * Computes exact world-space bounding box for an Object3D by transforming its bounding box corners.
 */
function computeObjectWorldBounds(object: THREE.Object3D): THREE.Box3 {
  const box = new THREE.Box3()

  if (object instanceof THREE.Mesh && object.geometry) {
    if (!object.geometry.boundingBox) {
      object.geometry.computeBoundingBox()
    }
    if (object.geometry.boundingBox) {
      box.copy(object.geometry.boundingBox).applyMatrix4(object.matrixWorld)
      return box
    }
  }

  // For non-mesh nodes or fallback, expand box to include children
  box.setFromObject(object)
  return box
}
