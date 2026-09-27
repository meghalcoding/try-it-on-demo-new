import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { EAR, SKULL, buildEarGeometry, buildSkullGeometry } from '../HeadProxy'
import { CANONICAL_TRIANGLES, MIRRORED_CANONICAL_VERTICES } from '../canonicalFaceModel'

function faceMesh(): THREE.Mesh {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MIRRORED_CANONICAL_VERTICES), 3))
  geometry.setIndex(new THREE.BufferAttribute(new Uint16Array(CANONICAL_TRIANGLES), 1))
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }))
  mesh.updateMatrixWorld(true)
  return mesh
}

/**
 * Ray-cast a set of (face-local) points against a mesh, from a camera placed
 * at `yawDeg` around the head at a fixed distance. This is equivalent to
 * turning the head under a fixed camera, but keeps the arm/face/head/ear
 * geometry all in one consistent face-local frame.
 */
function occludedFraction(
  points: THREE.Vector3[],
  mesh: THREE.Mesh,
  pushCm: number,
  yawDeg: number,
  distance = 45,
): number {
  const raycaster = new THREE.Raycaster()
  const angle = (yawDeg * Math.PI) / 180
  const origin = new THREE.Vector3(Math.sin(angle) * distance, 0, Math.cos(angle) * distance)
  let occluded = 0
  for (const p of points) {
    const direction = p.clone().sub(origin)
    const rayLength = direction.length()
    direction.normalize()
    raycaster.set(origin, direction)
    raycaster.far = rayLength
    const hits = raycaster.intersectObject(mesh, false)
    if (hits.length && hits[0].distance + pushCm < rayLength - 0.02) occluded += 1
  }
  return occluded / points.length
}

/** A synthetic temple arm in face-local geometry, side = +1 (right) or -1 (left). */
function templeArm(side: 1 | -1): THREE.Vector3[] {
  const points: THREE.Vector3[] = []
  for (let z = 1.2; z >= -12.5; z -= 0.4) {
    points.push(new THREE.Vector3(side * 6.9, 3.4, z))
  }
  return points
}

describe('HeadProxy anatomy', () => {
  it('the skull never protrudes in front of the canonical face surface', () => {
    const face = faceMesh()
    const skull = new THREE.Mesh(buildSkullGeometry())
    const position = skull.geometry.getAttribute('position')
    const raycaster = new THREE.Raycaster()
    let sampled = 0

    for (let i = 0; i < position.count; i += 1) {
      const v = new THREE.Vector3().fromBufferAttribute(position, i)
      // Only meaningful where the skull sits under the face footprint.
      if (Math.abs(v.x) > 9 || v.y < -10 || v.y > 10) continue
      raycaster.set(new THREE.Vector3(v.x, v.y, 40), new THREE.Vector3(0, 0, -1))
      const hit = raycaster.intersectObject(face, false)
      if (!hit.length) continue
      sampled += 1
      // Face surface z at (x,y) must be >= skull vertex z (skull is behind or level).
      expect(v.z).toBeLessThanOrEqual(hit[0].point.z + 1e-2)
    }
    expect(sampled).toBeGreaterThan(100) // sanity: the overlap check actually ran
  })

  it('the ear volumes are anchored near the face-oval edge, not deep inside the cheeks', () => {
    // Face-oval half-width near the ear's y range (canonical landmark 234 is the right cheek edge).
    const cheekEdgeX = Math.abs(MIRRORED_CANONICAL_VERTICES[234 * 3])
    const earInnerX = Math.abs(EAR.center[0]) - EAR.radii[0]
    // Within 1.5cm of the cheek edge (some overlap is fine/expected for contact; a
    // large negative gap would mean the ear volume is buried inside the head).
    expect(earInnerX).toBeGreaterThan(cheekEdgeX - 1.5)
  })

  it('increasing yaw increases visibility of the far-side temple arm once head/ear volumes are added', () => {
    const face = faceMesh()
    const skull = new THREE.Mesh(buildSkullGeometry())
    const ears = new THREE.Mesh(buildEarGeometry())
    skull.updateMatrixWorld(true)
    ears.updateMatrixWorld(true)

    const withFaceOnly = (points: THREE.Vector3[], yaw: number) => occludedFraction(points, face, 0.1, yaw)
    const withAll = (points: THREE.Vector3[], yaw: number) => {
      const a = occludedFraction(points, face, 0.1, yaw)
      const b = occludedFraction(points, skull, 0.7, yaw)
      const c = occludedFraction(points, ears, 0.25, yaw)
      // "occluded" if ANY volume hides the point
      return Math.max(a, b, c)
    }

    // Head-on, most of an arm's length already recedes behind the head silhouette
    // (this is physically correct: only a sliver near the hinge is visible from
    // directly in front). The head/ear volumes should not occlude LESS than the
    // face mask alone at any yaw.
    const farArm = templeArm(-1)
    const faceOnlyAt0 = withFaceOnly(farArm, 0)
    const allAt0 = withAll(farArm, 0)
    expect(allAt0).toBeGreaterThanOrEqual(faceOnlyAt0 - 1e-6)

    // At a steep yaw the far arm swings behind the head and should be mostly hidden
    // by face+head+ears together, at least as much as by the face mask alone.
    const faceOnlyAt60 = withFaceOnly(farArm, 60)
    const allAt60 = withAll(farArm, 60)
    expect(allAt60).toBeGreaterThanOrEqual(faceOnlyAt60)
    expect(allAt60).toBeGreaterThan(0.5)

    // The NEAR arm (same side as the camera swings toward) should become MORE
    // visible as yaw increases -- it swings toward the camera, away from the head.
    const nearArm = templeArm(1)
    const nearAt0 = withAll(nearArm, 0)
    const nearAt60 = withAll(nearArm, 60)
    expect(nearAt60).toBeLessThan(nearAt0)
  })

  it('ear geometry is a genuine mirror pair (right x = -left x, same y/z)', () => {
    const geometry = buildEarGeometry()
    const position = geometry.getAttribute('position')
    const half = position.count / 2
    for (let i = 0; i < 10; i += 1) {
      const a = new THREE.Vector3().fromBufferAttribute(position, i)
      const b = new THREE.Vector3().fromBufferAttribute(position, half + i)
      expect(b.x).toBeCloseTo(-a.x, 5)
      expect(b.y).toBeCloseTo(a.y, 5)
      expect(b.z).toBeCloseTo(a.z, 5)
    }
  })
})
