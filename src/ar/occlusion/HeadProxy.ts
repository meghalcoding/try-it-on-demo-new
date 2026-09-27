import * as THREE from 'three'

/**
 * Analytic head and ear volumes, expressed in the app's mirrored canonical
 * face-local space (centimetres; +X screen-right, +Y up, +Z toward the camera;
 * origin inside the head; nose tip at z = +7.5, face-oval edge at z = -2.4).
 *
 * WHY
 * ---
 * MediaPipe's face mesh is an open MASK that ends at the ear-front plane
 * (z = -2.44). Behind that plane nothing represents the head, so a temple arm
 * on the far side of a turned head has nothing in front of it and is drawn
 * straight over the hair/ear region like a sticker. These volumes supply the
 * missing depth: the cranium (a closed ellipsoid) and the two ears.
 *
 * Because the volumes live in canonical space they are placed by the SAME
 * smoothed pose (rotation, translation and uniform scale) as the glasses, so
 * they neither drift nor need per-user sizing: MediaPipe's pose scale already
 * carries the user's head size.
 *
 * HOW THEY ARE USED
 * -----------------
 * They are depth-only (colour writes off) and are pushed AWAY from the camera
 * along each view ray by a few millimetres in the vertex shader. That is not
 * the same as shrinking them: the silhouette stays full size (so an arm inside
 * the head outline in a frontal view is still hidden) while an arm that runs
 * along the skin on the near side (assets model arms straighter/narrower than a
 * real head) is not cut away. See `DepthOnlyMaterial`.
 */

export interface EllipsoidSpec {
  /** Centre in canonical face space, cm. */
  readonly center: readonly [number, number, number]
  /** Semi-axes along X, Y, Z before tilt, cm. */
  readonly radii: readonly [number, number, number]
  /** Rotation about the X axis, radians (positive tilts +Y toward +Z). */
  readonly tiltX: number
}

/**
 * Anthropometric fit. Sources of the numbers:
 *  - Head breadth ~15 cm, glabella-opisthocranion ~19 cm (adult means).
 *  - Canonical mesh: glabella z ~ +5.3, face-oval edge z = -2.44, eye line y = 2.6.
 *  - Ear: ~6 cm tall, ~3.5 cm wide, protrudes ~1.8 cm, top near the brow line,
 *    lobe near the nose base, tilted back ~15 degrees.
 *
 * The skull front is deliberately kept >= ~0.5 cm BEHIND the canonical face
 * surface everywhere in the face footprint (asserted by the unit test), so the
 * skull can never occlude eyewear that rests on or in front of the face.
 */
export const SKULL: EllipsoidSpec = Object.freeze({
  center: [0, 4.0, -4.6] as const,
  radii: [6.7, 9.6, 8.4] as const,
  tiltX: 0,
})

export const EAR: EllipsoidSpec = Object.freeze({
  center: [8.0, 1.3, -4.9] as const,
  radii: [0.9, 3.1, 1.8] as const,
  tiltX: -0.26,
})

function buildEllipsoid(
  spec: EllipsoidSpec,
  mirrorX: boolean,
  widthSegments: number,
  heightSegments: number,
): THREE.BufferGeometry {
  const sphere = new THREE.SphereGeometry(1, widthSegments, heightSegments)
  const [cx, cy, cz] = spec.center
  const matrix = new THREE.Matrix4().compose(
    new THREE.Vector3(mirrorX ? -cx : cx, cy, cz),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(spec.tiltX, 0, 0)),
    new THREE.Vector3(spec.radii[0], spec.radii[1], spec.radii[2]),
  )
  sphere.applyMatrix4(matrix)

  // Depth-only usage: keep just position + index to stay light.
  sphere.deleteAttribute('normal')
  sphere.deleteAttribute('uv')
  return sphere
}

function mergePositionOnly(parts: readonly THREE.BufferGeometry[]): THREE.BufferGeometry {
  let vertexCount = 0
  let indexCount = 0
  for (const part of parts) {
    vertexCount += part.getAttribute('position').count
    indexCount += part.getIndex()?.count ?? 0
  }

  const positions = new Float32Array(vertexCount * 3)
  const indices = new Uint32Array(indexCount)
  let vertexOffset = 0
  let indexOffset = 0

  for (const part of parts) {
    const position = part.getAttribute('position') as THREE.BufferAttribute
    positions.set(position.array as Float32Array, vertexOffset * 3)

    const index = part.getIndex()
    if (index) {
      for (let i = 0; i < index.count; i += 1) {
        indices[indexOffset + i] = index.getX(i) + vertexOffset
      }
      indexOffset += index.count
    }

    vertexOffset += position.count
  }

  const merged = new THREE.BufferGeometry()
  merged.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  merged.setIndex(new THREE.BufferAttribute(indices, 1))
  return merged
}

/** Closed cranium volume. */
export function buildSkullGeometry(): THREE.BufferGeometry {
  return buildEllipsoid(SKULL, false, 32, 24)
}

/** Both ears (mirror pair) merged into one draw. */
export function buildEarGeometry(): THREE.BufferGeometry {
  const right = buildEllipsoid(EAR, false, 16, 12)
  const left = buildEllipsoid(EAR, true, 16, 12)
  const merged = mergePositionOnly([right, left])
  right.dispose()
  left.dispose()
  return merged
}
