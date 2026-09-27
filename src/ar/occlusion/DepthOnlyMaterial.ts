import * as THREE from 'three'

/**
 * Depth-only material. Writes depth but no colour, and pushes each vertex
 * AWAY from the camera along its own view ray by `uPush` centimetres.
 *
 * A view-ray push is different from shrinking or offsetting the geometry:
 * the occluder's silhouette on screen is unchanged (every vertex slides along
 * the ray through itself), so hiding things that are genuinely behind the head
 * still works, while eyewear that lies within `uPush` of the surface (a pad on
 * the nose, a straight arm resting against the temple) is not cut away.
 *
 * Camera space is in centimetres (the whole AR scene is), so `uPush` is in cm.
 */
export function createDepthOnlyMaterial(pushCm: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { uPush: { value: pushCm } },
    vertexShader: /* glsl */ `
      uniform float uPush;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        mv.xyz += normalize(mv.xyz) * uPush;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */ `
      void main() {
        gl_FragColor = vec4(0.0);
      }
    `,
    colorWrite: false,
    depthWrite: true,
    depthTest: true,
    side: THREE.DoubleSide,
    transparent: false,
  })
}

export function setDepthOnlyPush(material: THREE.ShaderMaterial, pushCm: number): void {
  material.uniforms.uPush.value = pushCm
}

/**
 * Depth-only material with a PER-VERTEX push, driven by an `aLateral`
 * attribute (0..1) already present on the geometry (see
 * canonicalFaceModel.MIRRORED_CANONICAL_LATERAL_BIAS): push amount is
 * `mix(pushNearCm, pushFarCm, aLateral)`.
 *
 * Used for the face surface specifically, so the nose bridge/brow can stay
 * tight (small pushNearCm, protects Rule 4's anatomical accuracy) while the
 * cheek/temple is generous (larger pushFarCm, gives temple arms the
 * clearance they need — see MIRRORED_CANONICAL_LATERAL_BIAS for the full
 * rationale). Geometry without an `aLateral` attribute is not supported by
 * this material; use `createDepthOnlyMaterial` for the head/ear proxies and
 * anything else with a uniform push.
 */
export function createFaceDepthMaterial(pushNearCm: number, pushFarCm: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uPushNear: { value: pushNearCm },
      uPushFar: { value: pushFarCm },
    },
    vertexShader: /* glsl */ `
      attribute float aLateral;
      uniform float uPushNear;
      uniform float uPushFar;
      void main() {
        float push = mix(uPushNear, uPushFar, aLateral);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        mv.xyz += normalize(mv.xyz) * push;
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */ `
      void main() {
        gl_FragColor = vec4(0.0);
      }
    `,
    colorWrite: false,
    depthWrite: true,
    depthTest: true,
    side: THREE.DoubleSide,
    transparent: false,
  })
}

export function setFaceDepthPush(material: THREE.ShaderMaterial, pushNearCm: number, pushFarCm: number): void {
  material.uniforms.uPushNear.value = pushNearCm
  material.uniforms.uPushFar.value = pushFarCm
}

/** Translucent overlay used only by the "show occluders" diagnostic. */
export function createDebugMaterial(color: number, wireframe: boolean): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: wireframe ? 0.9 : 0.28,
    wireframe,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
  })
}
