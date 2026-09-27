import * as THREE from 'three'
import type { ForegroundMaskFrame } from './ForegroundSegmenter'
import { computeCoverMapping } from './TrackerProjection'
import type { FaceOcclusionSettings } from '../../types/FaceOcclusion'

const MAX_SHIFT = 0.15

/**
 * Shared GLSL. `fgOcclusion()` returns 0..1: how much of the pixel under the
 * current fragment is covered by hair / a hand / an object that is in front
 * of the face. It is used both to fade the glasses and by the debug overlay,
 * so the overlay shows exactly what the glasses will do.
 *
 * Mapping (see maskMath.screenToMaskUv for the tested JS twin):
 *   gl_FragCoord (bottom-left, device px) -> top-left px -> cover-crop into the
 *   video -> mirror x (display is mirrored, mask is not) -> subtract the
 *   motion-compensation shift.
 *
 * Gating:
 *   - hair only counts inside the face oval (bangs, strands); `uFgReach`
 *     widens that toward a ring around it (hair falling over the temples);
 *   - hands / objects count in a wider ring, and only where the segmenter does
 *     NOT think the pixel is face skin (protects against the model labelling
 *     part of the user's own face as "body-skin").
 */
const OCCLUSION_GLSL = /* glsl */ `
uniform sampler2D uFgMask;
uniform sampler2D uFgGate;
uniform float uFgEnabled;
uniform vec2 uFgViewport;
uniform vec4 uFgCover;
uniform vec2 uFgShift;
uniform vec3 uFgStrength;
uniform float uFgReach;

float fgOcclusion() {
  if (uFgEnabled < 0.5) return 0.0;
  vec2 px = vec2(gl_FragCoord.x, uFgViewport.y - gl_FragCoord.y);
  vec2 d = (px - uFgCover.xy) / uFgCover.zw;
  if (d.x < 0.0 || d.x > 1.0 || d.y < 0.0 || d.y > 1.0) return 0.0;
  vec2 uv = vec2(1.0 - d.x - uFgShift.x, d.y - uFgShift.y);
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;

  vec4 m = texture2D(uFgMask, uv);
  float gate = texture2D(uFgGate, uv).r;
  float hairGate = smoothstep(mix(0.60, 0.05, uFgReach), mix(0.85, 0.30, uFgReach), gate);
  float wideGate = smoothstep(0.05, 0.30, gate);
  float notFace = 1.0 - smoothstep(0.35, 0.70, m.a);

  float hair = smoothstep(0.45, 0.75, m.r) * hairGate * uFgStrength.x;
  float hand = smoothstep(0.50, 0.80, m.g) * wideGate * notFace * uFgStrength.y;
  float obj  = smoothstep(0.45, 0.75, m.b) * wideGate * notFace * uFgStrength.z;
  return 1.0 - (1.0 - hair) * (1.0 - hand) * (1.0 - obj);
}
`

const INJECTED_MAIN_TAIL = /* glsl */ `
{
  float fgOcc = fgOcclusion();
  #ifdef OPAQUE
    // Opaque materials write straight to a premultiplied-alpha canvas.
    gl_FragColor.rgb *= (1.0 - fgOcc);
  #endif
  gl_FragColor.a *= (1.0 - fgOcc);
}
`

const FRAGMENT_ANCHOR = '#include <dithering_fragment>'
const MAIN_ANCHOR = 'void main() {'
const PROGRAM_KEY = '|fgocc1'

type SharedUniforms = Record<string, THREE.IUniform>

/**
 * Screen-space foreground occlusion for the glasses (Rules 2 & 3).
 *
 * The glasses' own materials are patched (once) so their alpha is multiplied
 * by `1 - occlusion` per pixel. Patching fails CLOSED: if a material's shader
 * does not contain the expected anchors it is left untouched (no mask
 * occlusion for that material) and a single warning is logged; it can never
 * break rendering.
 */
export class ForegroundOcclusion {
  readonly uniforms: SharedUniforms
  private classTexture: THREE.DataTexture
  private gateTexture: THREE.DataTexture
  private textureWidth = 1
  private textureHeight = 1
  private readonly patched = new WeakSet<THREE.Material>()
  private warned = false
  private lastVersion = -1
  private hasMask = false
  private settingsEnabled = true
  private debugRequested = false
  private readonly debugMesh: THREE.Mesh
  private readonly debugMaterial: THREE.ShaderMaterial

  constructor() {
    this.classTexture = ForegroundOcclusion.makeTexture(1, 1, THREE.RGBAFormat)
    this.gateTexture = ForegroundOcclusion.makeTexture(1, 1, THREE.RedFormat)

    this.uniforms = {
      uFgMask: { value: this.classTexture },
      uFgGate: { value: this.gateTexture },
      uFgEnabled: { value: 0 },
      uFgViewport: { value: new THREE.Vector2(1, 1) },
      uFgCover: { value: new THREE.Vector4(0, 0, 1, 1) },
      uFgShift: { value: new THREE.Vector2(0, 0) },
      uFgStrength: { value: new THREE.Vector3(0.85, 0.9, 0.7) },
      uFgReach: { value: 0.35 },
    }

    this.debugMaterial = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: /* glsl */ `void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`,
      fragmentShader: /* glsl */ `
        ${OCCLUSION_GLSL}
        void main() {
          float o = fgOcclusion();
          gl_FragColor = vec4(1.0, 0.1, 0.35, 0.55 * o);
        }
      `,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    })
    this.debugMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.debugMaterial)
    this.debugMesh.name = 'ForegroundMaskDebug'
    this.debugMesh.frustumCulled = false
    this.debugMesh.renderOrder = 1000
    this.debugMesh.visible = false
  }

  getDebugObject(): THREE.Object3D {
    return this.debugMesh
  }

  setSettings(settings: FaceOcclusionSettings): void {
    this.settingsEnabled = settings.enabled && settings.foregroundMaskEnabled
    this.debugRequested = settings.debugShowMask
    const strength = this.uniforms.uFgStrength.value as THREE.Vector3
    strength.set(
      settings.hairOcclusionStrength,
      settings.handOcclusionStrength,
      settings.objectOcclusionStrength,
    )
    this.uniforms.uFgReach.value = settings.hairReach
    this.syncEnabled()
  }

  /** Drawing-buffer size (device px) and the video source size (for the cover crop). */
  setGeometry(
    viewportWidth: number,
    viewportHeight: number,
    videoWidth: number,
    videoHeight: number,
  ): void {
    if (!(viewportWidth > 0) || !(viewportHeight > 0)) return
    const cover = computeCoverMapping(viewportWidth, viewportHeight, videoWidth, videoHeight)
    ;(this.uniforms.uFgViewport.value as THREE.Vector2).set(viewportWidth, viewportHeight)
    ;(this.uniforms.uFgCover.value as THREE.Vector4).set(
      cover.offsetX,
      cover.offsetY,
      cover.renderedWidth,
      cover.renderedHeight,
    )
  }

  /** Publish a new mask (no-op if the version is unchanged). */
  setMask(frame: ForegroundMaskFrame | null): void {
    if (!frame) {
      this.hasMask = false
      this.lastVersion = -1
      this.syncEnabled()
      return
    }
    if (frame.version === this.lastVersion) return
    this.lastVersion = frame.version

    if (frame.width !== this.textureWidth || frame.height !== this.textureHeight) {
      this.classTexture.dispose()
      this.gateTexture.dispose()
      this.classTexture = ForegroundOcclusion.makeTexture(frame.width, frame.height, THREE.RGBAFormat)
      this.gateTexture = ForegroundOcclusion.makeTexture(frame.width, frame.height, THREE.RedFormat)
      this.textureWidth = frame.width
      this.textureHeight = frame.height
      this.uniforms.uFgMask.value = this.classTexture
      this.uniforms.uFgGate.value = this.gateTexture
    }

    ;(this.classTexture.image.data as Uint8Array).set(frame.classes)
    ;(this.gateTexture.image.data as Uint8Array).set(frame.gate)
    this.classTexture.needsUpdate = true
    this.gateTexture.needsUpdate = true
    this.hasMask = true
    this.syncEnabled()
  }

  /** Face displacement since the mask was captured, normalised un-mirrored frame units. */
  setMotionShift(shiftX: number, shiftY: number): void {
    const clamp = (v: number) => (Number.isFinite(v) ? Math.max(-MAX_SHIFT, Math.min(MAX_SHIFT, v)) : 0)
    ;(this.uniforms.uFgShift.value as THREE.Vector2).set(clamp(shiftX), clamp(shiftY))
  }

  isActive(): boolean {
    return this.settingsEnabled && this.hasMask
  }

  /** Patch every material under `root` (idempotent). Call after a model is set. */
  attach(root: THREE.Object3D): void {
    root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return
      const materials = Array.isArray(object.material) ? object.material : [object.material]
      for (const material of materials) this.patchMaterial(material as THREE.Material)
    })
  }

  dispose(): void {
    this.classTexture.dispose()
    this.gateTexture.dispose()
    this.debugMaterial.dispose()
    this.debugMesh.geometry.dispose()
  }

  private syncEnabled(): void {
    const active = this.isActive()
    this.uniforms.uFgEnabled.value = active ? 1 : 0
    this.debugMesh.visible = active && this.debugRequested
  }

  private patchMaterial(material: THREE.Material): void {
    if (this.patched.has(material)) return
    this.patched.add(material)

    const previousHook = material.onBeforeCompile
    const previousKey = material.customProgramCacheKey.bind(material)

    material.onBeforeCompile = (shader, renderer) => {
      previousHook.call(material, shader, renderer)
      this.inject(shader)
    }
    material.customProgramCacheKey = () => previousKey() + PROGRAM_KEY
    material.needsUpdate = true
  }

  private inject(shader: THREE.WebGLProgramParametersWithUniforms): void {
    const fragment = shader.fragmentShader
    if (!fragment.includes(FRAGMENT_ANCHOR) || !fragment.includes(MAIN_ANCHOR)) {
      if (!this.warned) {
        this.warned = true
        console.warn(
          '[ForegroundOcclusion] A material has no dithering_fragment/main anchor; ' +
            'hair/hand occlusion is skipped for it.',
        )
      }
      return
    }

    Object.assign(shader.uniforms, this.uniforms)
    shader.fragmentShader = fragment
      .replace(MAIN_ANCHOR, `${OCCLUSION_GLSL}\n${MAIN_ANCHOR}`)
      .replace(FRAGMENT_ANCHOR, `${FRAGMENT_ANCHOR}\n${INJECTED_MAIN_TAIL}`)
  }

  private static makeTexture(
    width: number,
    height: number,
    format: typeof THREE.RGBAFormat | typeof THREE.RedFormat,
  ): THREE.DataTexture {
    const channels = format === THREE.RGBAFormat ? 4 : 1
    const texture = new THREE.DataTexture(
      new Uint8Array(width * height * channels),
      width,
      height,
      format,
      THREE.UnsignedByteType,
    )
    texture.magFilter = THREE.LinearFilter
    texture.minFilter = THREE.LinearFilter
    texture.generateMipmaps = false
    texture.flipY = false
    texture.unpackAlignment = 1
    texture.needsUpdate = true
    return texture
  }
}
