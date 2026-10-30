// 3D brains: neurons as additive points; spikes flash and fade (~200 ms trails) under bloom.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import type { Brain, Sex } from "./data";

export type View = "side" | "overlay" | "diff";
export const PALETTE = {
  vivid: { male: [0.25, 0.85, 1.0], female: [1.0, 0.72, 0.2] },
  clean: { male: [0.0, 0.45, 0.70], female: [0.84, 0.37, 0.0] }, // Okabe-Ito blue / vermillion
};

const VERT = /* glsl */ `
attribute float act; attribute float sel; attribute vec3 col; attribute float size; attribute float vis;
uniform float uScale; uniform float uIdle;
varying vec3 vCol; varying float vA;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float a = clamp(act, 0.0, 1.0);
  gl_PointSize = vis * size * (1.0 + 1.6 * a + 2.5 * sel) * uScale / -mv.z;
  vCol = col * (uIdle + (1.0 - uIdle) * a) + vec3(sel * 0.9);
  vA = vis * (uIdle + a + sel);
}`;
const FRAG = /* glsl */ `
varying vec3 vCol; varying float vA;
void main() {
  vec2 d = gl_PointCoord - 0.5; float r = dot(d, d);
  if (r > 0.25) discard;
  float f = smoothstep(0.25, 0.0, r);
  gl_FragColor = vec4(vCol * f * min(vA, 1.6), 1.0);
}`;

interface Cloud { brain: Brain; points: THREE.Points; act: Float32Array; sel: Float32Array; col: Float32Array; vis: Float32Array; geo: THREE.BufferGeometry }

export class BrainScene {
  renderer: THREE.WebGLRenderer; scene = new THREE.Scene(); camera: THREE.PerspectiveCamera; controls: OrbitControls;
  composer: EffectComposer; bloom: UnrealBloomPass;
  clouds = new Map<Sex, Cloud>(); view: View = "side"; clean = false;
  private ray = new THREE.Raycaster(); private last = performance.now(); fps = 60; private fpsAcc: number[] = [];
  onFrame: (dt: number) => void = () => {};

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 1);
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    this.camera.position.set(0, 0.1, 3.3);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.15, 0.3);
    // Keep only the tight mip levels: the wide ones wash the whole frame grey around dense clusters.
    (this.bloom as any).compositeMaterial.uniforms.bloomFactors.value = [1.0, 0.7, 0.3, 0.0, 0.0];
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.ray.params.Points = { threshold: 0.02 };
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    const loop = () => {
      const now = performance.now(), dt = Math.min(0.1, (now - this.last) / 1000);
      this.last = now;
      this.fpsAcc.push(dt); if (this.fpsAcc.length > 30) this.fpsAcc.shift();
      this.fps = this.fpsAcc.length / this.fpsAcc.reduce((a, b) => a + b, 0);
      try { this.onFrame(dt); } catch (e) { console.error(e); }
      this.decay(dt);
      this.controls.update();
      if (this.clean) this.renderer.render(this.scene, this.camera); else this.composer.render();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  resize() {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    for (const c of this.clouds.values()) (c.points.material as THREE.ShaderMaterial).uniforms.uScale.value = h * 0.008;
  }

  setBrains(brains: Brain[]) {
    for (const b of brains) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(b.positions, 3));
      const act = new Float32Array(b.n), sel = new Float32Array(b.n), col = new Float32Array(b.n * 3), size = new Float32Array(b.n), vis = new Float32Array(b.n).fill(1);
      for (let i = 0; i < b.n; i++) size[i] = b.target[i] ? 7 : b.stage[i] === 0 ? 3.2 : 4.2;
      for (const [k, a] of [["act", act], ["sel", sel], ["size", size], ["vis", vis]] as const) geo.setAttribute(k, new THREE.BufferAttribute(a, 1));
      geo.setAttribute("col", new THREE.BufferAttribute(col, 3));
      const mat = new THREE.ShaderMaterial({
        vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uScale: { value: (this.host.clientHeight || 600) * 0.008 }, uIdle: { value: 0.07 } },
      });
      const points = new THREE.Points(geo, mat);
      points.frustumCulled = false;
      this.scene.add(points);
      this.clouds.set(b.sex, { brain: b, points, act, sel, col, vis, geo });
    }
    this.setView(this.view);
  }

  setView(v: View) {
    this.view = v;
    for (const [sex, c] of this.clouds) c.points.position.x = v === "overlay" ? 0 : sex === "male" ? -1.2 : 1.2;
    this.recolor();
  }

  setClean(on: boolean) { this.clean = on; this.recolor(); }

  /** Diverging colours for the difference view: +1 = male higher, -1 = female higher. */
  diff: Map<Sex, Float32Array> = new Map();
  recolor() {
    const pal = this.clean ? PALETTE.clean : PALETTE.vivid;
    for (const [sex, c] of this.clouds) {
      const base = pal[sex], d = this.diff.get(sex);
      for (let i = 0; i < c.brain.n; i++) {
        let rgb = base;
        if (this.view === "diff") {
          const x = d ? d[i] : 0, m = pal.male, f = pal.female, a = Math.min(1, Math.abs(x));
          rgb = Number.isNaN(x) ? [0.06, 0.06, 0.07] : x >= 0 ? mix([0.22, 0.22, 0.25], m, a) : mix([0.22, 0.22, 0.25], f, a);
        }
        c.col.set(rgb, i * 3);
      }
      c.geo.attributes.col.needsUpdate = true;
      (c.points.material as THREE.ShaderMaterial).uniforms.uIdle.value = this.view === "diff" ? 0.3 : this.clean ? 0.3 : 0.07;
    }
  }

  spike(sex: Sex, ids: ArrayLike<number>) {
    const c = this.clouds.get(sex); if (!c) return;
    for (let k = 0; k < ids.length; k++) c.act[ids[k]] = 1;
  }

  private decay(dt: number) {
    const f = Math.exp(-dt / 0.07); // ~200 ms to fade below 6 %
    for (const c of this.clouds.values()) {
      const a = c.act;
      for (let i = 0; i < a.length; i++) if (a[i] > 0.002) a[i] *= f; else a[i] = 0;
      c.geo.attributes.act.needsUpdate = true;
    }
  }

  select(sex: Sex | null, idx: number[]) {
    for (const [s, c] of this.clouds) { c.sel.fill(0); if (s === sex) for (const i of idx) c.sel[i] = 1; c.geo.attributes.sel.needsUpdate = true; }
  }

  setVisible(sex: Sex, mask: Uint8Array | null) {
    const c = this.clouds.get(sex); if (!c) return;
    for (let i = 0; i < c.brain.n; i++) c.vis[i] = mask ? mask[i] : 1;
    c.geo.attributes.vis.needsUpdate = true;
  }

  pick(clientX: number, clientY: number): { sex: Sex; idx: number } | null {
    const r = this.renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(p, this.camera);
    let best: { sex: Sex; idx: number; d: number } | null = null;
    for (const [sex, c] of this.clouds) for (const h of this.ray.intersectObject(c.points)) {
      if (h.index !== undefined && c.vis[h.index] > 0 && (!best || h.distanceToRay! < best.d)) best = { sex, idx: h.index, d: h.distanceToRay! };
    }
    return best;
  }

  preset(name: "front" | "top" | "side" | "auditory") {
    const t = new THREE.Vector3(0, 0, 0);
    const pos = { front: [0, 0.1, 3.3], top: [0, 3.6, 0.01], side: [3.6, 0.2, 0], auditory: [0, 0, 2.2] }[name];
    if (name === "auditory") {
      const c = this.clouds.get("male") ?? [...this.clouds.values()][0];
      if (c) {
        let x = 0, y = 0, z = 0, k = 0;
        for (let i = 0; i < c.brain.n; i++) if (c.brain.stage[i] <= 1) { x += c.brain.positions[3 * i]; y += c.brain.positions[3 * i + 1]; z += c.brain.positions[3 * i + 2]; k++; }
        if (k) t.set(x / k + c.points.position.x, y / k, z / k);
      }
      this.camera.position.set(t.x, t.y + 0.1, t.z + 1.6);
    } else this.camera.position.set(pos[0], pos[1], pos[2]);
    this.controls.target.copy(t);
  }
}

function mix(a: number[], b: number[], t: number): number[] { return a.map((v, i) => v + (b[i] - v) * t); }
