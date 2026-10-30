// Timeline: waveform + spectrogram, spike raster grouped by stage (male above female), pinned probe traces.
import { fft } from "./ear";
import type { Brain, Run, Sex } from "./data";

export interface Probe { sex: Sex; idx: number; label: string; color: string }
const ROWS = { wave: 34, spec: 46, raster: 132, probe: 54 };
const STAGE_COLORS = ["#6b7a8f", "#7fb8c9", "#9ac48f", "#e6c36a", "#e58f8f"];

export class Timeline {
  private ctx: CanvasRenderingContext2D;
  private layer = document.createElement("canvas");
  private rowOf = new Map<Sex, Int32Array>();
  private nRows = new Map<Sex, number>();
  private brains = new Map<Sex, Brain>();
  private replay: { runs: Map<Sex, Run>; audio: Float32Array; duration: number } | null = null;
  private live = false;
  private liveWindow = 3;
  private liveSp: { sex: Sex; ids: Uint32Array; t: Float32Array }[] = [];
  private liveAudio: Float32Array[] = [];
  private liveNow = 0;
  probes: Probe[] = [];
  private probeTraces: Float32Array[] = [];
  time = 0;
  private dragging = false;

  constructor(private canvas: HTMLCanvasElement, private onScrub: (t: number) => void) {
    this.ctx = canvas.getContext("2d")!;
    const scrub = (e: PointerEvent) => {
      if (!this.replay) return;
      const r = canvas.getBoundingClientRect(), x = (e.clientX - r.left - 64) / (r.width - 64);
      this.onScrub(Math.max(0, Math.min(1, x)) * this.replay.duration);
    };
    canvas.addEventListener("pointerdown", (e) => { this.dragging = true; canvas.setPointerCapture(e.pointerId); scrub(e); });
    canvas.addEventListener("pointermove", (e) => this.dragging && scrub(e));
    canvas.addEventListener("pointerup", () => (this.dragging = false));
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  get height() { return ROWS.wave + ROWS.spec + ROWS.raster + ROWS.probe + 8; }

  setBrains(brains: Brain[]) {
    for (const b of brains) {
      this.brains.set(b.sex, b);
      const order = Array.from({ length: b.n }, (_, i) => i).sort((a, c) => b.stage[a] - b.stage[c] || b.hop[a] - b.hop[c] || b.typeIdx[a] - b.typeIdx[c]);
      const row = new Int32Array(b.n); order.forEach((i, r) => (row[i] = r));
      this.rowOf.set(b.sex, row); this.nRows.set(b.sex, b.n);
    }
  }

  private resize() {
    const r = this.canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio, 2);
    this.canvas.width = Math.max(1, r.width * dpr); this.canvas.height = Math.max(1, r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.replay) this.renderLayer();
  }

  loadReplay(runs: Map<Sex, Run>) {
    const first = [...runs.values()][0];
    this.live = false;
    this.replay = { runs, audio: first.audio, duration: first.duration };
    this.renderLayer();
    this.computeProbes();
  }

  setLive() {
    this.live = true; this.replay = null; this.liveSp = []; this.liveAudio = []; this.liveNow = 0; this.drawnNow = 0;
    this.liveProbe = new Map();
    const g = this.liveLayer.getContext("2d")!; g.clearRect(0, 0, this.liveLayer.width, this.liveLayer.height);
  }
  // Live raster is drawn incrementally into a scrolling layer; probe rates are binned as spikes arrive.
  private liveLayer = document.createElement("canvas");
  private audioLayer = document.createElement("canvas");
  private audioDrawn = 0;
  private drawnNow = 0;
  private liveProbe = new Map<string, Float32Array>();
  private static readonly PB = 120;
  pushLive(sex: Sex, ids: Uint32Array, t: Float32Array, now: number) {
    this.liveSp.push({ sex, ids, t }); this.liveNow = Math.max(this.liveNow, now);
    const bw = this.liveWindow / Timeline.PB;
    for (const p of this.probes) {
      if (p.sex !== sex) continue;
      const key = `${p.sex}:${p.idx}`, tr = this.liveProbe.get(key) ?? new Float32Array(Timeline.PB + 1);
      const cur = Math.floor(now / bw);
      for (let b = tr[Timeline.PB] + 1; b <= cur && b <= tr[Timeline.PB] + Timeline.PB; b++) tr[b % Timeline.PB] = 0;
      tr[Timeline.PB] = Math.max(tr[Timeline.PB], cur);
      for (let k = 0; k < ids.length; k++) if (ids[k] === p.idx) tr[Math.floor(t[k] / bw) % Timeline.PB] += 1 / bw;
      this.liveProbe.set(key, tr);
    }
  }
  pushAudio(x: Float32Array) {
    this.liveAudio.push(x);
    let n = this.liveAudio.reduce((a, b) => a + b.length, 0);
    while (n > this.liveWindow * 10000 && this.liveAudio.length > 1) n -= this.liveAudio.shift()!.length;
  }

  setProbes(p: Probe[]) { this.probes = p; this.computeProbes(); }

  private computeProbes() {
    this.probeTraces = [];
    if (!this.replay) return;
    const bins = 200, dur = this.replay.duration;
    for (const p of this.probes) {
      const run = this.replay.runs.get(p.sex), tr = new Float32Array(bins);
      if (run) for (let k = 0; k < run.ids.length; k++) if (run.ids[k] === p.idx) tr[Math.min(bins - 1, Math.floor((run.t[k] / dur) * bins))] += bins / dur;
      this.probeTraces.push(smooth(tr, 3));
    }
  }

  private renderLayer() {
    if (!this.replay) return;
    const w = this.canvas.clientWidth, h = this.height, dpr = Math.min(devicePixelRatio, 2);
    if (w < 80) return;
    this.layer.width = w * dpr; this.layer.height = h * dpr;
    const g = this.layer.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const x0 = 64, W = w - x0, { audio, duration, runs } = this.replay;
    this.labels(g);
    drawWave(g, audio, x0, 0, W, ROWS.wave);
    drawSpec(g, audio, x0, ROWS.wave, W, ROWS.spec);
    let y = ROWS.wave + ROWS.spec;
    const half = ROWS.raster / 2;
    for (const sex of ["male", "female"] as Sex[]) {
      const run = runs.get(sex), row = this.rowOf.get(sex), b = this.brains.get(sex);
      if (run && row && b) {
        const n = this.nRows.get(sex)!;
        const img = g.getImageData(x0 * dpr, y * dpr, Math.max(1, Math.floor(W * dpr)), Math.max(1, Math.floor(half * dpr)));
        const iw = img.width, ih = img.height;
        for (let k = 0; k < run.ids.length; k++) {
          const i = run.ids[k], px = Math.floor((run.t[k] / duration) * (iw - 1)), py = Math.floor((row[i] / n) * (ih - 1));
          const c = hex(STAGE_COLORS[b.stage[i]]), o = (py * iw + px) * 4;
          img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
        }
        g.putImageData(img, x0 * dpr, y * dpr);
      }
      y += half;
    }
    this.rasterGuides(g, x0, ROWS.wave + ROWS.spec, W);
  }

  private labels(g: CanvasRenderingContext2D) {
    g.font = "10px 'JetBrains Mono', ui-monospace, monospace"; g.fillStyle = "#7d8794";
    let y = 0;
    for (const [k, h] of Object.entries(ROWS)) {
      const lab = { wave: "WAVE", spec: "SPEC 0-1k", raster: "", probe: "PROBES" }[k]!;
      g.fillText(lab, 6, y + 13); y += h;
    }
    g.fillText("MALE", 6, ROWS.wave + ROWS.spec + 13); g.fillText("FEMALE", 6, ROWS.wave + ROWS.spec + ROWS.raster / 2 + 13);
  }

  private rasterGuides(g: CanvasRenderingContext2D, x0: number, y0: number, W: number) {
    g.strokeStyle = "#1c232c"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0, y0 + ROWS.raster / 2 + 0.5); g.lineTo(x0 + W, y0 + ROWS.raster / 2 + 0.5); g.stroke();
    for (const sex of ["male", "female"] as Sex[]) {
      const b = this.brains.get(sex); if (!b) continue;
      const counts = [0, 0, 0, 0, 0]; for (let i = 0; i < b.n; i++) counts[b.stage[i]]++;
      let acc = 0; const yb = y0 + (sex === "female" ? ROWS.raster / 2 : 0);
      counts.forEach((c, s) => { g.fillStyle = STAGE_COLORS[s]; g.fillRect(x0 - 4, yb + (acc / b.n) * (ROWS.raster / 2), 3, Math.max(1, (c / b.n) * (ROWS.raster / 2))); acc += c; });
    }
  }

  draw() {
    const g = this.ctx, w = this.canvas.clientWidth, x0 = 64, W = w - x0;
    g.clearRect(0, 0, w, this.canvas.clientHeight);
    if (this.replay) {
      if (!this.layer.width || !this.layer.height) this.renderLayer();
      if (this.layer.width && this.layer.height) g.drawImage(this.layer, 0, 0, w, this.height);
      const px = x0 + (this.time / this.replay.duration) * W;
      this.drawProbes(g, x0, W, (i) => this.probeTraces[i], 200);
      g.strokeStyle = "#e8eef5"; g.lineWidth = 1; g.beginPath(); g.moveTo(px + 0.5, 0); g.lineTo(px + 0.5, this.height); g.stroke();
      g.fillStyle = "#e8eef5"; g.font = "10px 'JetBrains Mono', ui-monospace, monospace"; g.fillText(`${(this.time * 1000).toFixed(1)} ms`, Math.min(px + 4, w - 60), 10);
    } else if (this.live) this.drawLive(g, x0, W);
  }

  private drawLive(g: CanvasRenderingContext2D, x0: number, W: number) {
    this.labels(g);
    const A = this.audioLayer, now = performance.now();
    if (now - this.audioDrawn > 150 || A.width !== Math.floor(W)) {
      this.audioDrawn = now;
      A.width = Math.max(1, Math.floor(W)); A.height = ROWS.wave + ROWS.spec;
      const ag = A.getContext("2d")!;
      const a = new Float32Array(this.liveAudio.reduce((s, b) => s + b.length, 0)); let o = 0;
      for (const b of this.liveAudio) { a.set(b, o); o += b.length; }
      const span = this.liveWindow * 10000, xa = W * (1 - a.length / span);
      drawWave(ag, a, xa, 0, W * (a.length / span), ROWS.wave);
      if (a.length > 1024) drawSpec(ag, a, xa, ROWS.wave, W * (a.length / span), ROWS.spec);
    }
    g.drawImage(A, x0, 0);
    const y0 = ROWS.wave + ROWS.spec, half = ROWS.raster / 2, Wi = Math.max(1, Math.floor(W)), Hi = ROWS.raster;
    const L = this.liveLayer, lg = L.getContext("2d")!;
    if (L.width !== Wi || L.height !== Hi) { L.width = Wi; L.height = Hi; this.drawnNow = this.liveNow; }
    const shift = Math.floor(((this.liveNow - this.drawnNow) / this.liveWindow) * Wi);
    if (shift > 0) {
      lg.globalCompositeOperation = "copy"; lg.drawImage(L, -shift, 0); lg.globalCompositeOperation = "source-over";
      this.drawnNow += (shift / Wi) * this.liveWindow;
    }
    const t0 = this.drawnNow - this.liveWindow;
    for (const chunk of this.liveSp) {
      const row = this.rowOf.get(chunk.sex)!, b = this.brains.get(chunk.sex)!, n = b.n, yb = chunk.sex === "female" ? half : 0;
      for (let k = 0; k < chunk.ids.length; k++) {
        lg.fillStyle = STAGE_COLORS[b.stage[chunk.ids[k]]];
        lg.fillRect(Math.floor(((chunk.t[k] - t0) / this.liveWindow) * Wi), Math.floor(yb + (row[chunk.ids[k]] / n) * half), 1, 1);
      }
    }
    this.liveSp = [];
    g.drawImage(L, x0, y0);
    this.rasterGuides(g, x0, y0, W);
    const bw = this.liveWindow / Timeline.PB, cur = Math.floor(this.liveNow / bw);
    const traces = this.probes.map((p) => {
      const tr = this.liveProbe.get(`${p.sex}:${p.idx}`), out = new Float32Array(Timeline.PB);
      if (tr) for (let i = 0; i < Timeline.PB; i++) out[i] = tr[(((cur - Timeline.PB + 1 + i) % Timeline.PB) + Timeline.PB) % Timeline.PB];
      return smooth(out, 2);
    });
    this.drawProbes(g, x0, W, (i) => traces[i], Timeline.PB);
  }

  private drawProbes(g: CanvasRenderingContext2D, x0: number, W: number, get: (i: number) => Float32Array | undefined, bins: number) {
    const y0 = ROWS.wave + ROWS.spec + ROWS.raster + 4, h = ROWS.probe - 6;
    let max = 1; this.probes.forEach((_, i) => { const t = get(i); if (t) for (const v of t) max = Math.max(max, v); });
    this.probes.forEach((p, i) => {
      const t = get(i); if (!t) return;
      g.strokeStyle = p.color; g.lineWidth = 1.2; g.beginPath();
      for (let k = 0; k < bins; k++) { const x = x0 + (k / (bins - 1)) * W, y = y0 + h - (t[k] / max) * h; k ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
      g.fillStyle = p.color; g.font = "10px 'JetBrains Mono', ui-monospace, monospace"; g.fillText(p.label, x0 + 4 + i * 110, y0 + 9);
    });
    g.fillStyle = "#56606c"; g.fillText(`${max.toFixed(0)} Hz`, 6, y0 + 22);
  }
}

function smooth(x: Float32Array, r: number): Float32Array {
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) { let s = 0, c = 0; for (let k = -r; k <= r; k++) if (x[i + k] !== undefined) { s += x[i + k]; c++; } y[i] = s / c; }
  return y;
}
function hex(h: string): number[] { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); }

function drawWave(g: CanvasRenderingContext2D, a: Float32Array, x0: number, y0: number, W: number, H: number) {
  if (!a.length || W < 2) return;
  let peak = 1e-9; for (const v of a) peak = Math.max(peak, Math.abs(v));
  g.fillStyle = "#8fd3e6";
  const cols = Math.floor(W), per = a.length / cols;
  for (let c = 0; c < cols; c++) {
    let lo = 0, hi = 0; const i0 = Math.floor(c * per), i1 = Math.max(i0 + 1, Math.floor((c + 1) * per));
    for (let i = i0; i < i1 && i < a.length; i++) { lo = Math.min(lo, a[i]); hi = Math.max(hi, a[i]); }
    const m = y0 + H / 2, s = (H / 2 - 3) / peak;
    g.fillRect(x0 + c, m - hi * s, 1, Math.max(1, (hi - lo) * s));
  }
}

function drawSpec(g: CanvasRenderingContext2D, a: Float32Array, x0: number, y0: number, W: number, H: number) {
  const N = 512, fmax = 1000, nb = Math.round((fmax / 10000) * N), cols = Math.max(1, Math.floor(W / 2));
  const re = new Float64Array(N), im = new Float64Array(N), mags: Float32Array[] = [];
  let peak = 1e-9;
  for (let c = 0; c < cols; c++) {
    const center = Math.floor(((c + 0.5) / cols) * a.length);
    re.fill(0); im.fill(0);
    for (let i = 0; i < N; i++) { const j = center - N / 2 + i; if (j >= 0 && j < a.length) re[i] = a[j] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N)); }
    fft(re, im, false);
    const m = new Float32Array(nb);
    for (let k = 0; k < nb; k++) { m[k] = Math.log1p(Math.hypot(re[k], im[k])); peak = Math.max(peak, m[k]); }
    mags.push(m);
  }
  const cw = W / cols, bh = H / nb;
  mags.forEach((m, c) => m.forEach((v, k) => {
    const t = v / peak; if (t < 0.08) return;
    g.fillStyle = `rgba(${Math.round(40 + 215 * t)},${Math.round(90 + 110 * t * t)},${Math.round(160 - 120 * t)},${t})`;
    g.fillRect(x0 + c * cw, y0 + H - (k + 1) * bh, cw + 0.5, bh + 0.5);
  }));
}
