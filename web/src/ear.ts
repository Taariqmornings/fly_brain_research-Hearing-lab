// Port of pipeline/ear.py: SOS filter bank per JO subtype (coefficients exported from scipy) -> rectify -> envelope.
export class Sos {
  private z: Float64Array;
  constructor(private s: number[][]) { this.z = new Float64Array(s.length * 2); }
  step(x: number): number {
    const s = this.s, z = this.z;
    for (let k = 0; k < s.length; k++) {
      const [b0, b1, b2, , a1, a2] = s[k];
      const y = b0 * x + z[2 * k];
      z[2 * k] = b1 * x - a1 * y + z[2 * k + 1];
      z[2 * k + 1] = b2 * x - a2 * y;
      x = y;
    }
    return x;
  }
}

export interface EarSpec { bands: Record<string, { cf: number; gain: number }>; ear_sos: Record<string, number[][][]>; env_sos: number[][] }

/** One channel per (band, cf-scale); `process` returns gain-weighted envelopes, layout [channel][sample]. */
export class EarBank {
  keys: string[] = [];
  private bp: Sos[] = [];
  private env: Sos[] = [];
  private gain: number[] = [];
  private bandpass: boolean[] = [];
  constructor(spec: EarSpec) {
    for (const [name, b] of Object.entries(spec.bands)) {
      spec.ear_sos[name].forEach((sos, scale) => {
        this.keys.push(`${name}:${scale}`);
        this.bp.push(new Sos(sos));
        this.env.push(new Sos(spec.env_sos));
        this.gain.push(b.gain);
        this.bandpass.push(b.cf > 0);
      });
    }
  }
  process(x: Float32Array): Float32Array[] {
    return this.bp.map((f, c) => {
      const out = new Float32Array(x.length), e = this.env[c], g = this.gain[c], bp = this.bandpass[c];
      for (let i = 0; i < x.length; i++) {
        const y = f.step(x[i]);
        out[i] = g * e.step(bp ? Math.max(y, 0) : Math.abs(y));
      }
      return out;
    });
  }
}

export function rms(x: Float32Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, x.length));
}

// Built-in stimuli (same definitions as pipeline/ear.py), at fs.
export function pulseSong(fs: number, dur: number, carrier: number, ipiMs: number, cycles = 3): Float32Array {
  const n = Math.round(fs * dur), x = new Float32Array(n), sigma = cycles / carrier / 4;
  for (let c = ipiMs / 2000; c < dur; c += ipiMs / 1000) {
    const i0 = Math.max(0, Math.floor((c - 5 * sigma) * fs)), i1 = Math.min(n, Math.ceil((c + 5 * sigma) * fs));
    for (let i = i0; i < i1; i++) { const t = i / fs - c; x[i] += Math.exp(-0.5 * (t / sigma) ** 2) * Math.sin(2 * Math.PI * carrier * t); }
  }
  return x;
}
export function ramp(x: Float32Array, fs: number, ms = 20): Float32Array {
  const n = Math.min(Math.round((fs * ms) / 1000), x.length >> 1);
  for (let i = 0; i < n; i++) { const w = Math.sin((Math.PI / 2) * (i / n)) ** 2; x[i] *= w; x[x.length - 1 - i] *= w; }
  return x;
}
export function sineSong(fs: number, dur: number, carrier: number): Float32Array {
  const x = new Float32Array(Math.round(fs * dur));
  for (let i = 0; i < x.length; i++) x[i] = Math.sin((2 * Math.PI * carrier * i) / fs);
  return ramp(x, fs);
}
export function whiteNoise(fs: number, dur: number): Float32Array {
  const x = new Float32Array(Math.round(fs * dur));
  for (let i = 0; i < x.length; i += 2) {
    const u = Math.random() || 1e-9, v = Math.random(), r = Math.sqrt(-2 * Math.log(u));
    x[i] = r * Math.cos(2 * Math.PI * v); if (i + 1 < x.length) x[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return ramp(x, fs);
}
/** Same amplitude spectrum, random phases (FFT on a zero-padded power of two). */
export function phaseScramble(x: Float32Array): Float32Array {
  const n = x.length; let N = 1; while (N < n) N <<= 1;
  const re = new Float64Array(N), im = new Float64Array(N);
  re.set(x); fft(re, im, false);
  for (let k = 1; k < N / 2; k++) {
    const m = Math.hypot(re[k], im[k]), p = Math.random() * 2 * Math.PI;
    re[k] = m * Math.cos(p); im[k] = m * Math.sin(p); re[N - k] = re[k]; im[N - k] = -im[k];
  }
  fft(re, im, true);
  return Float32Array.from(re.subarray(0, n));
}
export function fft(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
