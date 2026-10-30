// Live LIF (Shiu et al. 2024 update rule, same as pipeline/sim.py) on both subnetworks, event-driven delivery.
import { EarBank, EarSpec } from "./ear";

interface BrainIn {
  sex: string; n: number; indptr: Int32Array; indices: Int32Array; weights: Float32Array;
  jo: Int32Array; joSubtype: string[]; wSyn: number; priority: Int32Array;
}
interface SimP {
  dt_ms: number; v_rest_mv: number; v_reset_mv: number; v_threshold_mv: number; tau_membrane_ms: number;
  tau_synapse_ms: number; t_refractory_ms: number; delay_ms: number; jo_max_rate_hz: number; jo_baseline_hz: number;
}

class LiveBrain {
  v: Float32Array; g: Float32Array; rfc: Int16Array; ring: Float32Array; reload: Int16Array;
  joChan: Int32Array; joGain: Float32Array; active: Uint8Array; silenced: Uint8Array; boosted: Uint8Array;
  delay: number; kick: number; outSp: number[] = []; outT: number[] = [];
  constructor(public b: BrainIn, p: SimP, keys: string[], jitterGain: number, jitterCf: number) {
    const n = b.n;
    this.v = new Float32Array(n).fill(p.v_rest_mv); this.g = new Float32Array(n); this.rfc = new Int16Array(n);
    this.delay = Math.round(p.delay_ms / p.dt_ms);
    this.ring = new Float32Array((this.delay + 1) * n);
    this.reload = new Int16Array(n).fill(Math.round(p.t_refractory_ms / p.dt_ms));
    this.kick = b.wSyn * 250;
    this.joChan = new Int32Array(b.jo.length); this.joGain = new Float32Array(b.jo.length);
    const scales = jitterCf > 0 ? 3 : 1;
    b.jo.forEach((idx, k) => {
      this.reload[idx] = 0;
      const band = keys.some((s) => s.startsWith(b.joSubtype[k] + ":")) ? b.joSubtype[k] : "other";
      this.joChan[k] = keys.indexOf(`${band}:${Math.floor(Math.random() * scales)}`);
      this.joGain[k] = Math.exp(jitterGain * gauss());
    });
    this.active = new Uint8Array(n).fill(1); this.silenced = new Uint8Array(n); this.boosted = new Uint8Array(n);
  }
  setSubset(count: number) {
    this.active.fill(count > 0 ? 0 : 1);
    if (count > 0) for (let k = 0; k < Math.min(count, this.b.priority.length); k++) this.active[this.b.priority[k]] = 1;
  }
}

function gauss(): number { const u = Math.random() || 1e-9, v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

let P: SimP, bank: EarBank, keys: string[] = [], brains: LiveBrain[] = [], tStep = 0, refRms = 0.05;
let maxRate = 250, baseRate = 2;

function stepBrains(env: Float32Array[], nSamples: number) {
  const dt = P.dt_ms, aV = Math.exp(-dt / P.tau_membrane_ms), bG = Math.exp(-dt / P.tau_synapse_ms);
  const tm = P.tau_membrane_ms, ts = P.tau_synapse_ms, cGV = (ts / (ts - tm)) * (bG - aV);
  const v0 = P.v_rest_mv * (1 - aV), vth = P.v_threshold_mv, vr = P.v_reset_mv, pdt = dt / 1000;
  for (const L of brains) {
    const { v, g, rfc, ring, reload, active, silenced, boosted, delay, kick } = L;
    const { n, indptr, indices, weights, jo } = L.b;
    const slots = delay + 1;
    for (let s = 0; s < nSamples; s++) {
      const step = tStep + s, slot = (step % slots) * n, fut = ((step + delay) % slots) * n;
      // Poisson kicks: JO from the ear, boosted neurons at a fixed 150 Hz.
      for (let k = 0; k < jo.length; k++) {
        const i = jo[k];
        if (!active[i]) continue;
        const e = (env[L.joChan[k]][s] * L.joGain[k]) / refRms, d = e > 0 ? e / (1 + e) : 0;
        if (Math.random() < (baseRate + maxRate * d) * pdt) ring[slot + i] += 1e6; // marker, handled below
      }
      for (let i = 0; i < n; i++) {
        if (!active[i]) continue;
        let syn = ring[slot + i]; ring[slot + i] = 0;
        let kicked = false;
        if (syn >= 5e5) { syn -= 1e6; kicked = true; }
        if (boosted[i] && Math.random() < 150 * pdt) kicked = true;
        let r = rfc[i] - 1; if (r < 0) r = 0;
        let spike = false;
        if (r === 0) {
          const vn = v0 + g[i] * cGV + v[i] * aV;
          g[i] = g[i] * bG + syn;
          v[i] = vn;
          if (vn > vth) spike = true;
        }
        if (spike) {
          v[i] = vr; g[i] = 0; rfc[i] = reload[i];
          L.outSp.push(i); L.outT.push(step * dt / 1000);
          if (!silenced[i]) for (let e = indptr[i]; e < indptr[i + 1]; e++) ring[fut + indices[e]] += weights[e];
        } else {
          rfc[i] = r;
          if (kicked && r === 0) v[i] += kick;
        }
      }
    }
  }
  tStep += nSamples;
  for (const L of brains) {
    const sp = Uint32Array.from(L.outSp), t = Float32Array.from(L.outT);
    L.outSp = []; L.outT = [];
    (self as any).postMessage({ type: "spikes", sex: L.b.sex, ids: sp, t, simTime: (tStep * P.dt_ms) / 1000 }, [sp.buffer, t.buffer]);
  }
}

self.onmessage = (ev: MessageEvent) => {
  const m = ev.data;
  if (m.type === "init") {
    P = m.sim; refRms = m.ear.drive_ref_rms; maxRate = P.jo_max_rate_hz; baseRate = P.jo_baseline_hz;
    bank = new EarBank(m.ear as EarSpec); keys = bank.keys;
    brains = (m.brains as BrainIn[]).map((b) => new LiveBrain(b, P, keys, m.ear.jitter_gain_sd, m.ear.jitter_cf_sd));
    tStep = 0;
    (self as any).postMessage({ type: "ready", keys });
  } else if (m.type === "audio") {
    // Raw samples at the ear rate; normalise by the running reference so loudness matches the pipeline's unit RMS.
    const x = m.samples as Float32Array, gain = m.gain ?? 1;
    if (gain !== 1) for (let i = 0; i < x.length; i++) x[i] *= gain;
    stepBrains(bank.process(x), x.length);
  } else if (m.type === "env") {
    stepBrains(m.env as Float32Array[], m.env[0].length);
  } else if (m.type === "reset") {
    for (const L of brains) { L.v.fill(P.v_rest_mv); L.g.fill(0); L.rfc.fill(0); L.ring.fill(0); }
    tStep = 0;
  } else if (m.type === "silence" || m.type === "boost") {
    const L = brains.find((b) => b.b.sex === m.sex);
    if (L) (m.type === "silence" ? L.silenced : L.boosted)[m.idx] = m.on ? 1 : 0;
  } else if (m.type === "subset") {
    for (const L of brains) L.setSubset(m.count);
  }
};
