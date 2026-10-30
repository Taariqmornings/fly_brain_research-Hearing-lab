// Data loading: manifest, per-sex neuron tables, CSR adjacency, precomputed runs.
export type Sex = "male" | "female";
export const SEXES: Sex[] = ["male", "female"];

export interface RunInfo {
  id: string; file: string; n_spikes: number; n: number; n_audio: number;
  stage_rates_shape: [number, number]; meta: Record<string, any>;
}
export interface Manifest {
  sexes: Record<Sex, { sex: Sex; dataset: string; n: number; nnz: number; runs: RunInfo[] }>;
  stages: string[]; stage_labels: Record<string, string>;
  ear: any; ear_sos: Record<string, number[][][]>; env_sos: number[][];
  sim: any; stimuli: any; config_hash: string; subnet_report: any; has_sweep: boolean;
}
export interface Brain {
  sex: Sex; dataset: string; n: number; nnz: number;
  ids: string[]; types: string[]; typeIdx: Int32Array; stage: Uint8Array; hop: Int16Array;
  nt: string[]; ntConf: Float32Array; sign: Int8Array;
  target: Uint8Array; song: Uint8Array; recep: Uint8Array; jo: Int32Array; joSubtype: string[];
  posMeasured: Uint8Array; positions: Float32Array; wSyn: number;
  indptr: Int32Array; indices: Int32Array; weights: Float32Array; syn: Int32Array;
  inPtr: Int32Array; inIdx: Int32Array; inSyn: Int32Array;
}
export interface Run {
  info: RunInfo; sex: Sex; ids: Uint32Array; t: Float32Array; rates: Float32Array;
  audio: Float32Array; stageRates: Float32Array; duration: number;
}

const base = "./data/";
async function bin(path: string): Promise<ArrayBuffer> {
  const r = await fetch(base + path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.arrayBuffer();
}
export async function loadManifest(): Promise<Manifest> {
  const r = await fetch(base + "manifest.json");
  if (!r.ok) throw new Error("manifest.json missing - run the pipeline export first");
  return r.json();
}

export async function loadBrain(sex: Sex): Promise<Brain> {
  const meta = await (await fetch(`${base}${sex}/neurons.json`)).json();
  const [pos, csr, syn] = await Promise.all([bin(`${sex}/positions.f32`), bin(`${sex}/csr.bin`), bin(`${sex}/csr_syn.i32`)]);
  const n = meta.n as number, nnz = meta.nnz as number;
  const flags = (idx: number[]) => { const a = new Uint8Array(n); for (const i of idx) a[i] = 1; return a; };
  const indptr = new Int32Array(csr, 0, n + 1);
  const indices = new Int32Array(csr, (n + 1) * 4, nnz);
  const weights = new Float32Array(csr, (n + 1 + nnz) * 4, nnz);
  const synA = new Int32Array(syn);
  // Transpose to CSC for inspector inputs.
  const inPtr = new Int32Array(n + 1);
  for (let k = 0; k < nnz; k++) inPtr[indices[k] + 1]++;
  for (let i = 0; i < n; i++) inPtr[i + 1] += inPtr[i];
  const fill = inPtr.slice(0, n), inIdx = new Int32Array(nnz), inSyn = new Int32Array(nnz);
  for (let p = 0; p < n; p++) for (let k = indptr[p]; k < indptr[p + 1]; k++) {
    const q = indices[k], at = fill[q]++; inIdx[at] = p; inSyn[at] = synA[k] * Math.sign(weights[k]);
  }
  return {
    sex, dataset: meta.dataset, n, nnz, ids: meta.ids, types: meta.types, typeIdx: Int32Array.from(meta.type_idx),
    stage: Uint8Array.from(meta.stage), hop: Int16Array.from(meta.hop), nt: meta.nt, ntConf: Float32Array.from(meta.nt_conf),
    sign: Int8Array.from(meta.sign), target: flags(meta.target), song: flags(meta.song), recep: flags(meta.recep),
    jo: Int32Array.from(meta.jo), joSubtype: meta.jo_subtype, posMeasured: Uint8Array.from(meta.pos_measured),
    positions: new Float32Array(pos), wSyn: meta.w_syn_mv, indptr, indices, weights, syn: synA, inPtr, inIdx, inSyn,
  };
}

const runCache = new Map<string, Run>();
export async function loadRun(sex: Sex, info: RunInfo): Promise<Run> {
  const key = info.file;
  const hit = runCache.get(key);
  if (hit) return hit;
  const buf = await bin(info.file);
  const ns = info.n_spikes, n = info.n, na = info.n_audio, [s, b] = info.stage_rates_shape;
  let o = 0;
  const take = <T>(C: new (b: ArrayBuffer, o: number, l: number) => T, len: number) => { const a = new C(buf, o, len); o += len * 4; return a; };
  const ids = take(Uint32Array, ns), t = take(Float32Array, ns), rates = take(Float32Array, n);
  const audio = take(Float32Array, na), stageRates = take(Float32Array, s * b);
  const run: Run = { info, sex, ids, t, rates, audio, stageRates, duration: info.meta.duration_s ?? na / 10000 };
  runCache.set(key, run);
  return run;
}

export async function loadSweep(): Promise<any | null> {
  try { const r = await fetch(base + "sweep.json"); return r.ok ? r.json() : null; } catch { return null; }
}
