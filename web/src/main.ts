// Fly Hearing Lab: app shell, replay/live orchestration, Experiment / Validate / Notebook views.
import "./styles.css";
import { Brain, loadBrain, loadManifest, loadRun, loadSweep, Manifest, Run, Sex, SEXES } from "./data";
import { BrainScene, View } from "./scene";
import { Timeline, Probe } from "./timeline";
import { Inspector } from "./inspector";
import { NotebookView, download, runToCsv, runToJson } from "./notebook";
import { decodeFile, FS, MicHandle, normalise, play, startMic, StimKind, synth } from "./audio";

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector(s) as T;
const PROBE_COLORS = ["#40d9ff", "#ffb833", "#9ac48f", "#e58f8f", "#c48fe5", "#f0f0f0"];

let M: Manifest;
const brains = new Map<Sex, Brain>();
let sweep: any = null;
let scene: BrainScene, timeline: Timeline, inspector: Inspector;

type Source = "library" | "mic" | "file" | "synth";
const S = {
  source: "library" as Source, playing: false, speed: 10, time: 0, view: "side" as View, clean: false,
  runId: "pulse35", runs: new Map<Sex, Run>(), cursor: new Map<Sex, number>(),
  liveAudio: null as Float32Array | null, livePos: 0, liveSimTime: 0, liveLabel: "", mic: null as MicHandle | null,
  player: null as { stop(): void } | null, pending: 0, slowFrames: 0, reduced: false, probes: [] as Probe[],
  stageRate: new Map<Sex, Float32Array>(), winRates: new Map<Sex, Float32Array>(), onset: -1,
  latency: new Map<Sex, number>(), synth: { kind: "pulse" as StimKind, ipi: 35, carrier: 220, dur: 2 },
};
let worker: Worker | null = null;

// ---------- boot ----------
async function boot() {
  try {
    M = await loadManifest();
  } catch (e) {
    $("#boot").innerHTML = `<b>No data.</b> Run <code>python pipeline/run_all.py --quick</code> first.<br><span class="muted">${String(e)}</span>`;
    return;
  }
  const loaded = await Promise.all(SEXES.map(loadBrain));
  loaded.forEach((b) => brains.set(b.sex, b));
  sweep = await loadSweep();
  for (const b of loaded) { S.stageRate.set(b.sex, new Float32Array(M.stages.length)); S.winRates.set(b.sex, new Float32Array(b.n)); }
  scene = new BrainScene($("#viewport"));
  scene.setBrains(loaded);
  scene.onFrame = frame;
  timeline = new Timeline($<HTMLCanvasElement>("#timeline"), (t) => seek(t));
  timeline.setBrains(loaded);
  inspector = new Inspector($("#inspector"), brains, {
    select: (sex, i) => selectNeuron(sex, i), pin: pinProbe,
    perturb: (kind, sex, idx, on) => worker?.postMessage({ type: kind, sex, idx, on }),
    rateTrace, tuning, isLive: () => S.source !== "library", stageLabel: (s) => M.stage_labels[M.stages[s]],
  });
  new NotebookView($("#tab-notebook"), notebookContext, exportRun);
  buildChrome();
  renderExperiment();
  renderValidate();
  $("#boot").remove();
  await loadLibraryRun(M.sexes.male.runs.some((r) => r.id === "pulse35") ? "pulse35" : M.sexes.male.runs[0]?.id);
  // Scripting hook for docs screenshots and debugging (scripts/screenshots.mjs).
  (window as any).flyLab = {
    brains, select: selectNeuron, pin: pinProbe, seek, setView, setSpeed, loadRun: loadLibraryRun,
    preset: (p: string) => scene.preset(p as any), pause: () => { if (S.playing) togglePlay(); },
    stage: (i: number) => document.querySelector<HTMLButtonElement>(`.stage[data-stage="${i}"]`)?.click(),
  };
}

// ---------- chrome ----------
function buildChrome() {
  $("#datasets").innerHTML = SEXES.map((s) => `<span class="chip"><span class="dot ${s}"></span>${s === "male" ? "MaleCNS v1.0" : "FlyWire FAFB v783"} · ${M.sexes[s].n.toLocaleString()} n</span>`).join("");
  const runSel = $<HTMLSelectElement>("#run-select");
  const runs = M.sexes.male.runs.map((r) => r.id);
  const named: Record<string, string> = { pulse35: "Pulse song · 35 ms IPI · 220 Hz", scrambled35: "Phase-scrambled pulse song", noise: "White noise", sine: "Sine song · 150 Hz" };
  runSel.innerHTML = `<optgroup label="Reference stimuli">${runs.filter((r) => named[r]).map((r) => `<option value="${r}">${named[r]}</option>`).join("")}</optgroup>
    <optgroup label="Sweep: pulse song">${runs.filter((r) => r.startsWith("pulse_")).map((r) => `<option value="${r}">${label(r)}</option>`).join("")}</optgroup>
    <optgroup label="Sweep: phase-scrambled">${runs.filter((r) => r.startsWith("scr_")).map((r) => `<option value="${r}">${label(r)}</option>`).join("")}</optgroup>`;
  runSel.addEventListener("change", () => loadLibraryRun(runSel.value));
  seg("#source", (v) => setSource(v as Source));
  seg("#speed", (v) => setSpeed(+v));
  seg("#view", (v) => setView(v as View));
  seg("#tabs", (v) => setTab(v));
  $("#play").addEventListener("click", togglePlay);
  $("#restart").addEventListener("click", () => { seek(0); if (S.source !== "library") restartLive(); });
  $("#clean").addEventListener("click", () => { S.clean = !S.clean; $("#clean").classList.toggle("on", S.clean); document.body.classList.toggle("clean", S.clean); scene.setClean(S.clean); });
  $<HTMLInputElement>("#file-input").addEventListener("change", async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0]; if (!f) return;
    S.liveLabel = f.name; startLive(normalise(await decodeFile(f)));
  });
  $("#synth-go").addEventListener("click", () => {
    const g = (id: string) => +$<HTMLInputElement>(id).value;
    S.synth = { kind: $<HTMLSelectElement>("#syn-kind").value as StimKind, ipi: g("#syn-ipi"), carrier: g("#syn-carrier"), dur: g("#syn-dur") };
    S.liveLabel = `${S.synth.kind} ${S.synth.kind === "pulse" || S.synth.kind === "scrambled" ? S.synth.ipi + " ms · " : ""}${S.synth.carrier} Hz`;
    startLive(normalise(synth(S.synth.kind, S.synth.dur, S.synth.carrier, S.synth.ipi)));
  });
  for (const id of ["#syn-ipi", "#syn-carrier", "#syn-dur"]) {
    const i = $<HTMLInputElement>(id), o = $(id + "-v"); const upd = () => (o.textContent = i.value); i.addEventListener("input", upd); upd();
  }
  $<HTMLInputElement>("#mic-gain").addEventListener("input", (e) => S.mic?.setGain(+(e.target as HTMLInputElement).value));
  document.querySelectorAll<HTMLButtonElement>("[data-cam]").forEach((b) => b.addEventListener("click", () => scene.preset(b.dataset.cam as any)));
  $("#viewport").addEventListener("click", (e) => {
    const h = scene.pick(e.clientX, e.clientY);
    if (h) selectNeuron(h.sex, h.idx); else { scene.select(null, []); inspector.empty(); }
  });
  $("#pathway").innerHTML = M.stages.map((s, i) => `
    <button class="stage" data-stage="${i}"><span class="sname">${i ? "↓ " : ""}${M.stage_labels[s]}</span>
      ${SEXES.map((sex) => `<span class="bar ${sex}"><i id="bar-${sex}-${i}"></i></span><span class="num mono" id="num-${sex}-${i}">0</span>`).join("")}</button>`).join("") +
    `<div class="legend"><span><span class="dot male"></span>male</span><span><span class="dot female"></span>female</span><span class="muted">spikes/s per neuron</span></div>`;
  document.querySelectorAll<HTMLButtonElement>(".stage").forEach((b) => b.addEventListener("click", () => {
    const st = +b.dataset.stage!, on = !b.classList.contains("on");
    document.querySelectorAll(".stage").forEach((x) => x.classList.remove("on"));
    if (on) b.classList.add("on");
    for (const [sex, br] of brains) scene.setVisible(sex, on ? Uint8Array.from(br.stage, (v) => +(v === st || v === 0 && st === 0)) : null);
  }));
  window.addEventListener("keydown", (e) => {
    if ((e.target as HTMLElement).closest("input,textarea,select")) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.key === "1") setView("side"); if (e.key === "2") setView("overlay"); if (e.key === "3") setView("diff");
    if (e.key.toLowerCase() === "c") $("#clean").click();
  });
}

function seg(sel: string, cb: (v: string) => void) {
  const el = $(sel);
  el.querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.addEventListener("click", () => {
    el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); cb(b.dataset.v!);
  }));
}
function segSet(sel: string, v: string) { $(sel).querySelectorAll<HTMLButtonElement>("button").forEach((x) => x.classList.toggle("on", x.dataset.v === v)); }
function label(id: string) { const m = id.match(/ipi(\d+)_f(\d+)/); return m ? `IPI ${m[1]} ms · ${m[2]} Hz` : id; }

function setTab(t: string) {
  document.querySelectorAll<HTMLElement>(".tab").forEach((x) => (x.hidden = x.id !== `tab-${t}`));
  document.body.dataset.tab = t;
  if (t === "explore") scene.resize();
  if (t === "experiment") renderExperiment();
}
function setSpeed(v: number) {
  S.speed = v; segSet("#speed", String(v));
  if (S.source === "file" || S.source === "synth") { S.player?.stop(); if (S.playing && S.liveAudio) S.player = play(S.liveAudio.subarray(S.livePos), S.speed); }
}
function setView(v: View) { S.view = v; segSet("#view", v); scene.setView(v); if (v === "diff") updateDiff(); $("#diff-legend").hidden = v !== "diff"; }

// ---------- sources ----------
function setSource(src: Source) {
  stopLive();
  S.source = src; segSet("#source", src);
  document.querySelectorAll<HTMLElement>("[data-src]").forEach((x) => (x.hidden = x.dataset.src !== src));
  $("#mode-pill").textContent = src === "library" ? "REPLAY" : "LIVE";
  $("#mode-pill").classList.toggle("live", src !== "library");
  if (src === "library") loadLibraryRun(S.runId);
  else if (src === "mic") startMicLive();
  if (inspector.sel) inspector.show(inspector.sel.sex, inspector.sel.idx);
}

async function loadLibraryRun(id: string | undefined) {
  if (!id) return;
  S.runId = id;
  $<HTMLSelectElement>("#run-select").value = id;
  const runs = new Map<Sex, Run>();
  for (const sex of SEXES) { const info = M.sexes[sex].runs.find((r) => r.id === id); if (info) runs.set(sex, await loadRun(sex, info)); }
  S.runs = runs; S.time = 0; S.cursor = new Map(SEXES.map((s) => [s, 0]));
  const r0 = [...runs.values()][0];
  S.onset = r0 ? onsetOf(r0.audio) : -1;
  S.latency.clear();
  for (const [sex, r] of runs) { const b = brains.get(sex)!; for (let k = 0; k < r.ids.length; k++) if (b.target[r.ids[k]] && r.t[k] >= S.onset) { S.latency.set(sex, r.t[k] - S.onset); break; } }
  timeline.loadReplay(runs);
  if (S.view === "diff") updateDiff();
  S.playing = true; $("#play").classList.add("on");
  $("#run-meta").textContent = r0 ? `${r0.info.meta.input_hash} · cfg ${r0.info.meta.config_hash} · seed ${r0.info.meta.seed} · git ${r0.info.meta.git_commit}` : "";
  if (inspector.sel) inspector.show(inspector.sel.sex, inspector.sel.idx);
}

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./liveSim.worker.ts", import.meta.url), { type: "module" });
  const prio = (b: Brain) => Int32Array.from(Array.from({ length: b.n }, (_, i) => i).sort((a, c) =>
    (b.target[c] - b.target[a]) || ((b.hop[a] < 0 ? 9 : b.hop[a]) - (b.hop[c] < 0 ? 9 : b.hop[c])) || ((b.indptr[c + 1] - b.indptr[c]) - (b.indptr[a + 1] - b.indptr[a]))));
  worker.postMessage({
    type: "init", sim: M.sim, ear: { ...M.ear, ear_sos: M.ear_sos, env_sos: M.env_sos },
    brains: [...brains.values()].map((b) => ({ sex: b.sex, n: b.n, indptr: b.indptr, indices: b.indices, weights: b.weights, jo: b.jo, joSubtype: b.joSubtype, wSyn: b.wSyn, priority: prio(b) })),
  });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type !== "spikes") return;
    S.pending = Math.max(0, S.pending - 0.5);
    S.liveSimTime = m.simTime;
    onSpikes(m.sex, m.ids, m.t);
    timeline.pushLive(m.sex, m.ids, m.t, m.simTime);
  };
  return worker;
}

function startLive(x: Float32Array) {
  stopLive();
  S.liveAudio = x; S.livePos = 0; S.onset = onsetOf(x); S.latency.clear(); S.time = 0;
  ensureWorker().postMessage({ type: "reset" });
  timeline.setLive();
  S.playing = true; $("#play").classList.add("on");
  S.player = play(x, S.speed);
}
function restartLive() { if (S.liveAudio) startLive(S.liveAudio); }
function stopLive() {
  S.player?.stop(); S.player = null; S.mic?.stop(); S.mic = null; S.liveAudio = null;
  if (S.reduced) { worker?.postMessage({ type: "subset", count: 0 }); for (const s of SEXES) scene.setVisible(s, null); S.reduced = false; $("#perf").hidden = true; }
}

async function startMicLive() {
  ensureWorker().postMessage({ type: "reset" });
  timeline.setLive(); S.onset = -1; S.latency.clear();
  setSpeed(1);
  try {
    S.mic = await startMic({ ...M.ear, ear_sos: M.ear_sos, env_sos: M.env_sos }, (env, raw, level) => {
      if (!S.playing) return;
      if (S.onset < 0 && level > 0.3) S.onset = S.liveSimTime;
      worker!.postMessage({ type: "env", env }, env.map((e) => e.buffer));
      timeline.pushAudio(raw);
      $("#mic-level").style.width = `${Math.min(100, level * 50)}%`;
    });
    S.mic.setGain(+$<HTMLInputElement>("#mic-gain").value);
    S.playing = true; $("#play").classList.add("on");
  } catch (e) {
    toast(`Microphone unavailable: ${(e as Error).message}`);
  }
}

function togglePlay() {
  S.playing = !S.playing; $("#play").classList.toggle("on", S.playing);
  if (S.source === "file" || S.source === "synth") {
    S.player?.stop(); S.player = null;
    if (S.playing && S.liveAudio) { if (S.livePos >= S.liveAudio.length) restartLive(); else S.player = play(S.liveAudio.subarray(S.livePos), S.speed); }
  }
}

function seek(t: number) {
  if (S.source !== "library") return;
  S.time = t;
  for (const [sex, r] of S.runs) S.cursor.set(sex, lowerBound(r.t, t));
  for (const sex of SEXES) S.winRates.get(sex)!.fill(0);
}

// ---------- per-frame ----------
let lastUi = 0, lastDiff = 0;
function frame(dt: number) {
  if (S.source === "library") stepReplay(dt);
  else if ((S.source === "file" || S.source === "synth") && S.playing && S.liveAudio && worker) {
    if (S.pending < 3 && S.livePos < S.liveAudio.length) {
      const n = Math.min(Math.round((dt * FS) / S.speed), 1200, S.liveAudio.length - S.livePos);
      if (n > 0) {
        const chunk = S.liveAudio.slice(S.livePos, S.livePos + n);
        worker.postMessage({ type: "audio", samples: chunk });
        timeline.pushAudio(chunk);
        S.livePos += n; S.pending += 1;
      }
    }
    if (S.livePos >= S.liveAudio.length && S.pending < 0.5) { S.playing = false; $("#play").classList.remove("on"); }
    S.time = S.liveSimTime;
  } else if (S.source === "mic") S.time = S.liveSimTime;
  // Stage-rate smoothing (exponential, ~100 ms of sim time).
  const tau = 0.1, k = Math.exp(-(S.source === "library" ? dt / S.speed : dt / Math.max(1, S.speed)) / tau);
  for (const sex of SEXES) { const a = S.stageRate.get(sex)!; for (let i = 0; i < a.length; i++) a[i] *= k; const w = S.winRates.get(sex)!; for (let i = 0; i < w.length; i++) w[i] *= k; }
  if (S.source !== "library" && S.playing) {
    if (scene.fps < 30) S.slowFrames++; else S.slowFrames = Math.max(0, S.slowFrames - 1);
    if (!S.reduced && S.slowFrames > 120) {
      S.reduced = true; worker?.postMessage({ type: "subset", count: 3000 }); $("#perf").hidden = false;
      for (const b of brains.values()) { const m = new Uint8Array(b.n); const p = prioList(b); for (let i = 0; i < 3000 && i < p.length; i++) m[p[i]] = 1; scene.setVisible(b.sex, m); }
    }
  }
  timeline.time = S.time;
  timeline.draw();
  const now = performance.now();
  if (now - lastUi > 100) { lastUi = now; updateHud(); inspector.refreshTrace(); }
  if (S.view === "diff" && S.source !== "library" && now - lastDiff > 300) { lastDiff = now; updateDiff(); }
}

function prioList(b: Brain): number[] {
  return Array.from({ length: b.n }, (_, i) => i).sort((a, c) => (b.target[c] - b.target[a]) || ((b.hop[a] < 0 ? 9 : b.hop[a]) - (b.hop[c] < 0 ? 9 : b.hop[c])));
}

function stepReplay(dt: number) {
  const r0 = [...S.runs.values()][0]; if (!r0) return;
  if (S.playing) {
    const t1 = S.time + dt / S.speed;
    for (const [sex, r] of S.runs) {
      let c = S.cursor.get(sex) ?? 0; const start = c;
      while (c < r.t.length && r.t[c] < t1) c++;
      S.cursor.set(sex, c);
      if (c > start) onSpikes(sex, r.ids.subarray(start, c), r.t.subarray(start, c));
    }
    S.time = t1;
    if (S.time >= r0.duration) { S.time = 0; S.cursor = new Map(SEXES.map((s) => [s, 0])); }
  }
}

function onSpikes(sex: Sex, ids: Uint32Array, t: Float32Array) {
  scene.spike(sex, ids);
  const b = brains.get(sex)!, a = S.stageRate.get(sex)!, w = S.winRates.get(sex)!;
  const tau = 0.1;
  for (let k = 0; k < ids.length; k++) { a[b.stage[ids[k]]] += 1 / tau; w[ids[k]] += 1 / tau; }
  if (S.source !== "library" && S.onset >= 0 && !S.latency.has(sex)) {
    for (let k = 0; k < ids.length; k++) if (b.target[ids[k]] && t[k] >= S.onset) { S.latency.set(sex, t[k] - S.onset); break; }
  }
}

const stageCounts = new Map<Sex, number[]>();
function updateHud() {
  for (const [sex, b] of brains) {
    if (!stageCounts.has(sex)) { const c = M.stages.map(() => 0); for (let i = 0; i < b.n; i++) c[b.stage[i]]++; stageCounts.set(sex, c); }
    const c = stageCounts.get(sex)!, a = S.stageRate.get(sex)!;
    M.stages.forEach((_, i) => {
      const r = a[i] / Math.max(1, c[i]);
      $(`#bar-${sex}-${i}`).style.width = `${Math.min(100, (Math.log1p(r) / Math.log1p(100)) * 100)}%`;
      $(`#num-${sex}-${i}`).textContent = r.toFixed(r < 10 ? 1 : 0);
    });
  }
  const lat = (s: Sex) => S.latency.has(s) ? `${(S.latency.get(s)! * 1000).toFixed(1)} ms` : "—";
  const src = S.source === "library" ? label(S.runId) === S.runId ? S.runId : label(S.runId) : S.source === "mic" ? "microphone" : S.liveLabel;
  $("#hud-tl").innerHTML = `<div><b>T</b> ${(S.time * 1000).toFixed(1).padStart(7)} ms</div><div><b>SPEED</b> ${S.speed === 1 ? "real time" : `${S.speed}× slow-mo`}</div><div><b>INPUT</b> ${src}</div>`;
  $("#hud-br").innerHTML = `<div><b>SOUND→TARGET</b> <span class="male-t">♂ ${lat("male")}</span> <span class="female-t">♀ ${lat("female")}</span></div><div><b>FPS</b> ${scene.fps.toFixed(0)}${S.reduced ? " · reduced to 3000 neurons" : ""}</div>`;
}

function updateDiff() {
  // Per-cell-type mean rate; neuron value = normalised (male − female) of its type, NaN if the type is missing in the other sex.
  const typeRate = new Map<Sex, Map<string, number>>();
  for (const [sex, b] of brains) {
    const rates = S.source === "library" ? S.runs.get(sex)?.rates : S.winRates.get(sex);
    const sum = new Map<string, [number, number]>();
    for (let i = 0; i < b.n; i++) { const t = b.types[b.typeIdx[i]]; if (!t) continue; const e = sum.get(t) ?? [0, 0]; e[0] += rates ? rates[i] : 0; e[1]++; sum.set(t, e); }
    typeRate.set(sex, new Map([...sum].map(([t, [s, n]]) => [t, s / n])));
  }
  for (const [sex, b] of brains) {
    const d = new Float32Array(b.n), m = typeRate.get("male")!, f = typeRate.get("female")!;
    for (let i = 0; i < b.n; i++) {
      const t = b.types[b.typeIdx[i]];
      if (!t || !m.has(t) || !f.has(t)) { d[i] = NaN; continue; }
      const x = m.get(t)! - f.get(t)!; d[i] = Math.tanh(x / 10);
    }
    scene.diff.set(sex, d);
  }
  scene.recolor();
}

// ---------- selection, probes ----------
function selectNeuron(sex: Sex, idx: number) { scene.select(sex, [idx]); inspector.show(sex, idx); }
function pinProbe(sex: Sex, idx: number) {
  if (S.probes.some((p) => p.sex === sex && p.idx === idx)) return;
  const b = brains.get(sex)!;
  S.probes.push({ sex, idx, label: `${sex === "male" ? "♂" : "♀"} ${b.types[b.typeIdx[idx]] || b.ids[idx]}`, color: PROBE_COLORS[S.probes.length % PROBE_COLORS.length] });
  if (S.probes.length > 6) S.probes.shift();
  timeline.setProbes(S.probes);
  $("#probes").innerHTML = S.probes.map((p, i) => `<span class="chip" style="border-color:${p.color}">${p.label}<button data-i="${i}" title="unpin">×</button></span>`).join("");
  $("#probes").querySelectorAll<HTMLButtonElement>("button").forEach((x) => x.addEventListener("click", () => { S.probes.splice(+x.dataset.i!, 1); timeline.setProbes(S.probes); x.parentElement!.remove(); }));
}

function rateTrace(sex: Sex, idx: number): Float32Array | null {
  const r = S.runs.get(sex);
  if (S.source === "library" && r) {
    const bins = 60, tr = new Float32Array(bins);
    for (let k = 0; k < r.ids.length; k++) if (r.ids[k] === idx) tr[Math.min(bins - 1, Math.floor((r.t[k] / r.duration) * bins))] += bins / r.duration;
    return tr;
  }
  if (S.source !== "library") {
    const h = (traceHist.get(`${sex}:${idx}`) ?? []).concat([S.winRates.get(sex)![idx]]).slice(-80);
    traceHist.set(`${sex}:${idx}`, h);
    return Float32Array.from(h);
  }
  return null;
}
const traceHist = new Map<string, number[]>();

async function tuning(sex: Sex, idx: number) {
  if (!sweep) return null;
  const g = sweep.meta.grid, ipis: number[] = g.ipi_ms, carriers: number[] = g.carrier_hz, rates: number[][] = [];
  for (const i of ipis) {
    const row: number[] = [];
    for (const f of carriers) {
      const info = M.sexes[sex].runs.find((r) => r.id === `pulse_ipi${i}_f${f}`);
      row.push(info ? (await loadRun(sex, info)).rates[idx] : 0);
    }
    rates.push(row);
  }
  return { ipis, carriers, rates };
}

// ---------- Experiment / Validate ----------
function renderExperiment() {
  const el = $("#tab-experiment");
  if (!sweep) { el.innerHTML = `<div class="card">No sweep results yet — run <code>python pipeline/run_all.py --quick</code>.</div>`; return; }
  const g = sweep.meta.grid;
  el.innerHTML = `<div class="exp-head"><h2>IPI × carrier sweep</h2><p class="muted">Mean firing rate of the song/receptivity target set (${sweep.meta.mode} mode, ${sweep.meta.seeds.length} seeds, ${g.duration_s}s per trial). Click a cell to replay that run (seed 0) in Explore. <b>Model predictions.</b></p></div>
    <div class="heat-grid">${SEXES.map((sex) => ["grid_mean", "grid_scrambled_mean"].map((k) => `
      <figure class="card heat"><figcaption><span class="dot ${sex}"></span>${sex} · ${k === "grid_mean" ? "pulse song" : "phase-scrambled control"}
      <span class="muted">${k === "grid_mean" ? peakText(sweep.per_sex[sex]?.peak) : ""}</span></figcaption>
      <canvas data-sex="${sex}" data-k="${k}"></canvas></figure>`).join("")).join("")}</div>`;
  el.querySelectorAll<HTMLCanvasElement>("canvas").forEach((c) => drawHeat(c, c.dataset.sex as Sex, c.dataset.k!));
}
function peakText(p: any) { return p ? `peak ${p.ipi_ms} ms / ${p.carrier_hz} Hz · ${p.rate_hz.toFixed(2)} Hz` : "no response (flat)"; }

function drawHeat(c: HTMLCanvasElement, sex: Sex, key: string) {
  const g = sweep.meta.grid, data: number[][] = sweep.per_sex[sex]?.[key] ?? [];
  const draw = () => {
    const dpr = Math.min(devicePixelRatio, 2), w = c.clientWidth, h = c.clientHeight; c.width = w * dpr; c.height = h * dpr;
    const x = c.getContext("2d")!; x.setTransform(dpr, 0, 0, dpr, 0, 0);
    let max = 1e-6; for (const sexK of SEXES) for (const r of sweep.per_sex[sexK]?.[key] ?? []) for (const v of r) max = Math.max(max, v);
    const L = 48, B = 26, cw = (w - L - 8) / g.carrier_hz.length, ch = (h - B - 8) / g.ipi_ms.length;
    x.font = "10px 'JetBrains Mono', monospace";
    data.forEach((row, i) => row.forEach((v, j) => {
      x.fillStyle = viridis(v / max); x.fillRect(L + j * cw, h - B - (i + 1) * ch, cw - 1, ch - 1);
      if (cw > 34) { x.fillStyle = v / max > 0.6 ? "#000" : "#ddd"; x.fillText(v.toFixed(2), L + j * cw + 3, h - B - i * ch - 5); }
    }));
    x.fillStyle = "#7d8794";
    g.ipi_ms.forEach((v: number, i: number) => x.fillText(`${v}`, 8, h - B - i * ch - ch / 2 + 3));
    g.carrier_hz.forEach((v: number, j: number) => x.fillText(`${v}`, L + j * cw + 2, h - 10));
    x.fillText("IPI ms", 4, 10); x.fillText("carrier Hz →", w - 80, h - 0.5);
    c.onclick = (e) => {
      const r = c.getBoundingClientRect(), j = Math.floor((e.clientX - r.left - L) / cw), i = Math.floor((h - B - (e.clientY - r.top)) / ch);
      if (i < 0 || j < 0 || i >= g.ipi_ms.length || j >= g.carrier_hz.length) return;
      const id = `${key === "grid_mean" ? "pulse" : "scr"}_ipi${g.ipi_ms[i]}_f${g.carrier_hz[j]}`;
      if (S.source !== "library") setSource("library");
      setTab("explore"); segSet("#tabs", "explore"); loadLibraryRun(id);
    };
  };
  requestAnimationFrame(draw);
}
function viridis(t: number): string {
  const s = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  t = Math.max(0, Math.min(1, t)) * 4; const i = Math.min(3, Math.floor(t)), f = t - i;
  return `rgb(${s[i].map((v, k) => Math.round(v + (s[i + 1][k] - v) * f)).join(",")})`;
}

function renderValidate() {
  const el = $("#tab-validate");
  const rep = M.subnet_report ?? {};
  const subnet = SEXES.map((sex) => {
    const r = rep[sex]; if (!r) return "";
    return `<div class="card"><h3><span class="dot ${sex}"></span>${sex} · ${r.dataset}</h3>
      <dl class="kv"><dt>Subnetwork</dt><dd>${r.n_neurons.toLocaleString()} neurons · ${r.n_edges.toLocaleString()} edges</dd>
      <dt>JO seeds</dt><dd>${Object.entries(r.jo_subtypes).map(([k, v]) => `${k}:${v}`).join(" · ")}</dd>
      ${Object.entries(r.targets).map(([k, v]: [string, any]) => `<dt>${k.replace("_", " ")}</dt><dd>${v.in_dataset} in dataset, ${v.within_hops} within 3 hops<br><span class="muted small">${v.types.slice(0, 14).join(", ") || "none found"}</span></dd>`).join("")}</dl></div>`;
  }).join("");
  if (!sweep) { el.innerHTML = `<div class="val-grid">${subnet}</div><div class="card">No sweep results yet.</div>`; return; }
  const ci = (e: any) => `${e.mean.toFixed(2)} <span class="muted">[${e.ci[0].toFixed(2)}, ${e.ci[1].toFixed(2)}]</span>`;
  const val = SEXES.map((sex) => {
    const v = sweep.validation[sex]; if (!v) return "";
    const refs = sweep.per_sex[sex].refs; let max = 0.1; for (const k in refs) max = Math.max(max, refs[k].ci[1]);
    return `<div class="card"><h3><span class="dot ${sex}"></span>${sex} validation <span class="badge ${v.result === "PASS" ? "ok" : "bad"}">${v.result}</span></h3>
      <p class="muted small">Does the target set fire more for pulse song (35 ms IPI, 220 Hz) than for phase-scrambled audio and white noise? Bootstrap 95% CI over seeds; parameters were not tuned to this test.</p>
      <div class="bars">${Object.entries(refs).map(([k, r]: [string, any]) => `<div class="brow"><span>${k}</span><span class="track"><i style="left:${(r.ci[0] / max) * 100}%;width:${Math.max(0.5, ((r.ci[1] - r.ci[0]) / max) * 100)}%"></i><b style="left:${(r.mean / max) * 100}%"></b></span><span class="mono">${r.mean.toFixed(2)} Hz</span></div>`).join("")}</div>
      <dl class="kv"><dt>pulse − scrambled</dt><dd>${ci(v.pulse_gt_scrambled)}</dd><dt>pulse − noise</dt><dd>${ci(v.pulse_gt_noise)}</dd></dl></div>`;
  }).join("");
  const cols = ["baseline", "thr10", "signflip", "shuffled", "scrambled_audio"];
  const colName: Record<string, string> = { baseline: "Baseline (w≥5)", thr10: "Weight ≥ 10", signflip: "Low-conf signs flipped", shuffled: "Shuffled connectome", scrambled_audio: "Scrambled audio" };
  const rows = Object.entries(sweep.robust).map(([k, r]: [string, any]) => `<tr><th>${k.replace(":", " · ").replace(/_/g, " ")}</th>
    ${cols.map((c) => r.effects[c] ? `<td class="${r.effects[c].sig ? "sig" : ""}">${ci(r.effects[c])}<div class="muted small">d=${Number.isFinite(r.effects[c].d) ? r.effects[c].d.toFixed(2) : "∞"}</div></td>` : `<td class="muted">—</td>`).join("")}
    <td>${Object.entries(r.checks).map(([ck, ok]) => `<div class="chk ${ok ? "ok" : "bad"}">${ok ? "✓" : "✗"} ${ck.replace(/_/g, " ")}</div>`).join("")}</td>
    <td><span class="badge ${r.robust ? "ok" : "bad"}">${r.robust ? "ROBUST" : "NOT ROBUST"}</span></td></tr>`).join("");
  el.innerHTML = `<div class="val-grid">${val}</div>
    <div class="card"><h3>Robustness of effects</h3><p class="muted small">Robust = significant at baseline, same sign and significant with weight ≥ 10 and with low-confidence transmitter signs flipped, and absent or smaller on a degree-preserving shuffled connectome (and, for IPI tuning, with phase-scrambled audio). Cells: mean [95% CI], Hz.</p>
    <div class="scroll"><table class="rob"><thead><tr><th>Effect</th>${cols.map((c) => `<th>${colName[c]}</th>`).join("")}<th>Checks</th><th></th></tr></thead><tbody>${rows}</tbody></table></div></div>
    <h3 class="sec">Subnetworks & target cell types</h3><div class="val-grid">${subnet}</div>
    <p class="muted small">Sweep ${sweep.meta.mode} · config ${sweep.meta.config_hash} · git ${sweep.meta.git_commit} · ${sweep.meta.created} · runtime ${sweep.meta.runtime_s}s</p>`;
}

// ---------- notebook / export ----------
function notebookContext() {
  const r = S.runs.get("male") ?? S.runs.get("female");
  return S.source === "library" && r
    ? { mode: "replay", run: S.runId, t_ms: +(S.time * 1000).toFixed(1), view: S.view, input_hash: r.info.meta.input_hash, config_hash: r.info.meta.config_hash, seed: r.info.meta.seed, git: r.info.meta.git_commit, datasets: `${M.sexes.male.dataset},${M.sexes.female.dataset}` }
    : { mode: "live", source: S.source, input: S.liveLabel || S.source, t_ms: +(S.time * 1000).toFixed(1), view: S.view, config_hash: M.config_hash, reduced: S.reduced, datasets: `${M.sexes.male.dataset},${M.sexes.female.dataset}` };
}
function exportRun(fmt: "json" | "csv") {
  if (S.source !== "library" || !S.runs.size) { toast("Run export works on precomputed runs (Library input)."); return; }
  for (const [sex, r] of S.runs) {
    const b = brains.get(sex)!;
    if (fmt === "csv") download(`${S.runId}_${sex}_spikes.csv`, runToCsv(r, b, M.stages), "text/csv");
    else download(`${S.runId}_${sex}.json`, runToJson(r, b, M.stages), "application/json");
  }
}

// ---------- utils ----------
function onsetOf(a: Float32Array): number {
  let peak = 0; for (const v of a) peak = Math.max(peak, Math.abs(v));
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > 0.1 * peak) return i / FS;
  return 0;
}
function lowerBound(a: Float32Array, x: number) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; } return lo; }
function toast(msg: string) { const t = $("#toast"); t.textContent = msg; t.hidden = false; setTimeout(() => (t.hidden = true), 4000); }

boot().catch((e) => { console.error(e); const b = document.querySelector("#boot"); if (b) b.textContent = `Failed to load: ${e}`; });
