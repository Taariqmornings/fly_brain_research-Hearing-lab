// Inspector panel: identity, connectivity, rate trace, IPI tuning curve, live perturbations, probe pinning.
import type { Brain, Sex } from "./data";

export interface InspectorHooks {
  select(sex: Sex, idx: number): void;
  pin(sex: Sex, idx: number): void;
  perturb(kind: "silence" | "boost", sex: Sex, idx: number, on: boolean): void;
  rateTrace(sex: Sex, idx: number): Float32Array | null;
  tuning(sex: Sex, idx: number): Promise<{ ipis: number[]; carriers: number[]; rates: number[][] } | null>;
  isLive(): boolean;
  stageLabel(s: number): string;
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

export class Inspector {
  private perturbed = new Map<string, Set<string>>();
  sel: { sex: Sex; idx: number } | null = null;
  constructor(private el: HTMLElement, private brains: Map<Sex, Brain>, private hooks: InspectorHooks) { this.empty(); }

  empty() {
    this.sel = null;
    this.el.innerHTML = `<div class="insp-empty"><div class="k">NO SELECTION</div><p>Click a neuron in the 3D view, a stage on the left, or a partner in this panel.</p>
      <ul class="hint"><li><b>Drag</b> orbit · <b>Wheel</b> zoom · <b>Right-drag</b> pan</li><li><b>Space</b> play/pause · <b>1/2/3</b> views · <b>C</b> clean mode</li></ul></div>`;
  }

  show(sex: Sex, idx: number) {
    const b = this.brains.get(sex)!;
    this.sel = { sex, idx };
    const type = b.types[b.typeIdx[idx]] || "(untyped)";
    const key = `${sex}:${idx}`, pset = this.perturbed.get(key) ?? new Set();
    const other = this.brains.get(sex === "male" ? "female" : "male");
    const otherCount = other ? countType(other, type) : 0;
    const partners = (ptr: Int32Array, ind: Int32Array, syn: Int32Array) => {
      const rows: [number, number][] = [];
      for (let k = ptr[idx]; k < ptr[idx + 1]; k++) rows.push([ind[k], syn[k]]);
      return rows.sort((a, c) => Math.abs(c[1]) - Math.abs(a[1])).slice(0, 6);
    };
    const outs = partners(b.indptr, b.indices, b.syn.map((s, k) => s * Math.sign(b.weights[k])) as unknown as Int32Array);
    const ins = partners(b.inPtr, b.inIdx, b.inSyn);
    const li = (rows: [number, number][]) => rows.length ? rows.map(([j, s]) =>
      `<li data-j="${j}"><span class="t">${esc(b.types[b.typeIdx[j]] || "untyped")}</span><span class="s ${s < 0 ? "neg" : "pos"}">${s > 0 ? "+" : ""}${s}</span></li>`).join("") : `<li class="none">none in subnetwork</li>`;
    const live = this.hooks.isLive();
    this.el.innerHTML = `
      <div class="insp-head"><span class="dot ${sex}"></span><div><div class="type">${esc(type)}</div>
        <div class="sub">${sex.toUpperCase()} · ${esc(b.dataset)} · id <span class="mono sel">${b.ids[idx]}</span></div></div></div>
      <dl class="kv">
        <dt>Stage</dt><dd>${esc(this.hooks.stageLabel(b.stage[idx]))}${b.hop[idx] >= 0 ? ` · hop ${b.hop[idx]}` : " · added target"}</dd>
        <dt>Transmitter</dt><dd>${esc(b.nt[idx] || "unknown")} <span class="muted">(conf ${b.ntConf[idx].toFixed(2)}, sign ${b.sign[idx] > 0 ? "+" : b.sign[idx] < 0 ? "−" : "0"})</span></dd>
        <dt>Role</dt><dd>${b.song[idx] ? "song-detector target" : b.recep[idx] ? "receptivity target" : b.stage[idx] === 0 ? `JO-${esc(b.joSubtype[b.jo.indexOf(idx)] ?? "?")} input` : "relay"}</dd>
        <dt>Position</dt><dd>${b.posMeasured[idx] ? "soma / skeleton (measured)" : "<span class='warn'>layout fallback</span>"}</dd>
        <dt>Other sex</dt><dd>${otherCount ? `${otherCount} × same type <button class="link" data-act="xsex">show</button>` : "<span class='warn'>type absent from its subnetwork</span>"}</dd>
      </dl>
      <h4>Firing rate</h4><canvas class="spark" height="46"></canvas>
      <h4>IPI tuning <span class="muted">(sweep, seed 0)</span></h4><canvas class="tune" height="80"></canvas>
      <div class="row2"><div><h4>Top inputs</h4><ul class="plist">${li(ins)}</ul></div><div><h4>Top outputs</h4><ul class="plist">${li(outs)}</ul></div></div>
      <div class="actions">
        <button data-act="silence" class="${pset.has("silence") ? "on" : ""}" ${live ? "" : "disabled"} title="Zero this neuron's outgoing synapses (live mode)">Silence</button>
        <button data-act="boost" class="${pset.has("boost") ? "on" : ""}" ${live ? "" : "disabled"} title="Add 150 Hz Poisson drive (live mode)">Boost</button>
        <button data-act="pin">Pin as probe</button>
      </div>
      ${live ? "" : `<p class="muted small">Silence/Boost act on the live simulation — switch input to Mic, File or Synth.</p>`}`;
    this.el.querySelectorAll<HTMLLIElement>("li[data-j]").forEach((l) => l.addEventListener("click", () => this.hooks.select(sex, +l.dataset.j!)));
    this.el.querySelectorAll<HTMLButtonElement>("button[data-act]").forEach((btn) => btn.addEventListener("click", () => {
      const a = btn.dataset.act!;
      if (a === "pin") this.hooks.pin(sex, idx);
      else if (a === "xsex" && other) { const j = firstOfType(other, type); if (j >= 0) this.hooks.select(other.sex, j); }
      else if (a === "silence" || a === "boost") {
        const on = !pset.has(a); on ? pset.add(a) : pset.delete(a); this.perturbed.set(key, pset);
        btn.classList.toggle("on", on); this.hooks.perturb(a, sex, idx, on);
      }
    }));
    this.refreshTrace();
    void this.hooks.tuning(sex, idx).then((t) => this.drawTuning(t));
  }

  refreshTrace() {
    if (!this.sel) return;
    const c = this.el.querySelector<HTMLCanvasElement>("canvas.spark"); if (!c) return;
    const tr = this.hooks.rateTrace(this.sel.sex, this.sel.idx);
    const g = fit(c); const w = c.clientWidth, h = c.clientHeight;
    g.clearRect(0, 0, w, h);
    if (!tr) { g.fillStyle = "#56606c"; g.fillText("no activity data", 4, 14); return; }
    let max = 1, sum = 0; for (const v of tr) { max = Math.max(max, v); sum += v; }
    g.strokeStyle = this.sel.sex === "male" ? "#40d9ff" : "#ffb833"; g.beginPath();
    tr.forEach((v, k) => { const x = (k / (tr.length - 1)) * w, y = h - 4 - (v / max) * (h - 14); k ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke(); g.fillStyle = "#9aa4b1"; g.font = "10px 'JetBrains Mono', monospace";
    g.fillText(`mean ${(sum / tr.length).toFixed(1)} Hz · peak ${max.toFixed(0)} Hz`, 4, 10);
  }

  private drawTuning(t: { ipis: number[]; carriers: number[]; rates: number[][] } | null) {
    const c = this.el.querySelector<HTMLCanvasElement>("canvas.tune"); if (!c) return;
    const g = fit(c), w = c.clientWidth, h = c.clientHeight;
    g.font = "10px 'JetBrains Mono', monospace"; g.fillStyle = "#56606c";
    if (!t) { g.fillText("sweep runs not available", 4, 14); return; }
    let max = 0.5; for (const r of t.rates) for (const v of r) max = Math.max(max, v);
    const cols = ["#5b8def", "#40d9ff", "#9ac48f", "#ffb833", "#e58f8f", "#c48fe5", "#ddd", "#888", "#f5f"];
    t.carriers.forEach((f, ci) => {
      g.strokeStyle = cols[ci % cols.length]; g.beginPath();
      t.ipis.forEach((_, ii) => { const x = 24 + (ii / Math.max(1, t.ipis.length - 1)) * (w - 30), y = h - 14 - (t.rates[ii][ci] / max) * (h - 24); ii ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke();
      g.fillStyle = cols[ci % cols.length]; g.fillText(`${f}`, w - 26, 10 + ci * 10);
    });
    g.fillStyle = "#56606c"; g.fillText(`${t.ipis[0]}`, 20, h - 2); g.fillText(`${t.ipis[t.ipis.length - 1]} ms IPI`, w - 70, h - 2);
    g.fillText(`${max.toFixed(1)} Hz`, 0, 10);
  }
}

function fit(c: HTMLCanvasElement): CanvasRenderingContext2D {
  const dpr = Math.min(devicePixelRatio, 2); c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
  const g = c.getContext("2d")!; g.setTransform(dpr, 0, 0, dpr, 0, 0); return g;
}
function countType(b: Brain, type: string): number { const t = b.types.indexOf(type); if (t < 0 || !type) return 0; let n = 0; for (const v of b.typeIdx) if (v === t) n++; return n; }
function firstOfType(b: Brain, type: string): number { const t = b.types.indexOf(type); return t < 0 ? -1 : b.typeIdx.indexOf(t); }
