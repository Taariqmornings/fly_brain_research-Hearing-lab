// Lab notebook: tagged observations with run metadata, persisted in localStorage; JSON/CSV export.
import type { Brain, Run } from "./data";

export type Tag = "VALIDATED" | "MODEL PREDICTION" | "ARTIFACT RISK";
export interface Entry { id: string; created: string; tag: Tag; text: string; context: Record<string, any> }
const KEY = "fly-hearing-lab.notebook.v1";

export function loadEntries(): Entry[] {
  try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; }
}
export function saveEntries(e: Entry[]): boolean {
  try { localStorage.setItem(KEY, JSON.stringify(e)); return true; } catch { return false; }
}

export function download(name: string, data: string, mime: string) {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function runToCsv(run: Run, b: Brain, stages: string[]): string {
  const lines = ["neuron_id,cell_type,stage,t_s"];
  for (let k = 0; k < run.ids.length; k++) {
    const i = run.ids[k];
    lines.push(`${b.ids[i]},${(b.types[b.typeIdx[i]] || "").replace(/,/g, ";")},${stages[b.stage[i]]},${run.t[k].toFixed(5)}`);
  }
  return lines.join("\n");
}

export function runToJson(run: Run, b: Brain, stages: string[]): string {
  const rates: Record<string, number> = {};
  for (let i = 0; i < b.n; i++) if (run.rates[i] > 0) rates[b.ids[i]] = +run.rates[i].toFixed(3);
  return JSON.stringify({
    run: run.info.id, sex: run.sex, meta: run.info.meta, duration_s: run.duration, n_spikes: run.ids.length, stages,
    rates_hz_by_neuron: rates, note: "Simulation outputs are model predictions.",
  }, null, 1);
}

export class NotebookView {
  entries = loadEntries();
  constructor(private el: HTMLElement, private context: () => Record<string, any>, private onExportRun: (fmt: "json" | "csv") => void) { this.render(); }

  render() {
    const e = this.entries;
    this.el.innerHTML = `
      <div class="nb">
        <section class="nb-new card">
          <h3>New observation</h3>
          <textarea id="nb-text" rows="4" placeholder="What did you see? e.g. 'Female vpoEN silent at all IPIs; male pC2 peaks at 15 ms'"></textarea>
          <div class="nb-tags" role="radiogroup">
            ${(["MODEL PREDICTION", "VALIDATED", "ARTIFACT RISK"] as Tag[]).map((t, i) => `<label class="tag t${i}"><input type="radio" name="nb-tag" value="${t}" ${i === 0 ? "checked" : ""}>${t}</label>`).join("")}
          </div>
          <p class="muted small">VALIDATED only when the claim matches published recordings; anything else is a model prediction. Current run metadata is attached automatically.</p>
          <div class="actions"><button id="nb-save" class="primary">Save observation</button>
            <button id="nb-json">Export notebook JSON</button><button id="nb-run-json">Run → JSON</button><button id="nb-run-csv">Run → CSV</button></div>
        </section>
        <section class="nb-list">
          ${e.length ? e.slice().reverse().map((x) => `
            <article class="card entry"><header><span class="tagpill ${x.tag.replace(/ /g, "-").toLowerCase()}">${x.tag}</span>
              <time>${new Date(x.created).toLocaleString()}</time><button class="link del" data-id="${x.id}">delete</button></header>
              <p>${x.text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!)}</p>
              <code>${Object.entries(x.context).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ")}</code></article>`).join("")
            : `<div class="card muted">No observations yet. They are stored in this browser only — export JSON to keep them.</div>`}
        </section>
      </div>`;
    this.el.querySelector("#nb-save")!.addEventListener("click", () => {
      const text = (this.el.querySelector("#nb-text") as HTMLTextAreaElement).value.trim(); if (!text) return;
      const tag = (this.el.querySelector("input[name=nb-tag]:checked") as HTMLInputElement).value as Tag;
      this.entries.push({ id: crypto.randomUUID?.() ?? String(Date.now()), created: new Date().toISOString(), tag, text, context: this.context() });
      if (!saveEntries(this.entries)) alert("Could not persist to localStorage (private mode?). Export JSON to keep it.");
      this.render();
    });
    this.el.querySelector("#nb-json")!.addEventListener("click", () => download("fly-hearing-notebook.json", JSON.stringify(this.entries, null, 1), "application/json"));
    this.el.querySelector("#nb-run-json")!.addEventListener("click", () => this.onExportRun("json"));
    this.el.querySelector("#nb-run-csv")!.addEventListener("click", () => this.onExportRun("csv"));
    this.el.querySelectorAll<HTMLButtonElement>(".del").forEach((b) => b.addEventListener("click", () => {
      this.entries = this.entries.filter((x) => x.id !== b.dataset.id); saveEntries(this.entries); this.render();
    }));
  }
}
