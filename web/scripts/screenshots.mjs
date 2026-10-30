// Capture documentation screenshots of the built app (dist/) into docs/images/. Usage: npm run build && npm run shots
import { createServer } from "node:http";
import { readFile, stat, mkdir, copyFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const root = fileURLToPath(new URL("../dist/", import.meta.url));
const out = fileURLToPath(new URL("../../docs/images/", import.meta.url));
await mkdir(out, { recursive: true });
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png" };
const server = createServer(async (req, res) => {
  const u = decodeURIComponent(req.url.split("?")[0]);
  const p = normalize(join(root, u === "/" ? "index.html" : u));
  try { await stat(p); res.writeHead(200, { "content-type": types[extname(p)] ?? "application/octet-stream" }); res.end(await readFile(p)); }
  catch { res.writeHead(404); res.end(); }
}).listen(4175);

const exe = [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean).find((c) => existsSync(c));
const gpu = process.env.SHOTS_SWIFTSHADER ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : ["--use-angle=d3d11", "--ignore-gpu-blocklist", "--enable-gpu"];
const browser = await puppeteer.launch({ executablePath: exe, headless: true, args: [...gpu, "--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.error("pageerror:", String(e)));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const lab = (fn, ...args) => page.evaluate(fn, ...args);
const shot = async (name, sel, opts = {}) => {
  const path = join(out, `${name}.png`);
  if (sel) await (await page.$(sel)).screenshot({ path }); else await page.screenshot({ path, ...opts });
  console.log("  " + name);
};
const tab = async (t) => { await page.click(`#tabs button[data-v=${t}]`); await wait(700); };
// Replay a run at 50x slow-mo from time t so spikes are mid-flight when the frame is grabbed.
const replayAt = async (run, t, settle = 1600) => {
  await lab(async (r) => { await window.flyLab.loadRun(r); }, run);
  await lab((tt) => { window.flyLab.setSpeed(50); window.flyLab.seek(tt); }, t);
  await wait(settle);
};

await page.goto("http://localhost:4175/", { waitUntil: "networkidle0", timeout: 90000 });
await page.waitForFunction(() => window.flyLab && !document.querySelector("#boot"), { timeout: 90000 });
await wait(1500);

// Pick neurons for the inspector/probes: strongest song-detector target, a JO afferent, an AMMC neuron (male).
const picks = await lab(() => {
  const b = window.flyLab.brains.get("male");
  const first = (pred) => { for (let i = 0; i < b.n; i++) if (pred(i)) return i; return 0; };
  const deg = (i) => b.indptr[i + 1] - b.indptr[i] + b.inPtr[i + 1] - b.inPtr[i];
  let song = -1, best = -1;
  for (let i = 0; i < b.n; i++) if (b.song[i] && deg(i) > best) { best = deg(i); song = i; }
  return { song, jo: first((i) => b.stage[i] === 0 && b.joSubtype[b.jo.indexOf(i)] === "B"), ammc: first((i) => b.stage[i] === 1 && b.hop[i] === 1) };
});

await replayAt("pulse35", 0.03);
await shot("01-explore-overview");
await lab(() => window.flyLab.preset("auditory")); await replayAt("pulse35", 0.05);
await shot("02-brains-auditory-closeup", ".center");
await lab(() => { window.flyLab.preset("front"); window.flyLab.setView("overlay"); }); await replayAt("pulse35", 0.05);
await shot("03-view-overlay", ".center");
await lab(() => window.flyLab.setView("diff")); await wait(1200);
await shot("04-view-difference", ".center");
await lab(() => window.flyLab.setView("side"));
await lab((p) => { window.flyLab.select("male", p.song); window.flyLab.pin("male", p.song); window.flyLab.pin("male", p.ammc); window.flyLab.pin("male", p.jo); }, picks);
await page.setViewport({ width: 1600, height: 1350, deviceScaleFactor: 1 }); // room for the full inspector
await replayAt("pulse35", 0.1, 2200);
await shot("05-inspector", ".right");
await shot("06-timeline", ".bottom");
await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });
await lab(() => window.flyLab.stage(3)); await replayAt("pulse35", 0.08);
await shot("07-pathway-stage-isolated");
await lab(() => window.flyLab.stage(3));
await page.click("#clean"); await replayAt("pulse35", 0.05);
await shot("08-clean-mode");
await page.click("#clean");
await page.click("#source button[data-v=synth]"); await lab(() => window.flyLab.setSpeed(10)); await page.click("#synth-go"); await wait(8000);
await shot("09-live-synth");
await page.click("#source button[data-v=library]"); await wait(800);
await tab("experiment"); await wait(500);
await shot("10-experiment-heatmaps", null, { fullPage: true });
await tab("validate");
await shot("11-validate", null, { fullPage: true });
await tab("notebook");
await page.type("#nb-text", "Female song-detector targets stay silent for every stimulus; activity stops at the wedge. Check whether the 8,000-neuron cap removes their inputs.");
await page.click("label.tag.t2"); await page.click("#nb-save"); await wait(400);
await shot("12-notebook");

await browser.close(); server.close();
const heat = fileURLToPath(new URL("../../data/results/heatmaps.png", import.meta.url));
if (existsSync(heat)) { await copyFile(heat, join(out, "13-pipeline-heatmaps.png")); console.log("  13-pipeline-heatmaps"); }
console.log(`screenshots written to ${out}`);
