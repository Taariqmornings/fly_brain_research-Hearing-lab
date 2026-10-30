// Headless smoke check: serve dist/, load the page in system Chrome/Edge, fail on console errors.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import puppeteer from "puppeteer-core";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../dist/", import.meta.url));
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png" };
const server = createServer(async (req, res) => {
  const p = normalize(join(root, decodeURIComponent(req.url.split("?")[0]) === "/" ? "index.html" : decodeURIComponent(req.url.split("?")[0])));
  try { await stat(p); res.writeHead(200, { "content-type": types[extname(p)] ?? "application/octet-stream" }); res.end(await readFile(p)); }
  catch { res.writeHead(404); res.end(); }
}).listen(4174);

const candidates = [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean);
const executablePath = candidates.find((c) => existsSync(c));
if (!executablePath) { console.error("no Chrome/Edge found; set CHROME_PATH"); process.exit(2); }
const browser = await puppeteer.launch({ executablePath, headless: true, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(String(e)));
page.on("response", (r) => { if (r.status() >= 400) errors.push(`HTTP ${r.status()} ${r.url()}`); });
page.on("requestfailed", (r) => errors.push(`request failed: ${r.url()}`));
await page.goto("http://localhost:4174/", { waitUntil: "networkidle0", timeout: 60000 });
await page.waitForFunction(() => !document.querySelector("#boot"), { timeout: 60000 }).catch(() => errors.push("app did not finish booting"));
await new Promise((r) => setTimeout(r, 2500));
for (const tab of ["experiment", "validate", "notebook", "explore"]) { await page.click(`#tabs button[data-v=${tab}]`); await new Promise((r) => setTimeout(r, 400)); }
await page.click("#view button[data-v=diff]"); await new Promise((r) => setTimeout(r, 400));
await page.click("#source button[data-v=synth]"); await page.click("#synth-go"); await new Promise((r) => setTimeout(r, 2500));
const hud = await page.$eval("#hud-tl", (e) => e.textContent);
await page.screenshot({ path: fileURLToPath(new URL("../../data/smoke.png", import.meta.url)) });
await browser.close(); server.close();
const real = errors.filter((e) => !/fonts\.(googleapis|gstatic)/.test(e));
console.log(`smoke: ${real.length ? "FAIL" : "PASS"} - ${real.length} console errors; HUD "${hud?.slice(0, 60)}"`);
if (real.length) { console.log(real.slice(0, 10).join("\n")); process.exit(1); }
