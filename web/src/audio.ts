// Audio I/O at the ear rate (10 kHz): mic via AudioWorklet, file decoding, synthesis and slowed playback.
import workletUrl from "./earWorklet.ts?worker&url";
import { phaseScramble, pulseSong, rms, sineSong, whiteNoise } from "./ear";

export const FS = 10000;
let ctx: AudioContext | null = null;
function context(): AudioContext {
  if (!ctx || ctx.state === "closed") ctx = new AudioContext({ sampleRate: FS });
  return ctx;
}

export interface MicHandle { stop(): void; setGain(g: number): void }
export async function startMic(spec: any, onEnv: (env: Float32Array[], raw: Float32Array, level: number) => void): Promise<MicHandle> {
  const c = context();
  await c.resume();
  await c.audioWorklet.addModule(workletUrl);
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  const src = c.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(c, "ear-processor");
  node.port.postMessage({ type: "spec", spec });
  node.port.onmessage = (e) => onEnv(e.data.env, e.data.raw, e.data.level);
  const sink = c.createGain(); sink.gain.value = 0;
  src.connect(node).connect(sink).connect(c.destination);
  return {
    stop() { node.disconnect(); src.disconnect(); stream.getTracks().forEach((t) => t.stop()); },
    setGain(g: number) { node.port.postMessage({ type: "gain", gain: g }); },
  };
}

export async function decodeFile(file: File): Promise<Float32Array> {
  const buf = await context().decodeAudioData(await file.arrayBuffer()); // resampled to FS by the context
  const x = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < x.length; i++) x[i] += d[i] / buf.numberOfChannels; }
  return x;
}

export function normalise(x: Float32Array): Float32Array {
  const r = rms(x);
  return r > 0 ? x.map((v) => v / r) : x;
}

export type StimKind = "pulse" | "sine" | "noise" | "scrambled";
export function synth(kind: StimKind, dur: number, carrier: number, ipi: number): Float32Array {
  if (kind === "pulse") return pulseSong(FS, dur, carrier, ipi);
  if (kind === "sine") return sineSong(FS, dur, carrier);
  if (kind === "noise") return whiteNoise(FS, dur);
  return phaseScramble(pulseSong(FS, dur, carrier, ipi));
}

/** Plays x slowed by `speed` (pitch drops by the same factor, as a tape would). */
export function play(x: Float32Array, speed: number, offsetS = 0): { stop(): void } {
  const c = context();
  void c.resume();
  const peak = x.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-9);
  const b = c.createBuffer(1, x.length, FS);
  b.getChannelData(0).set(x.map((v) => (0.4 * v) / peak));
  const s = c.createBufferSource();
  s.buffer = b; s.playbackRate.value = 1 / speed;
  s.connect(c.destination);
  s.start(0, Math.min(offsetS, b.duration));
  return { stop() { try { s.stop(); } catch { /* already stopped */ } } };
}
