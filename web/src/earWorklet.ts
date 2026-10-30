// AudioWorklet (context at 10 kHz): mic -> ear filter bank -> envelopes posted to the main thread in ~13 ms blocks.
import { EarBank } from "./ear";

declare const sampleRate: number;
declare class AudioWorkletProcessor { port: MessagePort; constructor(o?: any); }
declare function registerProcessor(name: string, c: any): void;

class EarProcessor extends AudioWorkletProcessor {
  private bank: EarBank | null = null;
  private gain = 10;
  constructor() {
    super();
    this.port.onmessage = (e) => {
      if (e.data.type === "spec") this.bank = new EarBank(e.data.spec);
      if (e.data.type === "gain") this.gain = e.data.gain;
    };
  }
  process(inputs: Float32Array[][]): boolean {
    const ch = inputs[0]?.[0];
    if (!ch || !this.bank) return true;
    const x = new Float32Array(ch.length);
    let s = 0;
    for (let i = 0; i < ch.length; i++) { x[i] = ch[i] * this.gain; s += x[i] * x[i]; }
    const env = this.bank.process(x);
    this.port.postMessage({ type: "env", env, raw: x, level: Math.sqrt(s / ch.length), fs: sampleRate }, env.map((e) => e.buffer));
    return true;
  }
}
registerProcessor("ear-processor", EarProcessor);
