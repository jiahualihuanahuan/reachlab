import { OBS } from "./field";

const ACT = 8;
const H = 48;

function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class Instinct {
  w1 = new Float32Array(H * OBS);
  b1 = new Float32Array(H);
  w2 = new Float32Array(H * H);
  b2 = new Float32Array(H);
  wA = new Float32Array(ACT * H);
  bA = new Float32Array(ACT);
  wV = new Float32Array(H);
  bV = 0;
  m = new Map<Float32Array | "bv", Float32Array>();
  v = new Map<Float32Array | "bv", Float32Array>();
  mvB = 0;
  vvB = 0;
  step = 0;
  entropy = 1;
  updates = 0;

  constructor() {
    const fill = (w: Float32Array, fan: number) => {
      const s = Math.sqrt(1 / fan);
      for (let i = 0; i < w.length; i++) w[i] = randn() * s;
    };
    fill(this.w1, OBS);
    fill(this.w2, H);
    fill(this.wA, H);
    fill(this.wV, H);
    for (const w of [this.w1, this.b1, this.w2, this.b2, this.wA, this.bA, this.wV]) {
      this.m.set(w, new Float32Array(w.length));
      this.v.set(w, new Float32Array(w.length));
    }
  }

  private tanh(x: number) {
    const e = Math.exp(2 * Math.max(-10, Math.min(10, x)));
    return (e - 1) / (e + 1);
  }

  forward(obs: Float32Array) {
    const h1 = new Float32Array(H);
    const h2 = new Float32Array(H);
    for (let o = 0; o < H; o++) {
      let s = this.b1[o];
      const row = o * OBS;
      for (let i = 0; i < OBS; i++) s += this.w1[row + i] * obs[i];
      h1[o] = this.tanh(s);
    }
    for (let o = 0; o < H; o++) {
      let s = this.b2[o];
      const row = o * H;
      for (let i = 0; i < H; i++) s += this.w2[row + i] * h1[i];
      h2[o] = this.tanh(s);
    }
    const logits = new Float32Array(ACT);
    let max = -1e9;
    for (let a = 0; a < ACT; a++) {
      let s = this.bA[a];
      const row = a * H;
      for (let i = 0; i < H; i++) s += this.wA[row + i] * h2[i];
      logits[a] = s;
      if (s > max) max = s;
    }
    let sum = 0;
    const p = new Float32Array(ACT);
    for (let a = 0; a < ACT; a++) {
      p[a] = Math.exp(logits[a] - max);
      sum += p[a];
    }
    let ent = 0;
    for (let a = 0; a < ACT; a++) {
      p[a] /= sum;
      if (p[a] > 1e-8) ent -= p[a] * Math.log(p[a]);
    }
    let value = this.bV;
    for (let i = 0; i < H; i++) value += this.wV[i] * h2[i];
    return { h1, h2, p, value, ent };
  }

  sample(obs: Float32Array, greedy = false) {
    const out = this.forward(obs);
    this.entropy = out.ent;
    if (greedy) {
      let best = 0;
      for (let a = 1; a < ACT; a++) if (out.p[a] > out.p[best]) best = a;
      return { action: best, logp: Math.log(out.p[best] + 1e-8), value: out.value };
    }
    let r = Math.random();
    let action = ACT - 1;
    for (let a = 0; a < ACT; a++) {
      r -= out.p[a];
      if (r <= 0) {
        action = a;
        break;
      }
    }
    return { action, logp: Math.log(out.p[action] + 1e-8), value: out.value };
  }

  update(
    obs: Float32Array[],
    actions: number[],
    oldLogp: number[],
    values: number[],
    rewards: number[],
    dones: boolean[],
    bootstrap: number,
  ) {
    const n = obs.length;
    if (n < 4) return;
    const adv = new Float32Array(n);
    const ret = new Float32Array(n);
    let gae = 0;
    for (let t = n - 1; t >= 0; t--) {
      const next = t === n - 1 ? bootstrap : values[t + 1];
      const nt = dones[t] ? 0 : 1;
      const delta = rewards[t] + 0.995 * next * nt - values[t];
      gae = delta + 0.995 * 0.95 * nt * gae;
      adv[t] = gae;
      ret[t] = gae + values[t];
    }
    let mean = 0;
    for (let i = 0; i < n; i++) mean += adv[i];
    mean /= n;
    let vari = 0;
    for (let i = 0; i < n; i++) vari += (adv[i] - mean) ** 2;
    const sd = Math.sqrt(vari / n) + 1e-6;
    for (let i = 0; i < n; i++) adv[i] = (adv[i] - mean) / sd;

    const lr = 0.0008;
    const gW1 = new Float32Array(this.w1.length);
    const gB1 = new Float32Array(H);
    const gW2 = new Float32Array(this.w2.length);
    const gB2 = new Float32Array(H);
    const gWA = new Float32Array(this.wA.length);
    const gBA = new Float32Array(ACT);
    const gWV = new Float32Array(H);
    let gBV = 0;

    for (let s = 0; s < n; s++) {
      const fwd = this.forward(obs[s]);
      const ratio = Math.exp(Math.log(fwd.p[actions[s]] + 1e-8) - oldLogp[s]);
      const clipped = Math.min(Math.max(ratio, 0.8), 1.2);
      const s1 = ratio * adv[s];
      const s2 = clipped * adv[s];
      const gLog = s1 <= s2 ? -adv[s] * ratio : 0;
      const dLogit = new Float32Array(ACT);
      for (let a = 0; a < ACT; a++) dLogit[a] = gLog * ((a === actions[s] ? 1 : 0) - fwd.p[a]);
      const hEnt = -fwd.p.reduce((acc, pi) => (pi > 1e-8 ? acc + pi * Math.log(pi) : acc), 0);
      for (let a = 0; a < ACT; a++) {
        dLogit[a] += 0.02 * fwd.p[a] * (Math.log(fwd.p[a] + 1e-8) + hEnt);
      }
      const dV = 0.5 * (fwd.value - ret[s]);
      const dH2 = new Float32Array(H);
      for (let a = 0; a < ACT; a++) {
        gBA[a] += dLogit[a];
        const row = a * H;
        for (let i = 0; i < H; i++) {
          gWA[row + i] += dLogit[a] * fwd.h2[i];
          dH2[i] += dLogit[a] * this.wA[row + i];
        }
      }
      gBV += dV;
      for (let i = 0; i < H; i++) {
        gWV[i] += dV * fwd.h2[i];
        dH2[i] += dV * this.wV[i];
        dH2[i] *= 1 - fwd.h2[i] * fwd.h2[i];
      }
      const dH1 = new Float32Array(H);
      for (let o = 0; o < H; o++) {
        gB2[o] += dH2[o];
        const row = o * H;
        for (let i = 0; i < H; i++) {
          gW2[row + i] += dH2[o] * fwd.h1[i];
          dH1[i] += dH2[o] * this.w2[row + i];
        }
      }
      for (let i = 0; i < H; i++) dH1[i] *= 1 - fwd.h1[i] * fwd.h1[i];
      for (let o = 0; o < H; o++) {
        gB1[o] += dH1[o];
        const row = o * OBS;
        for (let i = 0; i < OBS; i++) gW1[row + i] += dH1[o] * obs[s][i];
      }
    }
    const scale = 1 / n;
    const groups: [Float32Array, Float32Array][] = [
      [this.w1, gW1],
      [this.b1, gB1],
      [this.w2, gW2],
      [this.b2, gB2],
      [this.wA, gWA],
      [this.bA, gBA],
      [this.wV, gWV],
    ];
    let norm = gBV * gBV;
    for (const [, g] of groups) for (let i = 0; i < g.length; i++) norm += g[i] * g[i];
    const clip = norm > 0.5 * 0.5 * n * n ? (0.5 * n) / Math.sqrt(norm) : 1;
    this.step++;
    const b1c = 1 - 0.9 ** this.step;
    const b2c = 1 - 0.999 ** this.step;
    const adam = (w: Float32Array, g: Float32Array) => {
      const mm = this.m.get(w)!;
      const vv = this.v.get(w)!;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * scale * clip;
        mm[i] = 0.9 * mm[i] + 0.1 * gi;
        vv[i] = 0.999 * vv[i] + 0.001 * gi * gi;
        w[i] -= (lr * (mm[i] / b1c)) / (Math.sqrt(vv[i] / b2c) + 1e-5);
      }
    };
    for (const [w, g] of groups) adam(w, g);
    const gi = gBV * scale * clip;
    this.mvB = 0.9 * this.mvB + 0.1 * gi;
    this.vvB = 0.999 * this.vvB + 0.001 * gi * gi;
    this.bV -= (lr * (this.mvB / b1c)) / (Math.sqrt(this.vvB / b2c) + 1e-5);
    this.updates++;
  }

  export() {
    const parts = [this.w1, this.b1, this.w2, this.b2, this.wA, this.bA, this.wV];
    const size = parts.reduce((s, p) => s + p.length, 0) + 1;
    const all = new Float32Array(size);
    let o = 0;
    for (const p of parts) {
      all.set(p, o);
      o += p.length;
    }
    all[o] = this.bV;
    return all;
  }

  import(all: Float32Array) {
    const parts = [this.w1, this.b1, this.w2, this.b2, this.wA, this.bA, this.wV];
    let o = 0;
    for (const p of parts) {
      p.set(all.subarray(o, o + p.length));
      o += p.length;
    }
    this.bV = all[o] ?? 0;
  }
}

export const ACTION_NAMES = ["north", "east", "south", "west", "a", "b", "start", "select"] as const;
