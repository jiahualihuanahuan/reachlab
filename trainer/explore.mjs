import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Curiosity } from "../src/game/field.ts";
import { Instinct } from "../src/game/instinct.ts";
import { bootRom, buttons, wake } from "./emu.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const ROM = process.env.ROM_PATH || join(root, "attachments/Pokemon - Yellow Version (UE) [C][!].gbc");
const OUT = join(root, "trainer/checkpoints");
const LIMIT = Number(process.env.DECISIONS || 12000);
mkdirSync(OUT, { recursive: true });

const cart = await bootRom(ROM);
const woke = await wake(cart);
const curiosity = new Curiosity();
const obs = new Float32Array(32);
const demo = [];
const seen = new Set();
const maps = new Set();

function tile() {
  return `${cart.read(0xd35d)}:${cart.read(0xd361)}:${cart.read(0xd360)}`;
}
function hold(action) {
  cart.frames(20, buttons(action, true));
  cart.frames(6);
}
function note(line) {
  console.log(`${new Date().toISOString()} ${line}`);
}

note(`awake ${woke} facing ${cart.read(0xc109)} party ${cart.read(0xd162)}`);

let last = tile();
let idle = 0;
for (let step = 0; step < LIMIT; step++) {
  curiosity.observe(cart, obs, -1);
  const locked = curiosity.locked(cart);
  let action = 4;
  if (!locked) {
    const state = cart.capture();
    let best = null;
    for (const a of [0, 1, 2, 3]) {
      cart.restore(state);
      hold(a);
      const key = tile();
      const moved = key !== tile.call ? key : key;
      const novel = !seen.has(key) && key !== last;
      const score = (novel ? 10 : 0) + (key !== last ? 1 : 0) - (maps.has(cart.read(0xd35d)) ? 0 : 0);
      if (!best || score > best.score) best = { a, score, key };
    }
    cart.restore(state);
    if (!best || best.score === 0) {
      idle++;
      action = idle % 5 === 0 ? 4 : [0, 1, 2, 3][idle % 4];
    } else {
      idle = 0;
      action = best.a;
    }
  }
  const snapshot = new Float32Array(obs);
  hold(action);
  const result = curiosity.reward(cart);
  const key = tile();
  const map = cart.read(0xd35d);
  seen.add(key);
  if (!maps.has(map)) {
    maps.add(map);
    note(`map ${map} ${result.event || ""} tiles ${seen.size} party ${cart.read(0xd162)} badges ${cart.read(0xd355)}`);
  } else if (result.event && !result.event.startsWith("New ground") && result.event !== "Waking up") {
    note(result.event);
  }
  if (step < 12 || step % 500 === 0) note(`step ${step} a ${action} tile ${key} idle ${idle} locked ${locked}`);
  last = key;
  demo.push({ obs: Array.from(snapshot), action, reward: result.total });
  if (cart.read(0xd162) > 0 && maps.has(39) && step > 400) {
    note(`partner acquired at step ${step}`);
    break;
  }
}

const instinct = new Instinct();
const actions = demo.map((d) => d.action);
for (let epoch = 0; epoch < 8; epoch++) {
  const obsBatch = [];
  const act = [];
  const logp = [];
  const values = [];
  const rewards = [];
  const dones = [];
  for (let i = 0; i < demo.length; i++) {
    const row = new Float32Array(demo[i].obs);
    const sample = instinct.sample(row, false);
    obsBatch.push(row);
    act.push(demo[i].action);
    logp.push(Math.log(1 / 8));
    values.push(sample.value);
    rewards.push(1);
    dones.push(false);
    if (obsBatch.length === 32) {
      instinct.update(obsBatch, act, logp, values, rewards, dones, 0);
      obsBatch.length = 0;
      act.length = 0;
      logp.length = 0;
      values.length = 0;
      rewards.length = 0;
      dones.length = 0;
    }
  }
}
const weights = Array.from(instinct.export());
writeFileSync(
  join(OUT, "instinct.json"),
  JSON.stringify({
    weights,
    updates: instinct.updates,
    decisions: demo.length,
    maps: [...maps],
    tiles: seen.size,
    dex: curiosity.owned,
    badges: curiosity.badges,
    party: cart.read(0xd162),
  }),
);
note(`saved ${demo.length} steps, ${maps.size} maps, tiles ${[...seen].sort().join(" ")}`);
void actions;
