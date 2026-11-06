import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Curiosity } from "../src/game/field.ts";
import { Instinct } from "../src/game/instinct.ts";
import { bootRom, buttons, wake } from "./emu.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const ROM = process.env.ROM_PATH || join(root, "attachments/Pokemon - Yellow Version (UE) [C][!].gbc");
const OUT = join(root, "trainer/checkpoints");
const HOLD = 8;
const GAP = 2;
const ROLLOUT = 32;
const DECISIONS = Number(process.env.DECISIONS || 80000);

const MAPS = {
  0: "Pallet Town",
  37: "Red's house 1F",
  38: "Red's bedroom",
  39: "Oak's lab",
  12: "Route 1",
};

mkdirSync(OUT, { recursive: true });

const cart = await bootRom(ROM);
const woke = await wake(cart);
const instinct = new Instinct();
const curiosity = new Curiosity();
const obs = new Float32Array(32);
const roll = { obs: [], actions: [], logp: [], values: [], rewards: [], dones: [] };
let episode = 0;
let best = 0;
let decisions = 0;
let stuck = 0;
let lastTile = "";
const seenMaps = new Set();
const log = [];

function note(line) {
  const row = `${new Date().toISOString()} ${line}`;
  log.push(row);
  console.log(row);
}

function save() {
  const payload = {
    weights: Array.from(instinct.export()),
    updates: instinct.updates,
    best,
    decisions,
    maps: [...seenMaps],
    tiles: curiosity.tiles.size,
    dex: curiosity.owned,
    badges: curiosity.badges,
    party: cart.read(0xd162),
  };
  writeFileSync(join(OUT, "instinct.json"), JSON.stringify(payload));
  writeFileSync(join(root, "public/instinct.json"), JSON.stringify(payload));
}

note(`awake frame ${woke} at bedroom ${cart.read(0xd361)},${cart.read(0xd360)} facing ${cart.read(0xc109)}`);

let phase = "hold";
let left = HOLD;
let action = 2;
let forced = false;

function finishRollout() {
  curiosity.observe(cart, obs, action);
  const boot = instinct.sample(obs, true).value;
  instinct.update(roll.obs, roll.actions, roll.logp, roll.values, roll.rewards, roll.dones, boot);
  roll.obs = [];
  roll.actions = [];
  roll.logp = [];
  roll.values = [];
  roll.rewards = [];
  roll.dones = [];
}

while (decisions < DECISIONS) {
  if (phase === "hold") {
    cart.frames(1, forced ? buttons(4, true) : buttons(action, true));
    left--;
    if (left <= 0) {
      phase = "gap";
      left = GAP;
    }
    continue;
  }
  cart.frames(1);
  left--;
  if (left > 0) continue;

  const locked = curiosity.locked(cart);
  const result = curiosity.reward(cart);
  episode += result.total;
  if (roll.obs.length && !forced) roll.rewards[roll.rewards.length - 1] = result.total;
  if (result.event && result.event !== "Waking up" && !result.event.startsWith("New ground")) note(result.event);

  const map = cart.read(0xd35d);
  if (!seenMaps.has(map)) {
    seenMaps.add(map);
    note(`milestone map ${map} ${MAPS[map] || ""} tiles ${curiosity.tiles.size} return ${episode.toFixed(1)} party ${cart.read(0xd162)}`);
    save();
  }

  if (locked) {
    forced = true;
    action = 4;
    phase = "hold";
    left = 1;
    continue;
  }
  forced = false;
  curiosity.observe(cart, obs, action);
  const tile = `${map}:${cart.read(0xd361)}:${cart.read(0xd360)}`;
  stuck = tile === lastTile ? stuck + 1 : 0;
  lastTile = tile;
  if (stuck > 400) {
    cart.restore();
    stuck = 0;
    note(`stuck, back to the bedroom. maps ${[...seenMaps].join(",")} tiles ${curiosity.tiles.size} party ${cart.read(0xd162)}`);
  }

  const sample = instinct.sample(obs, false);
  action = sample.action;
  roll.obs.push(new Float32Array(obs));
  roll.actions.push(action);
  roll.logp.push(sample.logp);
  roll.values.push(sample.value);
  roll.rewards.push(0);
  roll.dones.push(false);
  if (roll.obs.length >= ROLLOUT) {
    finishRollout();
    if (episode > best) best = episode;
  }
  decisions++;
  if (decisions % 2000 === 0) {
    note(
      `step ${decisions} updates ${instinct.updates} entropy ${instinct.entropy.toFixed(2)} return ${episode.toFixed(1)} best ${best.toFixed(1)} tiles ${curiosity.tiles.size} maps ${[...seenMaps].join(",")}`,
    );
    save();
  }
  phase = "hold";
  left = HOLD;
}

save();
writeFileSync(join(OUT, "progress.log"), log.join("\n") + "\n");
note("done");
