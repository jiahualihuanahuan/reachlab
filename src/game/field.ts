import type { Cartridge } from "./gb";
import { MAPS, SPECIES } from "./names";

export const ADDR = {
  sprites: 0xcfca,
  joyIgnore: 0xcd6b,
  battle: 0xd056,
  text: 0xd124,
  enemySpecies: 0xcfe4,
  enemyHp: 0xcfe5,
  partyCount: 0xd162,
  partySpecies: 0xd163,
  mon1: 0xd16a,
  monHp: 0xd16b,
  monLevel: 0xd18b,
  monStride: 0x2c,
  playerName: 0xd157,
  dexOwned: 0xd2f6,
  dexSeen: 0xd309,
  bagCount: 0xd31c,
  bagItems: 0xd31d,
  money: 0xd346,
  badges: 0xd355,
  map: 0xd35d,
  y: 0xd360,
  x: 0xd361,
  friendship: 0xd46f,
};

export const OBS = 32;
const BADGE_NAMES = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"];

export function popcount(cart: Cartridge, addr: number, bytes: number) {
  let n = 0;
  for (let i = 0; i < bytes; i++) {
    let v = cart.read(addr + i);
    while (v) {
      n += v & 1;
      v >>= 1;
    }
  }
  return n;
}

export function bcd(cart: Cartridge, addr: number, bytes: number) {
  let n = 0;
  for (let i = 0; i < bytes; i++) {
    const v = cart.read(addr + i);
    n = n * 100 + ((v >> 4) * 10 + (v & 0xf));
  }
  return n;
}

export function pokeText(cart: Cartridge, addr: number, len: number) {
  let s = "";
  for (let i = 0; i < len; i++) {
    const b = cart.read(addr + i);
    if (b === 0x50 || b === 0xff || b === 0) break;
    if (b >= 0x80 && b <= 0x99) s += String.fromCharCode(65 + b - 0x80);
    else if (b >= 0xa0 && b <= 0xb9) s += String.fromCharCode(97 + b - 0xa0);
    else if (b === 0x7f || b === 0xe0) s += " ";
    else if (b >= 0xf6 && b <= 0xff) s += String(b - 0xf6);
  }
  return s;
}

export function speciesName(id: number) {
  return SPECIES[id] || (id ? `Species ${id}` : "—");
}

export function mapName(id: number) {
  return MAPS[id] || `Map ${id}`;
}

export type FieldView = {
  map: number;
  mapName: string;
  x: number;
  y: number;
  battle: number;
  sprites: number;
  joyIgnore: number;
  party: { id: number; name: string; level: number; hp: number; max: number }[];
  player: string;
  dex: number;
  seen: number;
  badges: number;
  badgeNames: string[];
  bag: number;
  money: number;
  friendship: number;
  enemy: string;
  enemyHp: number;
  text: number;
};

export function readField(cart: Cartridge): FieldView {
  const count = cart.read(ADDR.partyCount);
  const party = [];
  for (let i = 0; i < Math.min(6, count); i++) {
    const id = cart.read(ADDR.partySpecies + i);
    const base = ADDR.mon1 + i * ADDR.monStride;
    const hp = cart.read16(base + 1);
    const max = cart.read16(base + (0xd18c - 0xd16a));
    const level = cart.read(base + (ADDR.monLevel - ADDR.mon1));
    party.push({ id, name: speciesName(id), level, hp, max });
  }
  const badgeByte = cart.read(ADDR.badges);
  const badgeNames = BADGE_NAMES.filter((_, i) => badgeByte & (1 << i));
  const battle = cart.read(ADDR.battle);
  const enemyId = cart.read(ADDR.enemySpecies);
  return {
    map: cart.read(ADDR.map),
    mapName: mapName(cart.read(ADDR.map)),
    x: cart.read(ADDR.x),
    y: cart.read(ADDR.y),
    battle,
    sprites: cart.read(ADDR.sprites),
    joyIgnore: cart.read(ADDR.joyIgnore),
    party,
    player: pokeText(cart, ADDR.playerName, 11),
    dex: popcount(cart, ADDR.dexOwned, 19),
    seen: popcount(cart, ADDR.dexSeen, 19),
    badges: badgeNames.length,
    badgeNames,
    bag: cart.read(ADDR.bagCount),
    money: bcd(cart, ADDR.money, 3),
    friendship: cart.read(ADDR.friendship),
    enemy: battle ? speciesName(enemyId) : "",
    enemyHp: battle ? cart.read16(ADDR.enemyHp) : 0,
    text: cart.read(ADDR.text),
  };
}

export type RewardPart = { k: string; v: number };

export class Curiosity {
  tiles = new Set<string>();
  maps = new Set<number>();
  owned = 0;
  seen = 0;
  badges = 0;
  bagIds = new Set<number>();
  levels = 0;
  friend = 0;
  enemyHp = 0;
  ready = false;

  resetScores() {
    this.tiles.clear();
    this.maps.clear();
    this.owned = 0;
    this.seen = 0;
    this.badges = 0;
    this.bagIds.clear();
    this.levels = 0;
    this.friend = 0;
    this.enemyHp = 0;
    this.ready = false;
  }

  private bagSet(cart: Cartridge) {
    const n = Math.min(20, cart.read(ADDR.bagCount));
    const ids = new Set<number>();
    for (let i = 0; i < n; i++) ids.add(cart.read(ADDR.bagItems + i * 2));
    return ids;
  }

  private levelSum(cart: Cartridge) {
    const n = Math.min(6, cart.read(ADDR.partyCount));
    let s = 0;
    for (let i = 0; i < n; i++) s += cart.read(ADDR.monLevel + i * ADDR.monStride);
    return s;
  }

  observe(cart: Cartridge, out: Float32Array, lastAction: number) {
    const f = readField(cart);
    out.fill(0);
    out[0] = f.map / 255;
    out[1] = f.x / 40;
    out[2] = f.y / 40;
    out[3] = f.battle > 0 ? 1 : 0;
    out[4] = f.sprites === 1 ? 1 : 0;
    out[5] = f.party.length / 6;
    out[6] = (f.party[0]?.id ?? 0) / 190;
    out[7] = (f.party[0]?.level ?? 0) / 100;
    const max = f.party[0]?.max || 1;
    out[8] = f.party[0] ? f.party[0].hp / max : 0;
    out[9] = f.badges / 8;
    out[10] = f.dex / 151;
    out[11] = f.seen / 151;
    out[12] = Math.min(1, f.bag / 20);
    out[13] = Math.min(1, f.money / 100000);
    out[14] = f.battle ? Math.min(1, f.enemyHp / 200) : 0;
    out[15] = f.joyIgnore !== 0 || f.sprites !== 1 ? 1 : 0;
    out[16] = f.friendship / 255;
    const key = `${f.map}:${f.x}:${f.y}`;
    out[17] = Math.min(1, (this.tiles.has(key) ? 1 : 0));
    if (lastAction >= 0 && lastAction < 8) out[18 + lastAction] = 1;
    out[26] = f.x / 255;
    out[27] = f.y / 255;
    out[28] = f.text === 0xd1 ? 1 : 0;
    out[29] = f.party.length > 0 && (f.party[0]?.hp ?? 0) === 0 ? 1 : 0;
    out[30] = Math.min(1, this.tiles.size / 400);
    out[31] = Math.min(1, this.maps.size / 40);
  }

  reward(cart: Cartridge): { total: number; parts: RewardPart[]; event: string } {
    const f = readField(cart);
    const parts: RewardPart[] = [];
    const add = (k: string, v: number) => {
      if (v) parts.push({ k, v });
    };
    let event = "";
    if (!this.ready) {
      if (f.sprites === 1) {
        this.ready = true;
        this.maps.add(f.map);
        this.tiles.add(`${f.map}:${f.x}:${f.y}`);
        this.owned = f.dex;
        this.seen = f.seen;
        this.badges = f.badges;
        this.bagIds = this.bagSet(cart);
        this.levels = this.levelSum(cart);
        this.friend = f.friendship;
        this.enemyHp = f.enemyHp;
      }
      return { total: 0, parts, event: "Waking up" };
    }

    const tile = `${f.map}:${f.x}:${f.y}`;
    if (!this.tiles.has(tile)) {
      this.tiles.add(tile);
      add("new ground", 0.55);
      event = `New ground in ${f.mapName}`;
    } else if (f.sprites === 1 && f.battle === 0) add("retrace", -0.02);

    if (!this.maps.has(f.map)) {
      this.maps.add(f.map);
      add("new area", 8);
      event = `Entered ${f.mapName}`;
    }
    if (f.dex > this.owned) {
      add("caught", 18 * (f.dex - this.owned));
      event = `Dex ${f.dex}`;
      this.owned = f.dex;
    }
    if (f.seen > this.seen) {
      add("seen", 3.5 * (f.seen - this.seen));
      if (!event) event = `Seen ${f.seen}`;
      this.seen = f.seen;
    }
    if (f.badges > this.badges) {
      add("badge", 20 * (f.badges - this.badges));
      event = f.badgeNames[f.badgeNames.length - 1] ?? "Badge";
      this.badges = f.badges;
    }
    const bag = this.bagSet(cart);
    let fresh = 0;
    for (const id of bag) if (!this.bagIds.has(id)) fresh++;
    if (fresh) {
      add("item", 4 * fresh);
      event = event || `Bag +${fresh}`;
    }
    this.bagIds = bag;
    const levels = this.levelSum(cart);
    if (levels > this.levels) add("level", 1.6 * (levels - this.levels));
    this.levels = Math.max(this.levels, levels);
    if (f.friendship > this.friend) add("friend", 0.35 * (f.friendship - this.friend));
    this.friend = Math.max(this.friend, f.friendship);
    if (f.battle && this.enemyHp > 0 && f.enemyHp < this.enemyHp) {
      add("battle", 0.7);
    }
    this.enemyHp = f.battle ? f.enemyHp : 0;
    if (f.text === 0xd1) add("blackout", -12);
    add("step", -0.01);
    const total = parts.reduce((s, p) => s + p.v, 0);
    return { total, parts, event };
  }

  locked(cart: Cartridge) {
    return cart.read(ADDR.sprites) !== 1 || cart.read(ADDR.joyIgnore) !== 0;
  }
}
