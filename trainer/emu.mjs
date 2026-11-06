import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const wasm = readFileSync(join(root, "public/emu/binjgb.wasm"));
const source = readFileSync(join(root, "public/emu/binjgb.js"), "utf8");
const module = { exports: {} };
const factory = new Function("module", "exports", `${source}\nreturn module.exports;`)(module, module.exports);
const Binjgb = typeof factory === "function" ? factory : factory.default;

const FRAME = 70224;
const NONE = { up: false, down: false, left: false, right: false, a: false, b: false, start: false, select: false };

export function buttons(action, on) {
  const b = { ...NONE };
  if (!on) return b;
  if (action === 0) b.up = true;
  else if (action === 1) b.right = true;
  else if (action === 2) b.down = true;
  else if (action === 3) b.left = true;
  else if (action === 4) b.a = true;
  else if (action === 5) b.b = true;
  else if (action === 6) b.start = true;
  else b.select = true;
  return b;
}

export async function bootRom(path) {
  const rom = readFileSync(path);
  const mod = await Binjgb({ wasmBinary: wasm, locateFile: () => "binjgb.wasm" });
  const romPtr = mod._malloc(rom.length);
  mod.HEAPU8.set(rom, romPtr);
  const e = mod._emulator_new_simple(romPtr, rom.length, 44100, 2048, 1);
  if (!e) throw new Error("Cartridge rejected");
  const joy = mod._joypad_new();
  mod._emulator_set_default_joypad_callback(e, joy);
  const statePtr = mod._state_file_data_new(e);
  const cart = {
    mod,
    e,
    statePtr,
    checkpoint: null,
    read(addr) {
      return mod._emulator_read_mem(e, addr) & 0xff;
    },
    read16(addr) {
      return this.read(addr) | (this.read(addr + 1) << 8);
    },
    frames(n, b = NONE) {
      mod._set_joyp_up(joy, b.up ? 1 : 0);
      mod._set_joyp_down(joy, b.down ? 1 : 0);
      mod._set_joyp_left(joy, b.left ? 1 : 0);
      mod._set_joyp_right(joy, b.right ? 1 : 0);
      mod._set_joyp_A(joy, b.a ? 1 : 0);
      mod._set_joyp_B(joy, b.b ? 1 : 0);
      mod._set_joyp_start(joy, b.start ? 1 : 0);
      mod._set_joyp_select(joy, b.select ? 1 : 0);
      for (let i = 0; i < n; i++) {
        const target = mod._emulator_get_ticks_f64(e) + FRAME;
        let guard = 0;
        while (mod._emulator_get_ticks_f64(e) < target - 8 && guard++ < 4) {
          mod._emulator_run_until_f64(e, target);
        }
      }
    },
    capture() {
      mod._emulator_write_state(e, statePtr);
      const ptr = mod._get_file_data_ptr(statePtr);
      const size = mod._get_file_data_size(statePtr);
      return mod.HEAPU8.slice(ptr, ptr + size);
    },
    restore(state) {
      const src = state ?? this.checkpoint;
      if (!src) return false;
      const ptr = mod._get_file_data_ptr(statePtr);
      mod.HEAPU8.set(src, ptr);
      return mod._emulator_read_state(e, statePtr) === 0;
    },
  };
  return cart;
}

export async function wake(cart) {
  let frame = 0;
  let bedroom = false;
  for (; frame < 9000; frame++) {
    const b = { ...NONE };
    if (!bedroom) {
      if (frame === 450) b.start = true;
      else if (frame > 500 && frame % 36 === 0) b.a = true;
    }
    cart.frames(1, b);
    if (frame > 2000 && cart.read(0xcfca) === 1 && cart.read(0xd35d) === 0x26) bedroom = true;
    if (bedroom && cart.read(0xc109) !== 0) break;
  }
  cart.frames(8, NONE);
  cart.checkpoint = cart.capture();
  return frame;
}
