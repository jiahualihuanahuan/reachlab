export type Buttons = {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  a: boolean;
  b: boolean;
  start: boolean;
  select: boolean;
};

export const NO_BUTTONS: Buttons = {
  up: false,
  down: false,
  left: false,
  right: false,
  a: false,
  b: false,
  start: false,
  select: false,
};

type EmuModule = {
  HEAPU8: Uint8Array;
  _malloc: (n: number) => number;
  _emulator_new_simple: (rom: number, size: number, freq: number, frames: number, curve: number) => number;
  _joypad_new: () => number;
  _emulator_set_default_joypad_callback: (e: number, joy: number) => void;
  _emulator_get_ticks_f64: (e: number) => number;
  _emulator_run_until_f64: (e: number, until: number) => number;
  _emulator_read_mem: (e: number, addr: number) => number;
  _set_joyp_up: (e: number, v: number) => void;
  _set_joyp_down: (e: number, v: number) => void;
  _set_joyp_left: (e: number, v: number) => void;
  _set_joyp_right: (e: number, v: number) => void;
  _set_joyp_A: (e: number, v: number) => void;
  _set_joyp_B: (e: number, v: number) => void;
  _set_joyp_start: (e: number, v: number) => void;
  _set_joyp_select: (e: number, v: number) => void;
  _get_frame_buffer_ptr: (e: number) => number;
  _state_file_data_new: (e: number) => number;
  _emulator_write_state: (e: number, fd: number) => number;
  _emulator_read_state: (e: number, fd: number) => number;
  _get_file_data_ptr: (fd: number) => number;
  _get_file_data_size: (fd: number) => number;
};

const FRAME = 70224;

declare global {
  interface Window {
    Binjgb?: (opts: { locateFile: (path: string) => string }) => Promise<EmuModule>;
  }
}

function loadScript(): Promise<void> {
  if (window.Binjgb) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/emu/binjgb.js";
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Simulator failed to load"));
    document.head.appendChild(s);
  });
}

export class Cartridge {
  mod: EmuModule | null = null;
  e = 0;
  statePtr = 0;
  image = new ImageData(160, 144);
  checkpoint: Uint8Array | null = null;

  async boot(rom: Uint8Array) {
    await loadScript();
    const factory = window.Binjgb;
    if (!factory) throw new Error("Simulator failed to start");
    this.mod = await factory({ locateFile: (path) => `/emu/${path}` });
    const romPtr = this.mod._malloc(rom.length);
    this.mod.HEAPU8.set(rom, romPtr);
    this.e = this.mod._emulator_new_simple(romPtr, rom.length, 44100, 2048, 1);
    if (!this.e) throw new Error("Cartridge rejected");
    this.mod._emulator_set_default_joypad_callback(this.e, this.mod._joypad_new());
    this.statePtr = this.mod._state_file_data_new(this.e);
  }

  read(addr: number) {
    return this.mod ? this.mod._emulator_read_mem(this.e, addr) & 0xff : 0;
  }

  read16(addr: number) {
    return this.read(addr) | (this.read(addr + 1) << 8);
  }

  private apply(b: Buttons) {
    const m = this.mod;
    if (!m) return;
    m._set_joyp_up(this.e, b.up ? 1 : 0);
    m._set_joyp_down(this.e, b.down ? 1 : 0);
    m._set_joyp_left(this.e, b.left ? 1 : 0);
    m._set_joyp_right(this.e, b.right ? 1 : 0);
    m._set_joyp_A(this.e, b.a ? 1 : 0);
    m._set_joyp_B(this.e, b.b ? 1 : 0);
    m._set_joyp_start(this.e, b.start ? 1 : 0);
    m._set_joyp_select(this.e, b.select ? 1 : 0);
  }

  frames(n: number, b: Buttons = NO_BUTTONS) {
    const m = this.mod;
    if (!m) return;
    this.apply(b);
    for (let i = 0; i < n; i++) {
      const target = m._emulator_get_ticks_f64(this.e) + FRAME;
      let guard = 0;
      while (m._emulator_get_ticks_f64(this.e) < target - 8 && guard++ < 4) {
        m._emulator_run_until_f64(this.e, target);
      }
    }
  }

  present(ctx: CanvasRenderingContext2D) {
    const m = this.mod;
    if (!m) return;
    const ptr = m._get_frame_buffer_ptr(this.e);
    const src = m.HEAPU8.subarray(ptr, ptr + 160 * 144 * 4);
    this.image.data.set(src);
    ctx.putImageData(this.image, 0, 0);
  }

  /** Scripted intro that lands in the bedroom with sprites enabled. */
  wake(onProgress: (frame: number) => void) {
    const m = this.mod;
    if (!m) return 0;
    let frame = 0;
    for (; frame < 9000; frame++) {
      const b = { ...NO_BUTTONS };
      if (frame === 450) b.start = true;
      else if (frame > 500 && frame % 36 === 0) b.a = true;
      this.frames(1, b);
      if (frame > 2000 && frame % 20 === 0) {
        onProgress(frame);
        if (this.read(0xcfca) === 1 && this.read(0xd35d) === 0x26) break;
      }
    }
    this.frames(2, NO_BUTTONS);
    this.checkpoint = this.capture();
    return frame;
  }

  capture() {
    const m = this.mod;
    if (!m || !this.statePtr) return null;
    m._emulator_write_state(this.e, this.statePtr);
    const ptr = m._get_file_data_ptr(this.statePtr);
    const size = m._get_file_data_size(this.statePtr);
    return m.HEAPU8.slice(ptr, ptr + size);
  }

  restore(state?: Uint8Array | null) {
    const m = this.mod;
    const src = state ?? this.checkpoint;
    if (!m || !this.statePtr || !src) return false;
    const ptr = m._get_file_data_ptr(this.statePtr);
    m.HEAPU8.set(src, ptr);
    return m._emulator_read_state(this.e, this.statePtr) === 0;
  }
}
