import { useEffect, useRef, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { Cartridge, NO_BUTTONS, type Buttons } from "@/game/gb";
import { Curiosity, readField, type FieldView, type RewardPart } from "@/game/field";
import { ACTION_NAMES, Instinct } from "@/game/instinct";
import { askQwen, fieldBrief, modelfile, type Elite } from "@/game/qwen";

const OLLAMA = "http://192.168.86.35:11434";
const HOLD = 8;
const GAP = 2;
const ROLLOUT = 28;
const STORE = "reachlab-yellow-v1";

type Store = {
  weights?: number[];
  best?: number;
  url?: string;
  model?: string;
  elites?: Elite[];
};

function buttonOf(action: number, on: boolean): Buttons {
  const b = { ...NO_BUTTONS };
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

function emptyField(): FieldView {
  return {
    map: 0,
    mapName: "—",
    x: 0,
    y: 0,
    battle: 0,
    sprites: 0,
    joyIgnore: 0,
    party: [],
    player: "",
    dex: 0,
    seen: 0,
    badges: 0,
    badgeNames: [],
    bag: 0,
    money: 0,
    friendship: 0,
    enemy: "",
    enemyHp: 0,
    text: 0,
  };
}

export function YellowLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cartRef = useRef<Cartridge | null>(null);
  const instinctRef = useRef(new Instinct());
  const curiosityRef = useRef(new Curiosity());
  const obsRef = useRef(new Float32Array(32));
  const humanRef = useRef<Buttons>({ ...NO_BUTTONS });
  const trainRef = useRef(false);
  const guideRef = useRef(false);
  const speedRef = useRef(1);
  const aliveRef = useRef(true);
  const frameRef = useRef(0);
  const phaseRef = useRef<"hold" | "gap">("hold");
  const phaseLeftRef = useRef(HOLD);
  const actionRef = useRef(0);
  const lastActionRef = useRef(-1);
  const forcedRef = useRef(false);
  const rolloutRef = useRef({
    obs: [] as Float32Array[],
    actions: [] as number[],
    logp: [] as number[],
    values: [] as number[],
    rewards: [] as number[],
    dones: [] as boolean[],
  });
  const episodeRef = useRef(0);
  const fileRef = useRef<HTMLInputElement>(null);

  const [status, setStatus] = useState("Load a Yellow cartridge to begin.");
  const [booted, setBooted] = useState(false);
  const [waking, setWaking] = useState(false);
  const [training, setTraining] = useState(false);
  const [guideOn, setGuideOn] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [tab, setTab] = useState<"field" | "train" | "guide">("field");
  const [field, setField] = useState<FieldView>(emptyField);
  const [log, setLog] = useState<string[]>([]);
  const [parts, setParts] = useState<RewardPart[]>([]);
  const [returns, setReturns] = useState<{ i: number; r: number }[]>([]);
  const [entropy, setEntropy] = useState(1);
  const [updates, setUpdates] = useState(0);
  const [episodeReturn, setEpisodeReturn] = useState(0);
  const [best, setBest] = useState(0);
  const [url, setUrl] = useState(OLLAMA);
  const [model, setModel] = useState("qwen3.5:9b");
  const [elites, setElites] = useState<Elite[]>([]);
  const [guideNote, setGuideNote] = useState("Guide is idle.");
  const [decision, setDecision] = useState("—");
  const wakingRef = useRef(false);
  const elitesRef = useRef<Elite[]>([]);
  const urlRef = useRef(url);
  const modelRef = useRef(model);
  const uiRef = useRef(0);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORE);
      if (!raw) return;
      const saved = JSON.parse(raw) as Store;
      if (saved.weights?.length) instinctRef.current.import(Float32Array.from(saved.weights));
      if (saved.best) setBest(saved.best);
      if (saved.url && saved.url !== "http://127.0.0.1:11434") setUrl(saved.url);
      if (saved.model) setModel(saved.model);
      if (saved.elites) setElites(saved.elites);
    } catch {
      /* ignore a bad save */
    }
  }, []);

  useEffect(() => {
    const payload: Store = {
      weights: Array.from(instinctRef.current.export()),
      best,
      url,
      model,
      elites,
    };
    localStorage.setItem(STORE, JSON.stringify(payload));
  }, [best, url, model, elites, updates]);

  useEffect(() => {
    trainRef.current = training;
  }, [training]);
  useEffect(() => {
    guideRef.current = guideOn;
  }, [guideOn]);
  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const map: Record<string, keyof Buttons> = {
        ArrowUp: "up",
        ArrowDown: "down",
        ArrowLeft: "left",
        ArrowRight: "right",
        x: "a",
        X: "a",
        z: "b",
        Z: "b",
        Enter: "start",
        Shift: "select",
      };
      const key = map[e.key];
      if (!key) return;
      e.preventDefault();
      humanRef.current[key] = true;
    };
    const up = (e: KeyboardEvent) => {
      const map: Record<string, keyof Buttons> = {
        ArrowUp: "up",
        ArrowDown: "down",
        ArrowLeft: "left",
        ArrowRight: "right",
        x: "a",
        X: "a",
        z: "b",
        Z: "b",
        Enter: "start",
        Shift: "select",
      };
      const key = map[e.key];
      if (!key) return;
      humanRef.current[key] = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    let raf = 0;
    let acc = 0;
    let last = performance.now();
    let painted = 0;
    const tick = (now: number) => {
      if (!aliveRef.current) return;
      const cart = cartRef.current;
      const dt = Math.min(48, now - last);
      last = now;
      acc += dt;
      const budget = speedRef.current >= 12 ? 90 : speedRef.current >= 4 ? 36 : 8;
      let stepped = 0;
      while (cart?.mod && !wakingRef.current && acc > 16 && stepped < budget) {
        acc -= 16;
        stepped++;
        stepOnce(cart);
      }
      if (cart?.mod && now - painted > 80) {
        const ctx = canvasRef.current?.getContext("2d", { alpha: false });
        if (ctx) cart.present(ctx);
        painted = now;
        if (now - uiRef.current > 120) {
          uiRef.current = now;
          setField(readField(cart));
          setEntropy(instinctRef.current.entropy);
          setUpdates(instinctRef.current.updates);
          setEpisodeReturn(episodeRef.current);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      aliveRef.current = false;
      cancelAnimationFrame(raf);
    };
    // The loop reads refs. It must not restart when UI state changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function pushLog(line: string) {
    if (!line) return;
    setLog((prev) => [line, ...prev].slice(0, 12));
  }

  function stepOnce(cart: Cartridge) {
    frameRef.current++;
    const human = humanRef.current;
    const manual = Object.values(human).some(Boolean);
    if (manual) {
      cart.frames(1, human);
      return;
    }
    if (!trainRef.current && !guideRef.current) {
      cart.frames(1, NO_BUTTONS);
      return;
    }
    phaseLeftRef.current--;
    if (phaseLeftRef.current > 0) {
      const pressed = phaseRef.current === "hold" && !forcedRef.current ? buttonOf(actionRef.current, true) : buttonOf(4, forcedRef.current);
      cart.frames(1, phaseRef.current === "hold" ? pressed : NO_BUTTONS);
      return;
    }
    if (phaseRef.current === "hold") {
      phaseRef.current = "gap";
      phaseLeftRef.current = GAP;
      cart.frames(1, NO_BUTTONS);
      return;
    }
    const curiosity = curiosityRef.current;
    const locked = curiosity.locked(cart);
    const result = curiosity.reward(cart);
    episodeRef.current += result.total;
    if (result.event && result.event !== "Waking up") pushLog(result.event);
    if (result.parts.length) setParts(result.parts);
    const roll = rolloutRef.current;
    if (!forcedRef.current && roll.obs.length) {
      roll.rewards[roll.rewards.length - 1] = result.total;
      roll.dones[roll.dones.length - 1] = false;
    }
    if (locked) {
      forcedRef.current = true;
      actionRef.current = 4;
      phaseRef.current = "hold";
      phaseLeftRef.current = 1;
      cart.frames(1, buttonOf(4, true));
      return;
    }
    forcedRef.current = false;
    curiosity.observe(cart, obsRef.current, lastActionRef.current);
    const instinct = instinctRef.current;
    const sample = instinct.sample(obsRef.current, false);
    let action = sample.action;
    if (guideRef.current && roll.obs.length % 6 === 0) {
      const brief = fieldBrief(readField(cart), result.event);
      const elitesNow = elitesRef.current;
      void askQwen({ url: urlRef.current, model: modelRef.current, brief, elites: elitesNow })
        .then((guide) => {
          const idx = ACTION_NAMES.indexOf(guide.action);
          if (idx >= 0) actionRef.current = idx;
          setGuideNote(guide.action);
          setDecision(guide.action);
        })
        .catch((err: unknown) => {
          setGuideNote(err instanceof Error ? err.message : "Guide unreachable");
        });
    }
    actionRef.current = action;
    lastActionRef.current = action;
    setDecision(ACTION_NAMES[action]);
    roll.obs.push(new Float32Array(obsRef.current));
    roll.actions.push(action);
    roll.logp.push(sample.logp);
    roll.values.push(sample.value);
    roll.rewards.push(0);
    roll.dones.push(false);
    if (roll.obs.length >= ROLLOUT) {
      curiosity.observe(cart, obsRef.current, action);
      const boot = instinct.sample(obsRef.current, true).value;
      instinct.update(roll.obs, roll.actions, roll.logp, roll.values, roll.rewards, roll.dones, boot);
      const ep = episodeRef.current;
      setReturns((prev) => [...prev, { i: prev.length + 1, r: ep }].slice(-80));
      setBest((prev) => {
        if (ep > prev && ep > 1) {
          const note = fieldBrief(readField(cart), result.event);
          setElites((list) => [{ ret: ep, note, action: ACTION_NAMES[action] }, ...list].slice(0, 8));
        }
        return Math.max(prev, ep);
      });
      roll.obs = [];
      roll.actions = [];
      roll.logp = [];
      roll.values = [];
      roll.rewards = [];
      roll.dones = [];
    }
    phaseRef.current = "hold";
    phaseLeftRef.current = HOLD;
    cart.frames(1, buttonOf(action, true));
  }

  useEffect(() => {
    elitesRef.current = elites;
    urlRef.current = url;
    modelRef.current = model;
  }, [elites, url, model]);

  async function bootRom(bytes: Uint8Array) {
    setStatus("Booting the simulator…");
    setWaking(true);
    wakingRef.current = true;
    const cart = new Cartridge();
    cartRef.current = cart;
    curiosityRef.current.resetScores();
    episodeRef.current = 0;
    try {
      await cart.boot(bytes);
      setBooted(true);
      setStatus("Waking the bedroom. This takes a few seconds.");
      await new Promise((r) => setTimeout(r, 30));
      const ctx = canvasRef.current?.getContext("2d", { alpha: false });
      let frame = 0;
      const slice = () =>
        new Promise<void>((resolve) => {
          const run = () => {
            const end = Math.min(frame + 180, 9000);
            for (; frame < end; frame++) {
              const b = { ...NO_BUTTONS };
              if (frame === 450) b.start = true;
              else if (frame > 500 && frame % 36 === 0) b.a = true;
              cart.frames(1, b);
              if (frame > 2000 && cart.read(0xcfca) === 1 && cart.read(0xd35d) === 0x26) {
                resolve();
                return;
              }
            }
            if (ctx) cart.present(ctx);
            setStatus(`Waking… frame ${frame}`);
            if (frame >= 9000) resolve();
            else requestAnimationFrame(run);
          };
          requestAnimationFrame(run);
        });
      await slice();
      cart.frames(2, NO_BUTTONS);
      cart.checkpoint = cart.capture();
      if (ctx) cart.present(ctx);
      setField(readField(cart));
      const ready = cart.read(0xcfca) === 1;
      setStatus(ready ? "Bedroom is live. Train, or walk it yourself." : "Intro did not hand over control. Reset and try again.");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Boot failed");
    } finally {
      setWaking(false);
      wakingRef.current = false;
    }
  }

  async function loadServer() {
    setStatus("Looking for the attached cartridge…");
    const res = await fetch("/api/rom");
    if (!res.ok) {
      setStatus("No cartridge on this machine. Choose the Yellow ROM from your files.");
      fileRef.current?.click();
      return;
    }
    const buf = new Uint8Array(await res.arrayBuffer());
    await bootRom(buf);
  }

  function onFile(file: File | undefined) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const buf = new Uint8Array(reader.result as ArrayBuffer);
      void bootRom(buf);
    };
    reader.readAsArrayBuffer(file);
  }

  function resetEpisode() {
    const cart = cartRef.current;
    if (!cart) return;
    cart.restore();
    episodeRef.current = 0;
    rolloutRef.current = { obs: [], actions: [], logp: [], values: [], rewards: [], dones: [] };
    setStatus("Restored the bedroom checkpoint. Curiosity memory stays.");
  }

  function downloadModelfile() {
    const text = modelfile(elites, model);
    const blob = new Blob([text], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "Reachlab.Modelfile";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function testGuide() {
    const cart = cartRef.current;
    if (!cart?.mod) {
      setGuideNote("Boot the cartridge first.");
      return;
    }
    setGuideNote("Asking the local model…");
    try {
      const guide = await askQwen({
        url,
        model,
        brief: fieldBrief(readField(cart), "probe"),
        elites,
      });
      setGuideNote(`Heard ${guide.action}`);
    } catch (err) {
      setGuideNote(err instanceof Error ? err.message : "Guide unreachable");
    }
  }

  const hold = (key: keyof Buttons, on: boolean) => {
    humanRef.current[key] = on;
  };

  return (
    <div className="mx-auto min-h-full max-w-6xl px-4 py-5 md:px-6">
      <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs tracking-[0.22em] text-amber uppercase">Field lab</p>
          <h1 className="font-display text-4xl text-paper">Reachlab</h1>
        </div>
        <p className="max-w-md text-sm text-sage">{status}</p>
      </header>

      <div className="grid items-start gap-4 lg:grid-cols-[auto_1fr]">
        <section className="rounded-xl border border-line bg-moss-2 p-3">
          <canvas
            ref={canvasRef}
            width={160}
            height={144}
            className="mx-auto h-auto w-full max-w-[480px] bg-ink [image-rendering:pixelated]"
          />
          <div className="mt-3 grid grid-cols-[1fr_auto_1fr] items-center gap-2">
            <Pad
              onHold={hold}
              disabled={!booted || waking}
            />
            <div className="flex flex-col gap-2">
              <Face label="B" k="b" onHold={hold} disabled={!booted || waking} />
              <Face label="A" k="a" onHold={hold} disabled={!booted || waking} />
            </div>
            <div className="flex flex-col items-end gap-2">
              <Small label="Select" k="select" onHold={hold} disabled={!booted || waking} />
              <Small label="Start" k="start" onHold={hold} disabled={!booted || waking} />
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              className="min-h-11 rounded-md bg-amber px-3 text-sm font-medium text-ink"
              onClick={() => void loadServer()}
              disabled={waking}
            >
              {booted ? "Reload ROM" : "Load cartridge"}
            </button>
            <button
              type="button"
              className="min-h-11 rounded-md border border-line px-3 text-sm"
              onClick={() => fileRef.current?.click()}
              disabled={waking}
            >
              Choose file
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".gbc,.gb"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
            <button
              type="button"
              className="min-h-11 rounded-md border border-line px-3 text-sm"
              disabled={!booted || waking}
              onClick={() => {
                setTraining((v) => !v);
                setStatus(training ? "Trainer paused." : "Instinct is walking.");
              }}
            >
              {training ? "Pause" : "Train"}
            </button>
            {[1, 4, 12].map((n) => (
              <button
                key={n}
                type="button"
                className={`min-h-11 rounded-md px-3 text-sm ${speed === n ? "bg-moss-3 text-amber" : "border border-line"}`}
                onClick={() => setSpeed(n)}
              >
                {n}×
              </button>
            ))}
            <button
              type="button"
              className="min-h-11 rounded-md border border-line px-3 text-sm"
              disabled={!booted || waking}
              onClick={resetEpisode}
            >
              Reset
            </button>
          </div>
          <p className="mt-2 text-xs text-sage">Arrows move. X is A, Z is B, Enter starts, Shift selects. Holding a key overrides the trainer.</p>
        </section>

        <section className="rounded-xl border border-line bg-moss-2 p-3">
          <div className="mb-3 flex gap-2">
            {(["field", "train", "guide"] as const).map((name) => (
              <button
                key={name}
                type="button"
                className={`min-h-11 rounded-md px-3 text-sm capitalize ${tab === name ? "bg-amber text-ink" : "border border-line"}`}
                onClick={() => setTab(name)}
              >
                {name}
              </button>
            ))}
          </div>

          {tab === "field" && (
            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
                <Stat k="Place" v={`${field.mapName} ${field.x},${field.y}`} />
                <Stat k="Trainer" v={field.player || "—"} />
                <Stat k="Dex" v={`${field.dex} owned · ${field.seen} seen`} />
                <Stat k="Badges" v={field.badgeNames.join(", ") || "None"} />
                <Stat k="Bag / money" v={`${field.bag} · ₽${field.money}`} />
                <Stat k="Friendship" v={String(field.friendship)} />
              </div>
              <div>
                <p className="mb-1 text-xs tracking-wide text-sage uppercase">Party</p>
                {field.party.length === 0 ? (
                  <p className="text-paper-dim">Empty until Oak hands over the partner.</p>
                ) : (
                  <ul className="space-y-1">
                    {field.party.map((mon) => (
                      <li key={`${mon.id}-${mon.level}`} className="flex justify-between rounded-md bg-moss-3 px-2 py-1">
                        <span>{mon.name} · L{mon.level}</span>
                        <span>{mon.hp}/{mon.max}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              {field.battle > 0 && (
                <p className="rounded-md bg-moss-3 px-2 py-1">Wild {field.enemy} · HP {field.enemyHp}</p>
              )}
              <div>
                <p className="mb-1 text-xs tracking-wide text-sage uppercase">Last reward</p>
                <p>{parts.length ? parts.map((p) => `${p.k} ${p.v > 0 ? "+" : ""}${p.v.toFixed(2)}`).join(" · ") : "—"}</p>
              </div>
              <ul className="space-y-1 text-paper-dim">
                {log.map((line, i) => (
                  <li key={`${line}-${i}`}>{line}</li>
                ))}
              </ul>
            </div>
          )}

          {tab === "train" && (
            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                <Stat k="Return" v={episodeReturn.toFixed(1)} />
                <Stat k="Best" v={best.toFixed(1)} />
                <Stat k="Updates" v={String(updates)} />
                <Stat k="Entropy" v={entropy.toFixed(2)} />
              </div>
              <p className="text-paper-dim">Last decision {decision}. Tiles {curiosityRef.current.tiles.size} · areas {curiosityRef.current.maps.size}</p>
              <div className="h-36">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={returns}>
                    <YAxis hide domain={["auto", "auto"]} />
                    <Tooltip
                      contentStyle={{ background: "#1b221d", border: "1px solid #3a4a3e", fontSize: 12 }}
                      labelStyle={{ color: "#c5d4c4" }}
                    />
                    <Line type="monotone" dataKey="r" stroke="#e0a45a" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <label className="block text-xs text-sage">
                Ollama URL
                <input
                  className="mt-1 min-h-11 w-full rounded-md border border-line bg-moss px-2 text-sm text-paper"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
              </label>
              <label className="block text-xs text-sage">
                Model
                <input
                  className="mt-1 min-h-11 w-full rounded-md border border-line bg-moss px-2 text-sm text-paper"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                />
              </label>
              <div className="flex flex-wrap gap-2">
                <button type="button" className="min-h-11 rounded-md bg-amber px-3 text-sm text-ink" onClick={() => void testGuide()}>
                  Test guide
                </button>
                <button
                  type="button"
                  className={`min-h-11 rounded-md px-3 text-sm ${guideOn ? "bg-moss-3 text-amber" : "border border-line"}`}
                  onClick={() => setGuideOn((v) => !v)}
                >
                  {guideOn ? "Guide on" : "Guide off"}
                </button>
                <button type="button" className="min-h-11 rounded-md border border-line px-3 text-sm" onClick={downloadModelfile}>
                  Download Modelfile
                </button>
              </div>
              <p className="text-paper-dim">{guideNote}</p>
              <ul className="space-y-1">
                {elites.map((e, i) => (
                  <li key={`${e.ret}-${i}`} className="rounded-md bg-moss-3 px-2 py-1">
                    {e.ret.toFixed(1)} · {e.action} · {e.note}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {tab === "guide" && (
            <div className="space-y-3 text-sm leading-relaxed text-paper-dim">
              <p className="text-paper">The cartridge is yours. Reachlab emulates it here and does not ship the ROM. A published copy asks you to choose the file.</p>
              <p>Yellow is a curiosity game with gates. Towns are safe and almost zero-novelty. Routes are chapters. The dex is the long horizon: seeing is progress, catching is the score. Type advantage and remaining HP are the battle economy. Items and money are tension, not loot spam. Friendship on the partner is a slow resource. Blacking out is the real penalty.</p>
              <p>How to operate: load the cartridge, wait until the bedroom accepts input, then Train. The instinct holds a direction for a few frames and releases, because menus ignore a stuck button. Dialogue locks force A and those steps are not trained. Reset returns to the bedroom checkpoint without wiping the map memory. Your arrow keys override the policy.</p>
              <p>Rewards: new ground +0.55, retrace −0.02, new area +8, each new owned dex entry +18, seen +3.5, badge +20, new bag item +4, level +1.6, friendship +0.35, a battle HP drop +0.7, blackout −12, step −0.01.</p>
              <p>The 10 GB card cannot backprop a 9B model. Qwen 3.5 9B stays frozen in Ollama as a guide. The instinct is a tiny network in the browser. Distill by downloading the Modelfile (`ollama create reachlab -f Reachlab.Modelfile`). One model only. Context 2048. Do not load a second copy.</p>
              <p>Point the guide at the Ollama already running on the LAN: <span className="text-paper">http://192.168.86.35:11434</span>, model <span className="text-paper">qwen3.5:9b</span>. That copy is only queried. Its weights are not updated. The server must be started with <span className="text-paper">OLLAMA_ORIGINS=*</span> or the browser will be refused. If the test fails, the error is shown as-is.</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-md bg-moss-3 px-2 py-2">
      <p className="text-[11px] tracking-wide text-sage uppercase">{k}</p>
      <p className="text-paper">{v}</p>
    </div>
  );
}

function Pad({
  onHold,
  disabled,
}: {
  onHold: (k: keyof Buttons, on: boolean) => void;
  disabled: boolean;
}) {
  const key = (k: keyof Buttons, label: string) => (
    <button
      type="button"
      disabled={disabled}
      className="flex h-12 w-12 items-center justify-center rounded-md border border-line bg-moss text-lg"
      onPointerDown={() => onHold(k, true)}
      onPointerUp={() => onHold(k, false)}
      onPointerLeave={() => onHold(k, false)}
    >
      {label}
    </button>
  );
  return (
    <div className="grid w-36 grid-cols-3 justify-items-center gap-1">
      <span />
      {key("up", "↑")}
      <span />
      {key("left", "←")}
      <span />
      {key("right", "→")}
      <span />
      {key("down", "↓")}
      <span />
    </div>
  );
}

function Face({
  label,
  k,
  onHold,
  disabled,
}: {
  label: string;
  k: keyof Buttons;
  onHold: (key: keyof Buttons, on: boolean) => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      className="h-14 w-14 rounded-full border border-amber bg-moss-3 text-sm font-medium text-amber"
      onPointerDown={() => onHold(k, true)}
      onPointerUp={() => onHold(k, false)}
      onPointerLeave={() => onHold(k, false)}
    >
      {label}
    </button>
  );
}

function Small({
  label,
  k,
  onHold,
  disabled,
}: {
  label: string;
  k: keyof Buttons;
  onHold: (key: keyof Buttons, on: boolean) => void;
  disabled: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      className="min-h-11 min-w-20 rounded-full border border-line px-3 text-xs"
      onPointerDown={() => onHold(k, true)}
      onPointerUp={() => onHold(k, false)}
      onPointerLeave={() => onHold(k, false)}
    >
      {label}
    </button>
  );
}
