export const TRAINER = "https://192.168.86.35:8010";

export type GpuHealth = {
  ok: boolean;
  device: string;
  gpu: string;
  params: number;
  width: number;
  updates: number;
  vram_cap_mb: number;
  free_mb: number;
  total_mb: number;
};

export type GpuSample = {
  action: number;
  logp: number;
  value: number;
  entropy: number;
  hidden: number[];
};

export type GpuRollout = {
  obs: number[][];
  actions: number[];
  logp: number[];
  values: number[];
  rewards: number[];
  dones: boolean[];
  h0: number[] | null;
  bootstrap: number;
};

function base(url: string) {
  return url.replace(/\/$/, "");
}

function candidates(url: string) {
  const root = base(url);
  if (root.startsWith("https://")) return [root, `http://${root.slice("https://".length)}`];
  return [root];
}

let working = "";

async function call(url: string, path: string, init?: RequestInit) {
  const options = candidates(url);
  const order = working && options.some((item) => item === working) ? [working, ...options.filter((item) => item !== working)] : options;
  let last = "trainer unreachable";
  for (const root of order) {
    try {
      const res = await fetch(`${root}${path}`, init);
      working = root;
      return res;
    } catch (err) {
      last = err instanceof Error ? err.message : "trainer unreachable";
    }
  }
  throw new Error(last);
}

export async function gpuHealth(url: string): Promise<GpuHealth | null> {
  try {
    const res = await call(url, "/health", { signal: AbortSignal.timeout(2500) });
    if (!res.ok) return null;
    return (await res.json()) as GpuHealth;
  } catch {
    return null;
  }
}

export async function gpuAct(url: string, obs: number[], hidden: number[] | null): Promise<GpuSample> {
  const res = await call(url, "/act", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ obs, hidden }),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as GpuSample;
}

export async function gpuUpdate(url: string, roll: GpuRollout): Promise<{ updates: number; entropy: number }> {
  const res = await call(url, "/update", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(roll),
  });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as { updates: number; entropy: number };
}

export async function gpuReset(url: string) {
  try {
    await call(url, "/reset", { method: "POST" });
  } catch {
    /* trainer can be offline */
  }
}
