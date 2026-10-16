import { ACTION_NAMES } from "./instinct";
import type { FieldView } from "./field";

export type Elite = { ret: number; note: string; action: string };

export function fieldBrief(f: FieldView, event: string) {
  const lead = f.party[0];
  const party = lead ? `${lead.name} L${lead.level} ${lead.hp}/${lead.max}` : "no party";
  const foe = f.battle ? ` battle ${f.enemy} hp ${f.enemyHp}` : "";
  return `${f.mapName} (${f.x},${f.y}) ${party} dex ${f.dex}/${f.seen} badges ${f.badges} bag ${f.bag} ₽${f.money} friend ${f.friendship}${foe} ${event}`.trim();
}

export function parseAction(text: string): (typeof ACTION_NAMES)[number] | null {
  const cleaned = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const match = cleaned.match(/\{[\s\S]*\}/);
  const body = match ? match[0] : cleaned;
  try {
    const data = JSON.parse(body) as { action?: string };
    const name = String(data.action ?? "").toLowerCase();
    return ACTION_NAMES.find((a) => a === name) ?? null;
  } catch {
    const hit = ACTION_NAMES.find((a) => cleaned.toLowerCase().includes(`"${a}"`) || cleaned.toLowerCase().includes(a));
    return hit ?? null;
  }
}

export function systemPrompt(elites: Elite[]) {
  const lines = elites
    .slice(0, 6)
    .map((e) => `- return ${e.ret.toFixed(1)}: ${e.action} — ${e.note}`)
    .join("\n");
  return [
    "You guide a Yellow cartridge. Reply with JSON only: {\"action\":\"north|east|south|west|a|b|start|select\"}.",
    "Philosophy: curiosity is gated. New ground, new areas, new dex entries, and new items are the score. Towns are rest, not progress. Routes are chapters. Press A to talk and confirm. Walk off the map edge to leave. In battle, type and HP matter; do not black out. Do not mash Start.",
    "Leave the bedroom, meet Oak, take the partner, then walk new tiles.",
    lines ? `Elites:\n${lines}` : "No elites yet. Explore.",
  ].join("\n");
}

export async function askQwen(opts: {
  url: string;
  model: string;
  brief: string;
  elites: Elite[];
}) {
  const base = opts.url.replace(/\/$/, "");
  const res = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: opts.model,
      stream: false,
      think: false,
      format: "json",
      options: { temperature: 0.35, num_ctx: 2048, num_predict: 48 },
      messages: [
        { role: "system", content: systemPrompt(opts.elites) },
        { role: "user", content: opts.brief },
      ],
    }),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const data = (await res.json()) as { message?: { content?: string } };
  const content = data.message?.content ?? "";
  const action = parseAction(content);
  if (!action) throw new Error(content.slice(0, 180) || "Empty guide reply");
  return { action, content };
}

export function modelfile(elites: Elite[], model: string) {
  const sys = systemPrompt(elites).replaceAll('"""', "'''");
  return `FROM ${model}\nPARAMETER num_ctx 2048\nPARAMETER temperature 0.35\nPARAMETER num_predict 48\nSYSTEM """\n${sys}\n"""\n`;
}
