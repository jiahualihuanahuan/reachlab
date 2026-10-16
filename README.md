# Reachlab

A browser field lab for a Game Boy Yellow cartridge. It boots the game in [binjgb](https://github.com/binji/binjgb) (MIT), scores curiosity from RAM, and trains a tiny policy in the page. A frozen Qwen 3.5 9B in Ollama can guide play. The 9B weights are not updated.

The cartridge is not in this repo. Use a ROM you legally own. Do not commit it.

## Docker Compose

Needs Docker Compose v2. The guide calls the Ollama you already run at `http://192.168.86.35:11434`. Compose does not start a second model, and it does not change `qwen3.5:9b`.

```bash
mkdir -p roms
cp "/path/to/Pokemon - Yellow Version.gbc" roms/yellow.gbc
docker compose up --build
```

Open http://localhost:8080.

- `web` serves the lab on port 8080 and reads `/roms/yellow.gbc`.
- The guide defaults to `http://192.168.86.35:11434` and model `qwen3.5:9b`. That server is only queried. Start it with `OLLAMA_ORIGINS=*` so the browser may call it.
- The trained policy is not in the browser. On the 3080 machine, start the trainer (below). The page defaults to `http://192.168.86.35:8010`.

No database. The small fallback net stays in the browser (`localStorage` key `reachlab-yellow-v1`). The GPU policy is stored on the 3080 in the trainer checkpoint. `qwen3.5:9b` is not modified.

### Without Docker

Node 22 (or 20.19+):

```bash
npm install
npm run dev
```

Then open http://localhost:8080 and use **Choose file**. The server ROM path is `ROM_PATH` (default inside this sandbox only). The guide still calls `http://192.168.86.35:11434`.

## How to train

Training is the small network in the browser, not a GPU job on Qwen.

1. Load the cartridge (**Load cartridge** if `roms/yellow.gbc` is mounted, otherwise **Choose file**).
2. Wait until the status says the bedroom is live. The wake script presses Start, then A, until the overworld accepts input.
3. Press **Train**. The instinct holds a button for 8 frames and releases for 2, because a stuck key does not advance menus.
4. Use **4×** or **12×** to speed the emulator. Arrow keys, or the on-screen pad, override the policy while held. X is A, Z is B, Enter is Start, Shift is Select.
5. Open the **Train** tab.
   - **Return** is the episode score since the last bedroom checkpoint.
   - **Best** is the highest return this browser has stored.
   - **Updates** increments every 28 decisions (one PPO step: GAE, clip 0.2, entropy bonus, Adam).
   - **Entropy** should fall slowly as the policy gets less random. If it collapses near 0 immediately, press **Reset** and keep training.
   - The chart is return at each update.
6. **Reset** reloads the bedroom checkpoint. Curiosity (tiles, maps, dex, items) and the network weights stay. A new cartridge boot clears the curiosity score for that run, not the saved weights.
7. Leave the tab open. Closing the page stops training. Weights are written to `localStorage` after each update.

What it is rewarded for, read from cartridge RAM:

| Event | Score |
|---|---|
| New tile | +0.55 |
| Retrace | −0.02 |
| New area | +8 |
| New Pokédex owned entry | +18 |
| New seen entry | +3.5 |
| Badge | +20 |
| New bag item | +4 |
| Party level gained | +1.6 |
| Friendship gained | +0.35 per point |
| Enemy HP drop in battle | +0.7 |
| Blackout | −12 |
| Each decision | −0.01 |

Dialogue and other locks force A. Those steps are not trained. The useful run is: leave the bedroom, get through the intro, take the partner, then walk new tiles. Towns score almost nothing. New routes, catches, items, and badges are the point.

## Train on the 3080

Qwen stays loaded. The policy is a separate recurrent network (GRU, width 2560, about 110M parameters) that is allowed **2.5 GB** and at most 30% of the card. A Q4 9B model is about 6 GB, so both fit in 10 GB. If less memory is free, the trainer drops to width 1536 or 768 instead of pushing Qwen out.

On `192.168.86.35`, next to the existing Ollama:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r trainer/requirements.txt
python trainer/server.py
```

Or, on that same machine: `docker compose --profile gpu up --build trainer`.

Then press **Train** in the lab. The Train tab should name the GPU and a parameter count in the millions. **GPU on** uses that policy. If the trainer is down, the page falls back to the small net and says so.

The GPU update runs every 64 decisions (PPO, 4 epochs, clip 0.2). Weights are written to `trainer/checkpoints/policy.pt` on the trainer machine. They are too large for `localStorage`.

### Qwen guide (optional, not backprop)

On a 10 GB card, keep the single Q4 copy you already have. Do not load a second model. Reachlab does not write to it.

1. On `192.168.86.35`, Ollama must already be serving `qwen3.5:9b` and must allow the browser: `OLLAMA_ORIGINS='*'`.
2. **Train** tab: URL `http://192.168.86.35:11434`, model `qwen3.5:9b`. Those are the defaults.
3. **Test guide**. It should answer with one action. A failure is usually CORS, or this page being on HTTPS while Ollama is HTTP.
4. **Guide on**. About every sixth decision the frozen model can replace the instinct's next action. Its prompt includes the best episodes seen so far.
5. **Download Modelfile** only if you want a separate tag with those notes. It does not modify `qwen3.5:9b`:

```bash
OLLAMA_HOST=http://192.168.86.35:11434 ollama create reachlab -f Reachlab.Modelfile
```

Leave the model field on `qwen3.5:9b` if you do not want that extra tag.

## Layout

- `src/game/gb.ts` — emulator, joypad, save state
- `src/game/field.ts` — RAM observation and rewards
- `src/game/instinct.ts` — small fallback PPO network
- `src/game/gpu.ts` — client for the 3080 trainer
- `trainer/server.py` — recurrent policy and PPO update
- `src/game/qwen.ts` — Ollama client and Modelfile
- `src/components/yellow-lab.tsx` — lab UI
- `src/routes/api/rom.ts` — serves `ROM_PATH` when the file is present
