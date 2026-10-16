# Reachlab

A browser field lab for a Game Boy Yellow cartridge. It boots the game in [binjgb](https://github.com/binji/binjgb) (MIT), scores curiosity from RAM, and trains a tiny policy in the page. A frozen Qwen 3.5 9B in Ollama can guide play. The 9B weights are not updated.

The cartridge is not in this repo. Use a ROM you legally own. Do not commit it.

## Docker Compose

Needs Docker Compose v2 and, for the guide, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so the RTX 3080 is visible (`nvidia-smi` works inside Docker).

```bash
mkdir -p roms
cp "/path/to/Pokemon - Yellow Version.gbc" roms/yellow.gbc
docker compose up --build
```

Open http://localhost:8080.

- `web` serves the lab on port 8080 and reads `/roms/yellow.gbc`.
- `ollama` serves `http://127.0.0.1:11434` with `OLLAMA_ORIGINS=*` so the page can call it. The model lives in the `ollama` volume.
- `ollama-pull` runs once and pulls `qwen3.5:9b` (about 6 GB, Q4). The lab works before that finishes. Training does not need it. The guide does.

If `docker compose up` fails on `gpus: all`, the GPU runtime is missing. Install the toolkit and restart Docker, or delete the `gpus: all` line to run Ollama on CPU (too slow for 9B).

No database. Policy weights stay in the browser (`localStorage` key `reachlab-yellow-v1`), not in the container.

### Without Docker

Node 22 (or 20.19+):

```bash
npm install
npm run dev
```

Then open http://localhost:8080 and use **Choose file**. The server ROM path is `ROM_PATH` (default inside this sandbox only).

Ollama on the host, if you want the guide:

```bash
OLLAMA_ORIGINS='*' ollama serve
ollama pull qwen3.5:9b
```

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

### Qwen guide (optional, not backprop)

On a 10 GB card, keep a single Q4 copy. Do not load a second model.

1. Wait until `ollama-pull` has finished (`docker compose logs ollama-pull`).
2. **Train** tab: URL `http://127.0.0.1:11434`, model `qwen3.5:9b`.
3. **Test guide**. It should answer with one action. If it fails, the error is shown as-is (usually CORS or the model still pulling).
4. **Guide on**. About every sixth decision the frozen model can replace the instinct's next action. Its prompt includes the best episodes seen so far.
5. **Download Modelfile**. Then, from the directory that contains `Reachlab.Modelfile`:

```bash
docker compose cp Reachlab.Modelfile ollama:/tmp/Reachlab.Modelfile
docker compose exec ollama ollama create reachlab -f /tmp/Reachlab.Modelfile
```

If Ollama is running on the host instead of in Compose: `ollama create reachlab -f Reachlab.Modelfile`.

Point the model field at `reachlab` afterward. That bakes the elite notes into the system prompt. It still does not train the 9B weights.

## Layout

- `src/game/gb.ts` — emulator, joypad, save state
- `src/game/field.ts` — RAM observation and rewards
- `src/game/instinct.ts` — PPO network
- `src/game/qwen.ts` — Ollama client and Modelfile
- `src/components/yellow-lab.tsx` — lab UI
- `src/routes/api/rom.ts` — serves `ROM_PATH` when the file is present
