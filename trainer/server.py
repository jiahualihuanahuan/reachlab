"""PPO trainer for the Reachlab policy. Caps CUDA use so qwen3.5:9b can stay resident."""

import os
import threading
from typing import Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import Policy, param_count

OBS = 32
ACT = 8
GAMMA = 0.995
LAMBDA = 0.95
CLIP = 0.2
ENTROPY = 0.02
LR = 3e-4
EPOCHS = 4
CHECKPOINT = os.environ.get(
    "CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "checkpoints", "policy.pt"),
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

lock = threading.Lock()
device = torch.device("cpu")
model: Policy
opt: torch.optim.Adam
updates = 0
vram_cap = 0
gpu_name = "cpu"
save_ok = True


class ActIn(BaseModel):
    obs: list[float] = Field(min_length=OBS, max_length=OBS)
    hidden: Optional[list[float]] = None


class UpdateIn(BaseModel):
    obs: list[list[float]]
    actions: list[int]
    logp: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]
    h0: Optional[list[float]] = None
    bootstrap: float


def estimate_bytes(width: int) -> int:
    # stem + 4 residual blocks + GRU + actor + critic, fp32 weights+grad+Adam.
    params = 17.2 * width * width
    return int(params * 16)


def choose_width(budget: int) -> int:
    for width in (2560, 1536, 768):
        if estimate_bytes(width) < int(budget * 0.85):
            return width
    return 768


def pick_device():
    global device, vram_cap, gpu_name
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        gpu_name = "cpu"
        vram_cap = 0
        return 768
    free, total = torch.cuda.mem_get_info()
    wanted = float(os.environ.get("VRAM_BUDGET_GB", "2.5")) * 1024**3
    # 30% of the card, 80% of what is free, and never the whole 2.5 GB request
    # if Qwen is already sitting on the rest.
    budget = int(min(wanted, free * 0.80, total * 0.30))
    budget = max(budget, 256 * 1024**2)
    torch.cuda.set_per_process_memory_fraction(min(0.95, budget / total))
    device = torch.device("cuda")
    vram_cap = budget
    gpu_name = torch.cuda.get_device_name(0)
    return choose_width(budget)


def hidden_tensor(values: Optional[list[float]], width: int) -> Optional[torch.Tensor]:
    if not values:
        return None
    if len(values) != width:
        return None
    t = torch.tensor(values, dtype=torch.float32, device=device).view(1, 1, width)
    return t


def save() -> None:
    if not save_ok:
        return
    path = CHECKPOINT
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "width": model.width,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "updates": updates,
        },
        path,
    )


def try_load(width: int) -> int:
    global save_ok
    if not os.path.exists(CHECKPOINT):
        return width
    blob = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    saved = int(blob.get("width", width))
    too_big = device.type == "cuda" and estimate_bytes(saved) >= int((vram_cap or 1) * 0.95)
    if too_big:
        save_ok = False
        print(f"[trainer] checkpoint width {saved} does not fit the VRAM cap, starting at {width} and not overwriting it")
        return width
    return saved


def apply_checkpoint() -> None:
    global updates
    if not os.path.exists(CHECKPOINT):
        return
    blob = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if int(blob.get("width", -1)) != model.width:
        return
    model.load_state_dict(blob["model"])
    opt.load_state_dict(blob["opt"])
    updates = int(blob.get("updates", 0))
    print(f"[trainer] restored {updates} updates from {CHECKPOINT}")


@app.get("/health")
def health():
    free = total = 0
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
    return {
        "ok": True,
        "device": device.type,
        "gpu": gpu_name,
        "params": param_count(model),
        "width": model.width,
        "updates": updates,
        "vram_cap_mb": int(vram_cap / (1024**2)),
        "free_mb": int(free / (1024**2)),
        "total_mb": int(total / (1024**2)),
    }


@app.post("/act")
def act(body: ActIn):
    obs = torch.tensor(body.obs, dtype=torch.float32, device=device).view(1, OBS)
    with lock, torch.no_grad():
        hidden = hidden_tensor(body.hidden, model.width)
        logits, value, hidden_out = model.forward_step(obs, hidden)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
    return {
        "action": int(action.item()),
        "logp": float(logp.item()),
        "value": float(value.item()),
        "entropy": float(entropy.item()),
        "hidden": hidden_out.detach().flatten().cpu().tolist(),
    }


@app.post("/update")
def update(body: UpdateIn):
    global updates
    n = len(body.actions)
    if n < 4 or len(body.obs) != n:
        return {"updates": updates, "entropy": 0.0}
    obs = torch.tensor(body.obs, dtype=torch.float32, device=device)
    actions = torch.tensor(body.actions, dtype=torch.long, device=device)
    old_logp = torch.tensor(body.logp, dtype=torch.float32, device=device)
    old_values = torch.tensor(body.values, dtype=torch.float32, device=device)
    rewards = torch.tensor(body.rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(body.dones, dtype=torch.float32, device=device)

    with torch.no_grad():
        adv = torch.zeros(n, device=device)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            nxt = body.bootstrap if t == n - 1 else float(old_values[t + 1])
            mask = 1.0 - float(dones[t])
            delta = float(rewards[t]) + GAMMA * nxt * mask - float(old_values[t])
            gae = delta + GAMMA * LAMBDA * mask * gae
            adv[t] = gae
        ret = adv + old_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    entropy = torch.zeros((), device=device)
    with lock:
        h0 = hidden_tensor(body.h0, model.width)
        for _ in range(EPOCHS):
            hidden = None if h0 is None else h0.detach()
            logits_all = []
            values_all = []
            for t in range(n):
                logits, value, hidden = model.forward_step(obs[t : t + 1], hidden)
                logits_all.append(logits)
                values_all.append(value)
                if float(dones[t]) > 0.5:
                    hidden = torch.zeros_like(hidden)
            logits = torch.cat(logits_all, dim=0)
            values = torch.cat(values_all, dim=0)
            logp = F.log_softmax(logits, dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
            policy = -torch.min(ratio * adv, clipped * adv).mean()
            value_loss = 0.5 * (values - ret).pow(2).mean()
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(-1).mean()
            loss = policy + value_loss - ENTROPY * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        updates += 1
        save()
    return {"updates": updates, "entropy": float(entropy.detach().cpu())}


@app.post("/reset")
def reset():
    return {"ok": True}


def main() -> None:
    global model, opt
    torch.backends.cuda.matmul.allow_tf32 = True
    width = pick_device()
    width = try_load(width)
    model = Policy(width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    apply_checkpoint()
    n = param_count(model)
    cap = f"{vram_cap / 1024**2:.0f} MB cap" if device.type == "cuda" else "cpu"
    print(f"[trainer] {gpu_name} width {width} params {n/1e6:.1f}M {cap}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8010")), log_level="info")


if __name__ == "__main__":
    main()
