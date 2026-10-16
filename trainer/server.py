"""PPO trainer for the Reachlab policy. Caps CUDA use so qwen3.5:9b can stay resident.

The HTTP server is the standard library, so Python 3.14 does not need Pydantic.
The only install is PyTorch.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import torch
import torch.nn.functional as F

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

lock = threading.Lock()
device = torch.device("cpu")
model: Policy
opt: torch.optim.Adam
updates = 0
vram_cap = 0
gpu_name = "cpu"
save_ok = True


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
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, 1, width)


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


def act(body: dict):
    obs_list = body.get("obs")
    if not isinstance(obs_list, list) or len(obs_list) != OBS:
        raise ValueError("obs must be 32 numbers")
    obs = torch.tensor(obs_list, dtype=torch.float32, device=device).view(1, OBS)
    with lock, torch.no_grad():
        hidden = hidden_tensor(body.get("hidden"), model.width)
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


def update(body: dict):
    global updates
    actions_list = body.get("actions") or []
    obs_list = body.get("obs") or []
    n = len(actions_list)
    if n < 4 or len(obs_list) != n:
        return {"updates": updates, "entropy": 0.0}
    obs = torch.tensor(obs_list, dtype=torch.float32, device=device)
    actions = torch.tensor(actions_list, dtype=torch.long, device=device)
    old_logp = torch.tensor(body["logp"], dtype=torch.float32, device=device)
    old_values = torch.tensor(body["values"], dtype=torch.float32, device=device)
    rewards = torch.tensor(body["rewards"], dtype=torch.float32, device=device)
    dones = torch.tensor(body["dones"], dtype=torch.float32, device=device)
    bootstrap = float(body.get("bootstrap", 0.0))

    with torch.no_grad():
        adv = torch.zeros(n, device=device)
        gae = 0.0
        for t in range(n - 1, -1, -1):
            nxt = bootstrap if t == n - 1 else float(old_values[t + 1])
            mask = 1.0 - float(dones[t])
            delta = float(rewards[t]) + GAMMA * nxt * mask - float(old_values[t])
            gae = delta + GAMMA * LAMBDA * mask * gae
            adv[t] = gae
        ret = adv + old_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    entropy = torch.zeros((), device=device)
    with lock:
        h0 = hidden_tensor(body.get("h0"), model.width)
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


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/health":
            self._send(200, health())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            return
        path = self.path.split("?", 1)[0]
        try:
            if path == "/act":
                self._send(200, act(body))
            elif path == "/update":
                self._send(200, update(body))
            elif path == "/reset":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print("[trainer]", self.address_string(), fmt % args)


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
    port = int(os.environ.get("PORT", "8010"))
    print(f"[trainer] {gpu_name} width {width} params {n / 1e6:.1f}M {cap}")
    print(f"[trainer] listening on 0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
