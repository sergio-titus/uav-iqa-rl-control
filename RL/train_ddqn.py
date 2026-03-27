#!/usr/bin/env python3
import os, random, collections
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import rclpy
from uav_iqa_env import UAVIQAEnv

# ----------------- Hyperparams (stable DDQN) -----------------
MAX_EPISODES = 500
MAX_STEPS_PER_EP = 50

GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 64

BUFFER_SIZE = 100_000
MIN_REPLAY_SIZE = 2_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = MAX_EPISODES * MAX_STEPS_PER_EP  # ~25k

TAU = 0.005  # soft target update
GRAD_CLIP = 10.0

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(THIS_DIR, "runs", "DDQN_training")
CKPT_DIR = os.path.join(THIS_DIR, "checkpoints", "DDQN_training")
os.makedirs(CKPT_DIR, exist_ok=True)

BEST_PATH = os.path.join(CKPT_DIR, "best_policy.pth")
LAST_PATH = os.path.join(CKPT_DIR, "last_policy.pth")

Transition = collections.namedtuple("Transition", ("s", "a", "r", "ns", "d"))

class ReplayBuffer:
    def __init__(self, cap: int):
        self.cap = cap
        self.buf = []
        self.pos = 0

    def __len__(self):
        return len(self.buf)

    def add(self, s, a, r, ns, d):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.pos] = Transition(s, a, r, ns, d)
        self.pos = (self.pos + 1) % self.cap

    def sample(self, bs: int):
        batch = random.sample(self.buf, bs)
        batch = Transition(*zip(*batch))
        return batch

# ----------------- Dueling Q Network -----------------
class DuelingQ(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv   = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        f = self.feat(x)
        v = self.value(f)
        a = self.adv(f)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data * tau)

def speed_schedule(global_step: int, total_steps: int) -> float:
    """
    Start slow, then speed up as training progresses so the drone reaches chosen altitude faster.
    """
    frac = min(1.0, global_step / max(1, total_steps))
    # 0.6 m/s -> 2.4 m/s
    return 0.6 + 1.8 * frac

def train():
    rclpy.init()
    env = UAVIQAEnv()

    obs_dim = env.obs_dim
    n_actions = env.n_actions
    print(f"obs_dim={obs_dim}, n_actions={n_actions}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    q = DuelingQ(obs_dim, n_actions).to(device)
    tq = DuelingQ(obs_dim, n_actions).to(device)
    tq.load_state_dict(q.state_dict())
    tq.eval()

    opt = optim.Adam(q.parameters(), lr=LR)
    replay = ReplayBuffer(BUFFER_SIZE)

    writer = SummaryWriter(RUN_DIR)
    print(f"TensorBoard: tensorboard --logdir {os.path.join(THIS_DIR,'runs')} --port 6006")

    epsilon = EPS_START
    global_step = 0
    total_steps = MAX_EPISODES * MAX_STEPS_PER_EP

    best_mean = -1e9
    win = collections.deque(maxlen=30)

    state = env.reset().astype(np.float32)

    try:
        for ep in range(MAX_EPISODES):
            ep_ret = 0.0
            ep_abs_err80 = []
            ep_ge80 = 0
            ep_alt = []
            ep_wind = []

            for t in range(MAX_STEPS_PER_EP):
                global_step += 1

                # speed schedule (faster later)
                env.set_target_speed(speed_schedule(global_step, total_steps))

                # epsilon-greedy
                if random.random() < epsilon:
                    action = random.randrange(n_actions)
                else:
                    with torch.no_grad():
                        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = int(torch.argmax(q(s), dim=1).item())

                next_state, reward, done, info = env.step(action)
                next_state = next_state.astype(np.float32)

                replay.add(state, action, reward, next_state, float(done))
                state = next_state

                ep_ret += reward

                # ----- TB per step -----
                if info is None:
                    info = {}

                writer.add_scalar("step/reward", reward, global_step)
                writer.add_scalar("step/epsilon", epsilon, global_step)

                if "quality" in info:
                    writer.add_scalar("step/quality", float(info["quality"]), global_step)
                if "altitude" in info:
                    writer.add_scalar("step/altitude", float(info["altitude"]), global_step)
                if "wind" in info:
                    writer.add_scalar("step/wind_speed", float(info["wind"]), global_step)

                # episode stats
                if "abs_err_to_80" in info:
                    ep_abs_err80.append(float(info["abs_err_to_80"]))
                if info.get("is_ge80", False):
                    ep_ge80 += 1
                if "altitude" in info:
                    ep_alt.append(float(info["altitude"]))
                if "wind" in info:
                    ep_wind.append(float(info["wind"]))

                # ----- learn (stable DDQN) -----
                if len(replay) >= MIN_REPLAY_SIZE:
                    batch = replay.sample(BATCH_SIZE)

                    S  = torch.tensor(np.array(batch.s),  dtype=torch.float32, device=device)
                    A  = torch.tensor(batch.a, dtype=torch.int64, device=device).unsqueeze(1)
                    R  = torch.tensor(batch.r, dtype=torch.float32, device=device).unsqueeze(1)
                    NS = torch.tensor(np.array(batch.ns), dtype=torch.float32, device=device)
                    D  = torch.tensor(batch.d, dtype=torch.float32, device=device).unsqueeze(1)

                    q_sa = q(S).gather(1, A)

                    with torch.no_grad():
                        # DDQN: argmax from online net, value from target net
                        next_a = q(NS).argmax(dim=1, keepdim=True)
                        next_q = tq(NS).gather(1, next_a)
                        target = R + GAMMA * (1.0 - D) * next_q

                    loss = nn.functional.smooth_l1_loss(q_sa, target)  # Huber (stable)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), GRAD_CLIP)
                    opt.step()

                    soft_update(tq, q, TAU)

                    writer.add_scalar("train/loss", float(loss.item()), global_step)

                    # epsilon decay only after warmup
                    epsilon = max(
                        EPS_END,
                        EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS)
                    )

                if done:
                    break

            # ----- episode logging -----
            win.append(ep_ret)
            mean_ret = float(np.mean(win))
            std_ret  = float(np.std(win))

            writer.add_scalar("episode/return", ep_ret, ep)
            writer.add_scalar("episode/return_mean_win", mean_ret, ep)
            writer.add_scalar("episode/return_std_win", std_ret, ep)

            if ep_abs_err80:
                writer.add_scalar("episode/mean_abs_err_to_80", float(np.mean(ep_abs_err80)), ep)

            frac_ge80 = ep_ge80 / max(1, len(ep_alt))
            writer.add_scalar("episode/fraction_ge80", float(frac_ge80), ep)

            if ep_alt:
                writer.add_scalar("episode/mean_altitude", float(np.mean(ep_alt)), ep)

            # simple “policy vs wind” curve (binned)
            if ep_alt and ep_wind and len(ep_alt) == len(ep_wind):
                bins = [0.0, 0.5, 1.0, 2.0, 999.0]
                for bi in range(len(bins)-1):
                    lo, hi = bins[bi], bins[bi+1]
                    idx = [i for i,w in enumerate(ep_wind) if (w >= lo and w < hi)]
                    if idx:
                        writer.add_scalar(f"policy/alt_mean_wind_{lo}_{hi}", float(np.mean([ep_alt[i] for i in idx])), ep)

            print(f"Ep {ep:03d} | return={ep_ret:.2f} | mean30={mean_ret:.2f} ±{std_ret:.2f} | eps={epsilon:.3f}")

            # checkpoints
            torch.save({"episode": ep, "q": q.state_dict(), "opt": opt.state_dict(), "eps": epsilon, "step": global_step}, LAST_PATH)

            if len(win) == win.maxlen and mean_ret > best_mean:
                best_mean = mean_ret
                torch.save({"episode": ep, "q": q.state_dict(), "opt": opt.state_dict(),
                            "eps": epsilon, "step": global_step, "best_mean": best_mean}, BEST_PATH)
                print(f"[CKPT] best mean30={best_mean:.2f} -> {BEST_PATH}")

            # reset for next episode
            state = env.reset().astype(np.float32)

    finally:
        torch.save({"q": q.state_dict(), "opt": opt.state_dict(), "eps": epsilon, "step": global_step}, LAST_PATH)
        writer.close()
        env.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    train()
