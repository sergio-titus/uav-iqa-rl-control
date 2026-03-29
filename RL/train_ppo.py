#!/usr/bin/env python3
import os, time, random, collections
from pathlib import Path
import yaml
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import rclpy
from uav_iqa_env import UAVIQAEnv

def load_yaml_config(filename: str):
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / filename

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ----------------- Config loading -----------------
ppo_cfg = load_yaml_config("ppo.yaml")["ppo"]
training_cfg = load_yaml_config("training.yaml")
reward_cfg = load_yaml_config("reward.yaml")["reward"]

MAX_EPISODES = ppo_cfg["training"]["episodes"]
MAX_STEPS_PER_EP = ppo_cfg["training"]["steps_per_episode"]

GAMMA = ppo_cfg["gamma"]
GAE_LAMBDA = ppo_cfg["gae_lambda"]

LR = ppo_cfg["learning_rate"]
ENT_COEF = ppo_cfg["loss"]["entropy_coef"]
VF_COEF = ppo_cfg["loss"]["value_coef"]
MAX_GRAD_NORM = ppo_cfg["optimization"]["max_grad_norm"]

CLIP_EPS = ppo_cfg["clipping"]["clip_eps"]
TARGET_KL = ppo_cfg["clipping"]["target_kl"]

ROLLOUT_STEPS = ppo_cfg["rollout"]["steps"]
UPDATE_EPOCHS = ppo_cfg["rollout"]["update_epochs"]
MINIBATCH_SIZE = ppo_cfg["rollout"]["minibatch_size"]

STEP_PENALTY = reward_cfg["shaping"]["step_penalty"]
Q_BASE_CENTER = reward_cfg["shaping"]["q_base_center"]
Q_BASE_SCALE = reward_cfg["shaping"]["q_base_scale"]
BONUS_Q80 = reward_cfg["shaping"]["bonus_q80"]
BONUS_Q90 = reward_cfg["shaping"]["bonus_q90"]
ALT_OPT = reward_cfg["shaping"]["alt_opt"]
ALT_PENALTY_K = reward_cfg["shaping"]["alt_penalty_k"]
TARGET_Q = reward_cfg["target_quality"]
Q_THR1 = reward_cfg["shaping"]["q_threshold_1"]
Q_THR2 = reward_cfg["shaping"]["q_threshold_2"]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = training_cfg["logging"]["tensorboard"]["ppo_log_dir"]
RUN_ROOT = os.path.join(THIS_DIR, LOG_DIR)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUN_ROOT, RUN_ID)

CKPT_DIR = os.path.join(THIS_DIR, training_cfg["logging"]["checkpoints"]["ppo_ckpt_dir"])
os.makedirs(CKPT_DIR, exist_ok=True)
BEST_CKPT_PATH = os.path.join(CKPT_DIR, "best_policy.pth")
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last_policy.pth")


# ----------------- Utils -----------------
def shape_reward(env_reward: float, info: Optional[Dict[str, Any]]) -> float:
    if info is None:
        info = {}
    q = info.get("quality", None)
    z = info.get("altitude", None)

    base = 0.0
    bonus = 0.0
    alt_pen = 0.0

    if q is not None:
        base = (float(q) - Q_BASE_CENTER) / Q_BASE_SCALE
        if q >= Q_THR1:
              bonus += BONUS_Q80

        if q >= Q_THR2:
              bonus += BONUS_Q90

    if z is not None:
        z = float(z)
        diff = max(0.0, z - ALT_OPT)
        alt_pen = -ALT_PENALTY_K * diff * diff

    return float(env_reward) + base + bonus + STEP_PENALTY + alt_pen


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # 1 - Var[y-ypred]/Var[y]
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return np.nan
    return float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))


# ----------------- PPO Network -----------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()

        hidden1 = ppo_cfg.get("network", {}).get("hidden1", 256)
        hidden2 = ppo_cfg.get("network", {}).get("hidden2", 256)

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden2, n_actions)
        self.v = nn.Linear(hidden2, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, dist.entropy()


# ----------------- Rollout Buffer -----------------
@dataclass
class Rollout:
    obs: np.ndarray
    act: np.ndarray
    logp: np.ndarray
    rew: np.ndarray
    done: np.ndarray
    val: np.ndarray

def to_torch(x, device, dtype=torch.float32):
    return torch.tensor(x, device=device, dtype=dtype)


# ----------------- Train -----------------
def train():
    rclpy.init()
    env = UAVIQAEnv(env_config_path="env.yaml", reward_config_path="reward.yaml")

    obs_dim = env.obs_dim
    n_actions = env.n_actions
    print(f"obs_dim={obs_dim}, n_actions={n_actions}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    net = ActorCritic(obs_dim, n_actions).to(device)
    opt = optim.Adam(net.parameters(), lr=LR)

    writer = SummaryWriter(log_dir=RUN_DIR)
    print("TensorBoard:", f"tensorboard --logdir {os.path.join(THIS_DIR,'runs')} --port 6006")
    print("[TB] run:", RUN_DIR)

    # stats
    returns_window = collections.deque(maxlen=30)
    best_mean30 = -1e9
    global_step = 0

    # reset
    obs = env.reset().astype(np.float32)

    # rollout storage lists (dynamic, then pack)
    buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []

    ep = 0
    ep_step = 0
    ep_return = 0.0

    # per-episode stats for graphs
    ep_q_sum = 0.0
    ep_q_count = 0
    ep_ge80 = 0
    ep_alt_sum = 0.0
    ep_alt_count = 0

    try:
        while ep < MAX_EPISODES:
            global_step += 1
            ep_step += 1

            # policy step
            with torch.no_grad():
                obs_t = to_torch(obs[None, :], device)
                a_t, logp_t, v_t, ent_t = net.act(obs_t)
                action = int(a_t.item())
                logp = float(logp_t.item())
                value = float(v_t.item())

            next_obs, env_reward, done, info = env.step(action)
            next_obs = next_obs.astype(np.float32)

            shaped = shape_reward(env_reward, info)

            # logs from info
            if info is None:
                info = {}
            q = info.get("quality", None)
            z = info.get("altitude", None)
            wind = info.get("wind", None)
            sun = info.get("sun", None)

            writer.add_scalar("step/env_reward", float(env_reward), global_step)
            writer.add_scalar("step/shaped_reward", float(shaped), global_step)

            if q is not None:
                qf = float(q)
                writer.add_scalar("step/quality", qf, global_step)
                ep_q_sum += qf
                ep_q_count += 1
                if qf >= 80.0:
                    ep_ge80 += 1

            if z is not None:
                zf = float(z)
                writer.add_scalar("step/altitude", zf, global_step)
                ep_alt_sum += zf
                ep_alt_count += 1

            if wind is not None:
                writer.add_scalar("step/wind", float(wind), global_step)
            if sun is not None:
                writer.add_scalar("step/sun", float(sun), global_step)

            # store rollout
            buf_obs.append(obs.copy())
            buf_act.append(action)
            buf_logp.append(logp)
            buf_rew.append(shaped)
            buf_done.append(float(done))
            buf_val.append(value)

            # advance
            obs = next_obs
            ep_return += shaped

            # episode ended?
            if done or ep_step >= MAX_STEPS_PER_EP:
                # episode logs
                returns_window.append(ep_return)
                mean30 = float(np.mean(returns_window))
                std30 = float(np.std(returns_window))

                writer.add_scalar("episode/return", ep_return, ep)
                writer.add_scalar("episode/mean30_return", mean30, ep)
                writer.add_scalar("episode/std30_return", std30, ep)

                if ep_q_count > 0:
                    writer.add_scalar("quality/mean_episode", ep_q_sum / ep_q_count, ep)
                    writer.add_scalar("quality/fraction_ge80", ep_ge80 / ep_q_count, ep)

                if ep_alt_count > 0:
                    writer.add_scalar("episode/mean_altitude", ep_alt_sum / ep_alt_count, ep)

                print(f"Ep {ep:03d} | return={ep_return:.2f} | mean30={mean30:.2f} ±{std30:.2f}")

                # checkpoint best by mean30 (once window has enough)
                if len(returns_window) == returns_window.maxlen and mean30 > best_mean30:
                    best_mean30 = mean30
                    torch.save({
                        "episode": ep,
                        "global_step": global_step,
                        "best_mean30": best_mean30,
                        "policy_state_dict": net.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        # keep ALSO a simple key for compatibility
                        "pi_v_state_dict": net.state_dict(),
                    }, BEST_CKPT_PATH)
                    print(f"[CKPT] best mean30={best_mean30:.2f} -> {BEST_CKPT_PATH}")

                # reset episode accumulators
                ep += 1
                ep_step = 0
                ep_return = 0.0
                ep_q_sum = 0.0
                ep_q_count = 0
                ep_ge80 = 0
                ep_alt_sum = 0.0
                ep_alt_count = 0

                # reset env
                obs = env.reset().astype(np.float32)

            # rollout update condition
            if len(buf_obs) >= ROLLOUT_STEPS:
                # bootstrap value for last obs
                with torch.no_grad():
                    obs_t = to_torch(obs[None, :], device)
                    _, last_v = net.forward(obs_t)
                    last_v = float(last_v.item())

                # pack arrays
                obs_arr = np.array(buf_obs, dtype=np.float32)
                act_arr = np.array(buf_act, dtype=np.int64)
                logp_arr = np.array(buf_logp, dtype=np.float32)
                rew_arr = np.array(buf_rew, dtype=np.float32)
                done_arr = np.array(buf_done, dtype=np.float32)
                val_arr = np.array(buf_val, dtype=np.float32)

                # compute GAE advantages + returns
                adv = np.zeros_like(rew_arr, dtype=np.float32)
                last_gae = 0.0
                for t in reversed(range(len(rew_arr))):
                    next_nonterminal = 1.0 - done_arr[t]
                    next_value = last_v if t == len(rew_arr) - 1 else val_arr[t + 1]
                    delta = rew_arr[t] + GAMMA * next_value * next_nonterminal - val_arr[t]
                    last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
                    adv[t] = last_gae
                ret = adv + val_arr

                # normalize advantages
                adv_mean, adv_std = adv.mean(), adv.std() + 1e-8
                adv = (adv - adv_mean) / adv_std

                # PPO update
                n = len(obs_arr)
                inds = np.arange(n)

                # some metrics
                clipfracs = []
                approx_kls = []
                entropies = []
                pg_losses = []
                v_losses = []

                for epoch in range(UPDATE_EPOCHS):
                    np.random.shuffle(inds)
                    for start in range(0, n, MINIBATCH_SIZE):
                        mb = inds[start:start + MINIBATCH_SIZE]

                        mb_obs = to_torch(obs_arr[mb], device)
                        mb_act = to_torch(act_arr[mb], device, dtype=torch.long)
                        mb_old_logp = to_torch(logp_arr[mb], device)
                        mb_adv = to_torch(adv[mb], device)
                        mb_ret = to_torch(ret[mb], device)
                        mb_old_val = to_torch(val_arr[mb], device)

                        logits, value = net.forward(mb_obs)
                        dist = Categorical(logits=logits)
                        new_logp = dist.log_prob(mb_act)
                        entropy = dist.entropy().mean()

                        ratio = (new_logp - mb_old_logp).exp()
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                        pg_loss = -torch.min(surr1, surr2).mean()

                        # value loss with optional clip
                        v_clipped = mb_old_val + torch.clamp(value - mb_old_val, -CLIP_EPS, CLIP_EPS)
                        v_loss1 = (value - mb_ret).pow(2)
                        v_loss2 = (v_clipped - mb_ret).pow(2)
                        v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                        loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy

                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                        opt.step()

                        # metrics
                        approx_kl = 0.5 * (mb_old_logp - new_logp).pow(2).mean().item()
                        clipfrac = (torch.abs(ratio - 1.0) > CLIP_EPS).float().mean().item()

                        approx_kls.append(approx_kl)
                        clipfracs.append(clipfrac)
                        entropies.append(entropy.item())
                        pg_losses.append(pg_loss.item())
                        v_losses.append(v_loss.item())

                    # early stop if KL too high
                    if TARGET_KL is not None and np.mean(approx_kls) > TARGET_KL:
                        break

                # log training curves (THIS is the "train" tab you want)
                writer.add_scalar("train/policy_loss", float(np.mean(pg_losses)), global_step)
                writer.add_scalar("train/value_loss", float(np.mean(v_losses)), global_step)
                writer.add_scalar("train/entropy", float(np.mean(entropies)), global_step)
                writer.add_scalar("train/approx_kl", float(np.mean(approx_kls)), global_step)
                writer.add_scalar("train/clipfrac", float(np.mean(clipfracs)), global_step)

                # explained variance (value fit quality)
                with torch.no_grad():
                    v_pred = val_arr
                ev = explained_variance(v_pred, ret)
                writer.add_scalar("train/explained_variance", ev, global_step)

                # save last
                torch.save({
                    "episode": ep,
                    "global_step": global_step,
                    "policy_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "run_dir": RUN_DIR,
                }, LAST_CKPT_PATH)

                # clear rollout buffers
                buf_obs.clear(); buf_act.clear(); buf_logp.clear()
                buf_rew.clear(); buf_done.clear(); buf_val.clear()

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # save last
        torch.save({
            "episode": ep,
            "global_step": global_step,
            "policy_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "run_dir": RUN_DIR,
        }, LAST_CKPT_PATH)
        print("[CKPT] last ->", LAST_CKPT_PATH)

        env.destroy_node()
        rclpy.shutdown()
        writer.close()


if __name__ == "__main__":
    train()
