#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/uav_results")
OUT  = os.path.expanduser("~/uav_results/analysis")
os.makedirs(OUT, exist_ok=True)

# -----------------------
# Load data
# -----------------------
def load_runs(method: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(BASE, method, "run*"))):
        run = os.path.basename(run_dir)

        for jf in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
            try:
                d = json.load(open(jf))
            except Exception:
                continue

            # prefer recorded unix time if present; else file mtime
            t = d.get("t_unix", os.path.getmtime(jf))
            q = d.get("quality", np.nan)
            alt = d.get("telemetry", {}).get("rel_alt_m", np.nan)

            # robust cast
            try: t = float(t)
            except Exception: t = float(os.path.getmtime(jf))
            try: q = float(q)
            except Exception: q = np.nan
            try: alt = float(alt)
            except Exception: alt = np.nan

            rows.append({
                "method": method,
                "run": run,
                "t": t,
                "quality": q,
                "altitude": alt,
                "json": jf
            })
    return pd.DataFrame(rows)

df = pd.concat([load_runs("fixed_alt"), load_runs("ddqn")], ignore_index=True)

# drop unusable rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["t", "quality", "altitude"])
df = df.sort_values(["run", "method", "t"]).reset_index(drop=True)

df.to_csv(os.path.join(OUT, "all_data.csv"), index=False)

def rel_time(series_t: pd.Series) -> np.ndarray:
    arr = series_t.to_numpy(dtype=float)
    return arr - np.nanmin(arr)

# -----------------------
# Paired-by-run plots
# -----------------------
runs = sorted(df["run"].unique())

def _ensure_axes(n):
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2*n), sharex=False)
    if n == 1:
        axes = [axes]
    return fig, axes

# FIG 1: Quality vs Time (paired by run)
fig, axes = _ensure_axes(len(runs))
for ax, r in zip(axes, runs):
    for m in ["fixed_alt", "ddqn"]:
        d = df[(df["run"] == r) & (df["method"] == m)]
        if len(d) == 0:
            continue
        x = rel_time(d["t"])
        y = d["quality"].to_numpy(dtype=float)
        ax.plot(x, y, label=m)
    ax.set_title(f"{r}: Quality vs Time")
    ax.set_ylabel("Quality (%)")
    ax.legend()
axes[-1].set_xlabel("Time since start (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "quality_vs_time_by_run.png"), dpi=200)
plt.close()

# FIG 2: Altitude vs Time (paired by run)
fig, axes = _ensure_axes(len(runs))
for ax, r in zip(axes, runs):
    for m in ["fixed_alt", "ddqn"]:
        d = df[(df["run"] == r) & (df["method"] == m)]
        if len(d) == 0:
            continue
        x = rel_time(d["t"])
        y = d["altitude"].to_numpy(dtype=float)
        ax.plot(x, y, label=m)
    ax.set_title(f"{r}: Altitude vs Time")
    ax.set_ylabel("Altitude (m)")
    ax.legend()
axes[-1].set_xlabel("Time since start (s)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "altitude_vs_time_by_run.png"), dpi=200)
plt.close()

# -----------------------
# High-quality image yield (Q >= 85)
# -----------------------
TH = 85.0
rows = []
for r in runs:
    for m in ["fixed_alt", "ddqn"]:
        d = df[(df["run"] == r) & (df["method"] == m)]
        if len(d) == 0:
            continue
        n_total = len(d)
        n_q85 = int(np.sum(d["quality"].to_numpy(dtype=float) >= TH))
        pct_q85 = 100.0 * (n_q85 / max(n_total, 1))
        # mission duration (s): last-first
        tarr = d["t"].to_numpy(dtype=float)
        duration = float(np.nanmax(tarr) - np.nanmin(tarr)) if len(tarr) > 1 else 0.0
        # efficiency: HQ images per minute
        hq_per_min = (n_q85 / (duration/60.0)) if duration > 1e-6 else np.nan

        rows.append({
            "run": r, "method": m,
            "n_total": n_total,
            "n_q85": n_q85,
            "pct_q85": pct_q85,
            "duration_s": duration,
            "hq_images_per_min": hq_per_min
        })

hq = pd.DataFrame(rows)
hq.to_csv(os.path.join(OUT, "high_quality_counts.csv"), index=False)

# Bar plot: number of images >=85 per run (paired)
plt.figure(figsize=(8,4.5))
x = np.arange(len(runs))
w = 0.35

fixed = []
ddqn = []
for r in runs:
    fixed.append(hq[(hq.run==r) & (hq.method=="fixed_alt")]["n_q85"].values[0])
    ddqn.append(hq[(hq.run==r) & (hq.method=="ddqn")]["n_q85"].values[0])

plt.bar(x - w/2, fixed, w, label="fixed_alt")
plt.bar(x + w/2, ddqn,  w, label="ddqn")

plt.xticks(x, runs)
plt.ylabel(f"Number of images (Q ≥ {TH:.0f}%)")
plt.title("High-quality image yield per mission (paired by run)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "high_quality_images_by_run.png"), dpi=200)
plt.close()

# Optional: efficiency plot (HQ images per minute)
plt.figure(figsize=(8,4.5))
fixed_eff = []
ddqn_eff = []
for r in runs:
    fixed_eff.append(hq[(hq.run==r) & (hq.method=="fixed_alt")]["hq_images_per_min"].values[0])
    ddqn_eff.append(hq[(hq.run==r) & (hq.method=="ddqn")]["hq_images_per_min"].values[0])

plt.bar(x - w/2, fixed_eff, w, label="fixed_alt")
plt.bar(x + w/2, ddqn_eff,  w, label="ddqn")
plt.xticks(x, runs)
plt.ylabel(f"HQ images/min (Q ≥ {TH:.0f}%)")
plt.title("High-quality acquisition efficiency (paired by run)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "hq_images_per_min_by_run.png"), dpi=200)
plt.close()

print("✅ Paired-by-run analysis complete.")
print("Saved to:", OUT)
print("Files:")
for f in [
    "all_data.csv",
    "quality_vs_time_by_run.png",
    "altitude_vs_time_by_run.png",
    "high_quality_counts.csv",
    "high_quality_images_by_run.png",
    "hq_images_per_min_by_run.png",
]:
    print(" -", f)
