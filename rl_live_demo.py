# rl_live_demo.py
import os
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Matplotlib only for the final plot window
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# -----------------------------
# 1) Minimal surrogate env (single-step, weight-conditioned)
# -----------------------------
class SurrogatePPAEnv(gym.Env):
    """
    Obs = [w_power, w_delay] (weights sampled per episode; sum=1)
    Act = [size, vdd] in [0,1]
    power = a * size^2 * vdd^2
    delay = b / (eps + size * vdd)
    Reward = - (w_pwr * power_norm + w_del * delay_norm)
    """
    #metadata = {"render_modes": []}

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space      = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)

        self.a, self.b, self.eps = 1.0, 1.0, 1e-3

        # Precompute min/max for normalization over [0,1]^2 grid
        grid = np.linspace(0, 1, 101)
        ss, vv = np.meshgrid(grid, grid)
        power = self.a * (ss**2) * (vv**2)
        delay = self.b / (self.eps + ss*vv)
        self.pwr_min, self.pwr_max = float(power.min()), float(power.max())
        self.dly_min, self.dly_max = float(delay.min()), float(delay.max())

        self.w = np.array([0.5, 0.5], dtype=np.float32)

    def _normalize(self, val, vmin, vmax):
        return float((val - vmin) / (vmax - vmin + 1e-12))

    def _eval(self, size, vdd):
        power = self.a * (size**2) * (vdd**2)
        delay = self.b / (self.eps + size*vdd)
        pz = self._normalize(power, self.pwr_min, self.pwr_max)
        dz = self._normalize(delay, self.dly_min, self.dly_max)
        return dict(power=power, delay=delay, p_norm=pz, d_norm=dz)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        w = self.rng.random(2)
        w = w / (w.sum() + 1e-12)
        self.w = w.astype(np.float32)
        return self.w.copy(), {}

    def step(self, action):
        size = float(np.clip(action[0], 0.0, 1.0))
        vdd  = float(np.clip(action[1], 0.0, 1.0))
        m = self._eval(size, vdd)
        reward = -(self.w[0]*m["p_norm"] + self.w[1]*m["d_norm"])
        info = {"size": size, "vdd": vdd, **m}
        return self.w.copy(), float(reward), True, False, info


# -----------------------------
# 2) Pareto utilities
# -----------------------------
def get_pareto(points):
    """
    points: list of dicts with keys 'power' and 'delay' (both to MINIMIZE).
    Returns nondominated subset, sorted by power ascending (and delay strictly improving).
    """
    P = sorted(points, key=lambda d: (d["power"], d["delay"]))
    front = []
    best_d = float("inf")
    for p in P:
        if p["delay"] < best_d:
            front.append(p)
            best_d = p["delay"]
    return front


# -----------------------------
# 3) Training thread worker
# -----------------------------
def train_and_log(run_dir, done_event, sweep_done_event, total_timesteps=5000):
    """
    - Configures SB3 CSV logger to run_dir/progress.csv
    - Pre-creates progress.csv (so GUI can read immediately)
    - Trains PPO on surrogate env
    - After training, runs weight sweep -> writes candidates.csv and pareto.csv
    - Signals GUI via Events
    """
    os.makedirs(run_dir, exist_ok=True)
    # Configure SB3 logger (CSV)
    new_logger = configure(run_dir, ["csv"])

    env = SurrogatePPAEnv()
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, learning_rate=3e-4, gamma=0.99, verbose=0)
    model.set_logger(new_logger)

    # PRE-CREATE progress.csv with one dummy row so GUI can open it immediately
    model.logger.record("time/iterations", 0)
    model.logger.record("time/total_timesteps", 0)
    model.logger.record("rollout/ep_len_mean", 1)
    model.logger.record("rollout/ep_rew_mean", 0.0)
    model.logger.dump(step=0)

    # Train
    model.learn(total_timesteps=total_timesteps)

    # Signal training finished
    done_event.set()

    # Sweep weights -> collect candidates, save CSVs
    candidates_csv = os.path.join(run_dir, "candidates.csv")
    pareto_csv     = os.path.join(run_dir, "pareto.csv")

    rows = []
    for t in np.linspace(0.0, 1.0, 61):
        w = np.array([t, 1.0 - t], dtype=np.float32)
        action, _ = model.predict(w, deterministic=True)
        env.w = w
        _, _, _, _, info = env.step(action)
        rows.append(dict(
            w_power=float(w[0]), w_delay=float(w[1]),
            size=float(info["size"]), vdd=float(info["vdd"]),
            power=float(info["power"]), delay=float(info["delay"])
        ))

    with open(candidates_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["w_power","w_delay","size","vdd","power","delay"])
        wtr.writeheader()
        for r in rows: wtr.writerow(r)

    pareto = get_pareto(rows)
    with open(pareto_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["power","delay","size","vdd","w_power","w_delay"])
        wtr.writeheader()
        for r in pareto: wtr.writerow(r)

    # Signal sweep + files ready
    sweep_done_event.set()


# -----------------------------
# 4) Final plot window (Tk Toplevel with matplotlib)
# -----------------------------
def open_plot_window(root, candidates_csv, pareto_csv):
    top = tk.Toplevel(root)
    top.title("RL PPA — Candidates & Pareto")

    fig = plt.Figure(figsize=(6.5, 4.3), dpi=120)
    ax = fig.add_subplot(111)

    def load_csv(path):
        out=[]
        if not os.path.exists(path): return out
        with open(path) as f:
            r = csv.DictReader(f)
            for d in r:
                out.append({k: float(v) for k,v in d.items()})
        return out

    cand = load_csv(candidates_csv)
    par  = load_csv(pareto_csv)

    if cand:
        xs = [d["power"] for d in cand]
        ys = [d["delay"] for d in cand]
        ax.scatter(xs, ys, s=18, alpha=0.5, label="Candidates")

    if par:
        px = [d["power"] for d in par]
        py = [d["delay"] for d in par]
        ax.plot(px, py, lw=2, label="Pareto")

    ax.set_xlabel("Power (lower is better)")
    ax.set_ylabel("Delay (lower is better)")
    ax.legend()
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw_idle()

    # Path labels
    ttk.Label(top, text=f"candidates: {candidates_csv}").pack(anchor="w", padx=8, pady=(6,0))
    ttk.Label(top, text=f"pareto:     {pareto_csv}").pack(anchor="w", padx=8, pady=(0,8))


# -----------------------------
# 5) Live stats GUI (main window)
# -----------------------------
FIELD_ORDER = [
    ("rollout/ep_len_mean",        "ep_len_mean"),
    ("rollout/ep_rew_mean",        "ep_rew_mean"),
    ("time/fps",                   "fps"),
    ("time/iterations",            "iterations"),
    ("time/time_elapsed",          "time_elapsed"),
    ("time/total_timesteps",       "total_timesteps"),
    ("train/approx_kl",            "approx_kl"),
    ("train/clip_fraction",        "clip_fraction"),
    ("train/clip_range",           "clip_range"),
    ("train/entropy_loss",         "entropy_loss"),
    ("train/explained_variance",   "explained_variance"),
    ("train/learning_rate",        "learning_rate"),
    ("train/loss",                 "loss"),
    ("train/n_updates",            "n_updates"),
    ("train/policy_gradient_loss", "policy_gradient_loss"),
    ("train/std",                  "std"),
    ("train/value_loss",           "value_loss"),
]

class LiveStatsApp(tk.Tk):
    def __init__(self, run_dir, done_event, sweep_done_event):
        super().__init__()
        self.title("RL Training — Live Stats")
        self.geometry("720x560")

        self.run_dir = run_dir
        self.progress_csv = os.path.join(run_dir, "progress.csv")
        self.candidates_csv = os.path.join(run_dir, "candidates.csv")
        self.pareto_csv = os.path.join(run_dir, "pareto.csv")

        self.done_event = done_event
        self.sweep_done_event = sweep_done_event
        self.plot_opened = False

        # Header
        hf = ttk.Frame(self)
        hf.pack(fill="x", padx=8, pady=8)
        ttk.Label(hf, text=f"Run directory: {run_dir}").pack(anchor="w")

        # Stats box
        self.txt = tk.Text(self, height=26, width=90, font=("Consolas", 11))
        self.txt.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # Footer
        self.footer = ttk.Label(self, text="Starting…")
        self.footer.pack(fill="x", padx=8, pady=(0,8))

        self.after(300, self._periodic_update)

    def _read_last_row(self, path):
        last = None
        try:
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                for idx, rec in enumerate(r):
                    last = rec
                    last["_row_idx"] = idx + 1
        except Exception:
            return None
        return last

    def _fmt_val(self, v):
        if v is None or v == "":
            return ""
        try:
            iv = int(float(v))
            if abs(float(v) - iv) < 1e-12:
                return str(iv)
        except Exception:
            pass
        try:
            return f"{float(v):.6g}"
        except Exception:
            return str(v)

    def _format_block(self, row):
        width = 41
        hr = "-" * width
        lines = [hr, "| rollout/                |             |"]
        for key, short in FIELD_ORDER:
            if key.startswith("rollout/"):
                lines.append(f"|    {short:<20} | {self._fmt_val(row.get(key)):<11} |")
        lines.append("| time/                   |             |")
        for key, short in FIELD_ORDER:
            if key.startswith("time/"):
                lines.append(f"|    {short:<20} | {self._fmt_val(row.get(key)):<11} |")
        lines.append("| train/                  |             |")
        for key, short in FIELD_ORDER:
            if key.startswith("train/"):
                lines.append(f"|    {short:<20} | {self._fmt_val(row.get(key)):<11} |")
        lines.append(hr)
        return "\n".join(lines)

    def _periodic_update(self):
        # Update live stats if progress.csv exists
        if os.path.exists(self.progress_csv):
            row = self._read_last_row(self.progress_csv)
            if row:
                block = self._format_block(row)
                self.txt.delete("1.0", tk.END)
                self.txt.insert(tk.END, block)
                self.footer.config(text=f"Rows: {row.get('_row_idx','?')}  |  {time.strftime('%H:%M:%S')}")
            else:
                self.footer.config(text=f"Waiting for logs… {time.strftime('%H:%M:%S')}")
        else:
            self.footer.config(text=f"Waiting for progress.csv… {time.strftime('%H:%M:%S')}")

        # If training finished and sweep done and plot not opened yet → open
        if self.done_event.is_set() and self.sweep_done_event.is_set() and not self.plot_opened:
            if os.path.exists(self.candidates_csv) and os.path.exists(self.pareto_csv):
                self.plot_opened = True
                open_plot_window(self, self.candidates_csv, self.pareto_csv)

        self.after(500, self._periodic_update)


# -----------------------------
# 6) Main: start GUI + training thread
# -----------------------------
if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", ts)

    done_event = threading.Event()
    sweep_done_event = threading.Event()

    # Start training in background thread
    t = threading.Thread(target=train_and_log, args=(run_dir, done_event, sweep_done_event, 10_000), daemon=True)
    t.start()

    # Start GUI (main thread)
    app = LiveStatsApp(run_dir, done_event, sweep_done_event)
    app.mainloop()
