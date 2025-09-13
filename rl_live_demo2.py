# rl_live_demo2.py (config-driven)
import os, csv, time, threading, argparse
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk

# Matplotlib only for the final plot window
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


# ---------- Config loading ----------
DEFAULT_CONFIG = {
    "training": {
        "policy": "MlpPolicy",
        "total_timesteps": 10000,
        "n_steps": 64,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "seed": None
    },
    "env": {
        "name": "SurrogatePPAEnv",
        "params": {"a": 1.0, "b": 1.0, "eps": 1e-3, "seed": None}
    },
    "run": {"root": "runs"},
    "log": {
        "outputs": ["csv"],
        "precreate_progress_row": True,
        "files": {"progress": "progress.csv", "candidates": "candidates.csv", "pareto": "pareto.csv"}
    },
    "sweep": {"points": 61, "grid": None},
    "gui": {"refresh_ms": 500},
    "plot": {"figure_dpi": 120, "figure_size": [6.5, 4.3]},
}

def load_config(path: str):
    if not os.path.exists(path):
        # auto-create default config for convenience
        try:
            import yaml
        except ImportError:
            yaml = None
        if yaml:
            with open(path, "w") as f:
                yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)
        else:
            # fallback: write minimal YAML by hand
            with open(path, "w") as f:
                f.write("training:\n  policy: MlpPolicy\n  total_timesteps: 12000\n  n_steps: 64\n  batch_size: 64\n  learning_rate: 0.0003\n  gamma: 0.99\n  seed: null\n")
                f.write("env:\n  name: SurrogatePPAEnv\n  params:\n    a: 1.0\n    b: 1.0\n    eps: 0.001\n    seed: null\n")
                f.write("run:\n  root: runs\n")
                f.write("log:\n  outputs: [csv]\n  precreate_progress_row: true\n  files:\n    progress: progress.csv\n    candidates: candidates.csv\n    pareto: pareto.csv\n")
                f.write("sweep:\n  points: 61\n  grid: null\n")
                f.write("gui:\n  refresh_ms: 500\n")
                f.write("plot:\n  figure_dpi: 120\n  figure_size: [6.5, 4.3]\n")
        print(f"[INFO] Wrote default config to {path}")
        return DEFAULT_CONFIG

    # load existing config
    try:
        import yaml
    except ImportError:
        raise RuntimeError("Please `pip install pyyaml` to use a YAML config file.")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # shallow-merge defaults for any missing keys
    def merge(d, default):
        for k, v in default.items():
            if k not in d:
                d[k] = v
            elif isinstance(v, dict) and isinstance(d[k], dict):
                merge(d[k], v)
        return d
    return merge(cfg, DEFAULT_CONFIG.copy())


import importlib, inspect

def make_env(cfg):
    env_cfg = cfg["env"]
    module_path = env_cfg["module"]
    class_name  = env_cfg["class"]
    params      = env_cfg.get("params", {}) or {}

    # import the module and get the class
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    # filter params to only what __init__ accepts
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self", "*", "**"}
    filtered = {k: v for k, v in params.items() if k in accepted}

    # helpful warning for mismatched keys
    dropped = [k for k in params.keys() if k not in accepted]
    if dropped:
        print(f"[WARN] Dropping unsupported env params for {class_name}: {dropped}")

    return cls(**filtered)


# ---------- Pareto ----------
def get_pareto(points):
    P = sorted(points, key=lambda d: (d["power"], d["delay"]))
    front, best_d = [], float("inf")
    for p in P:
        if p["delay"] < best_d:
            front.append(p)
            best_d = p["delay"]
    return front

# ---------- Training worker ----------
def train_and_log(cfg, run_dir, done_event, sweep_done_event):
    os.makedirs(run_dir, exist_ok=True)
    log_files = cfg["log"]["files"]
    progress_csv = os.path.join(run_dir, log_files["progress"])
    cand_csv     = os.path.join(run_dir, log_files["candidates"])
    pareto_csv   = os.path.join(run_dir, log_files["pareto"])

    # SB3 logger
    new_logger = configure(run_dir, cfg["log"]["outputs"])

    # Env + Model
    env = make_env(cfg)
    tr  = cfg["training"]
    model = PPO(
        tr["policy"], env,
        n_steps=tr["n_steps"],
        batch_size=tr["batch_size"],
        learning_rate=tr["learning_rate"],
        gamma=tr["gamma"],
        seed=tr["seed"],
        verbose=0
    )
    model.set_logger(new_logger)

    # Pre-create progress.csv (so GUI can start reading)
    if cfg["log"]["precreate_progress_row"]:
        model.logger.record("time/iterations", 0)
        model.logger.record("time/total_timesteps", 0)
        model.logger.record("rollout/ep_len_mean", 1)
        model.logger.record("rollout/ep_rew_mean", 0.0)
        model.logger.dump(step=0)

    # Train
    model.learn(total_timesteps=tr["total_timesteps"])
    done_event.set()

    # Sweep weights
    grid = cfg["sweep"]["grid"]
    if grid is None:
        n = int(cfg["sweep"]["points"])
        weights = np.linspace(0.0, 1.0, n, dtype=np.float32).tolist()
    else:
        weights = [float(x) for x in grid]

    rows = []
    for t in weights:
        w = np.array([t, 1.0 - t], dtype=np.float32)
        action, _ = model.predict(w, deterministic=True)
        # evaluate once in env
        env.w = w
        _, _, _, _, info = env.step(action)
        rows.append(dict(
            w_power=float(w[0]), w_delay=float(w[1]),
            size=float(info["size"]), vdd=float(info["vdd"]),
            power=float(info["power"]), delay=float(info["delay"])
        ))

    with open(cand_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["w_power","w_delay","size","vdd","power","delay"])
        wtr.writeheader(); wtr.writerows(rows)

    pareto = get_pareto(rows)
    with open(pareto_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["power","delay","size","vdd","w_power","w_delay"])
        wtr.writeheader(); wtr.writerows(pareto)

    sweep_done_event.set()

# ---------- Plot window ----------
def open_plot_window(root, cfg, candidates_csv, pareto_csv):
    top = tk.Toplevel(root)
    top.title("RL PPA — Candidates & Pareto")

    fig = plt.Figure(figsize=tuple(cfg["plot"]["figure_size"]), dpi=cfg["plot"]["figure_dpi"])
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

    ttk.Label(top, text=f"candidates: {candidates_csv}").pack(anchor="w", padx=8, pady=(6,0))
    ttk.Label(top, text=f"pareto:     {pareto_csv}").pack(anchor="w", padx=8, pady=(0,8))

# ---------- Live stats GUI ----------
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
    def __init__(self, cfg, run_dir, done_event, sweep_done_event):
        super().__init__()
        self.title("RL Training — Live Stats")
        self.geometry("720x560")
        self.cfg = cfg
        self.run_dir = run_dir
        log_files = cfg["log"]["files"]
        self.progress_csv = os.path.join(run_dir, log_files["progress"])
        self.candidates_csv = os.path.join(run_dir, log_files["candidates"])
        self.pareto_csv = os.path.join(run_dir, log_files["pareto"])
        self.done_event = done_event
        self.sweep_done_event = sweep_done_event
        self.plot_opened = False

        hf = ttk.Frame(self); hf.pack(fill="x", padx=8, pady=8)
        ttk.Label(hf, text=f"Run directory: {run_dir}").pack(anchor="w")

        self.txt = tk.Text(self, height=26, width=90, font=("Consolas", 11))
        self.txt.pack(fill="both", expand=True, padx=8, pady=(0,8))

        self.footer = ttk.Label(self, text="Starting…")
        self.footer.pack(fill="x", padx=8, pady=(0,8))

        self.after(self.cfg["gui"]["refresh_ms"], self._periodic_update)

    def _read_last_row(self, path):
        last = None
        try:
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                for idx, rec in enumerate(r):
                    last = rec; last["_row_idx"] = idx + 1
        except Exception:
            return None
        return last

    def _fmt(self, v):
        if v is None or v == "": return ""
        try:
            iv = int(float(v))
            if abs(float(v) - iv) < 1e-12: return str(iv)
        except: pass
        try: return f"{float(v):.6g}"
        except: return str(v)

    def _format_block(self, row):
        width = 41; hr = "-"*width
        lines = [hr, "| rollout/                |             |"]
        for k,s in FIELD_ORDER:
            if k.startswith("rollout/"): lines.append(f"|    {s:<20} | {self._fmt(row.get(k)):<11} |")
        lines.append("| time/                   |             |")
        for k,s in FIELD_ORDER:
            if k.startswith("time/"):    lines.append(f"|    {s:<20} | {self._fmt(row.get(k)):<11} |")
        lines.append("| train/                  |             |")
        for k,s in FIELD_ORDER:
            if k.startswith("train/"):   lines.append(f"|    {s:<20} | {self._fmt(row.get(k)):<11} |")
        lines.append(hr)
        return "\n".join(lines)

    def _periodic_update(self):
        if os.path.exists(self.progress_csv):
            row = self._read_last_row(self.progress_csv)
            if row:
                self.txt.delete("1.0", tk.END)
                self.txt.insert(tk.END, self._format_block(row))
                self.footer.config(text=f"Rows: {row.get('_row_idx','?')}  |  {time.strftime('%H:%M:%S')}")
            else:
                self.footer.config(text=f"Waiting for logs… {time.strftime('%H:%M:%S')}")
        else:
            self.footer.config(text=f"Waiting for progress.csv… {time.strftime('%H:%M:%S')}")

        if self.done_event.is_set() and self.sweep_done_event.is_set() and not self.plot_opened:
            if os.path.exists(self.candidates_csv) and os.path.exists(self.pareto_csv):
                self.plot_opened = True
                open_plot_window(self, self.cfg, self.candidates_csv, self.pareto_csv)

        self.after(self.cfg["gui"]["refresh_ms"], self._periodic_update)

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["run"]["root"], ts)

    done_event = threading.Event()
    sweep_done_event = threading.Event()

    t = threading.Thread(target=train_and_log, args=(cfg, run_dir, done_event, sweep_done_event), daemon=True)
    t.start()

    app = LiveStatsApp(cfg, run_dir, done_event, sweep_done_event)
    app.mainloop()
