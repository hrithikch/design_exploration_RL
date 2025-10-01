#live_stats.py

import os, csv, time, tkinter as tk
from tkinter import ttk
from .final_plot import open_plot_window  # circular import avoidance: only function

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
        files = cfg["log"]["files"]
        self.progress_csv = os.path.join(run_dir, files["progress"])
        self.candidates_csv = os.path.join(run_dir, files["candidates"])
        self.pareto_csv = os.path.join(run_dir, files["pareto"])
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
