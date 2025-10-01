#final_plot.py
import os, csv, tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
