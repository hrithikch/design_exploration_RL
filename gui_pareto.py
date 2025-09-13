# gui_pareto.py
# This can open an existing candidates.csv and pareto.csv
# instead of waiting for a training run
import tkinter as tk
from tkinter import filedialog
import csv, os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

REFRESH_MS = 2000

def load_csv(path):
    if not os.path.exists(path): return []
    rows=[]
    try:
        with open(path) as f:
            r=csv.DictReader(f)
            for d in r:
                rows.append({k: float(v) for k,v in d.items()})
    except Exception:
        pass  # file might be mid-write; ignore this cycle
    return rows

def get_pareto(rows):
    P = sorted(rows, key=lambda d: (d["power"], d["delay"]))
    out=[]; best=float("inf")
    for r in P:
        if r["delay"] < best:
            out.append(r); best=r["delay"]
    return out

class ParetoGUI(tk.Tk):
    def __init__(self, cand_path, pareto_path=None):
        super().__init__()
        self.title("RL PPA Pareto — Live")
        self.cand_path = cand_path
        self.pareto_path = pareto_path

        top = tk.Frame(self); top.pack(fill="x")
        tk.Label(top, text=f"Candidates: {cand_path}").pack(anchor="w")
        if pareto_path:
            tk.Label(top, text=f"Pareto:     {pareto_path}").pack(anchor="w")

        self.fig = plt.Figure(figsize=(6,4), dpi=120)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        btn_frame = tk.Frame(self); btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="Open candidates...", command=self.pick_cand).pack(side="left")
        tk.Button(btn_frame, text="Open pareto...", command=self.pick_par).pack(side="left")
        self.after(500, self.refresh)

    def pick_cand(self):
        p = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if p: self.cand_path = p

    def pick_par(self):
        p = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if p: self.pareto_path = p

    def refresh(self):
        cand = load_csv(self.cand_path) if self.cand_path else []
        par = load_csv(self.pareto_path) if self.pareto_path else []
        if cand:
            self.ax.clear()
            xs = [d["power"] for d in cand]
            ys = [d["delay"] for d in cand]
            self.ax.scatter(xs, ys, s=18, alpha=0.5, label="Candidates")
            if par:
                px = [d["power"] for d in par]
                py = [d["delay"] for d in par]
                self.ax.plot(px, py, lw=2, label="Pareto")
            else:
                # compute on the fly if no separate pareto file
                pareto = get_pareto(cand)
                px = [d["power"] for d in pareto]
                py = [d["delay"] for d in pareto]
                self.ax.plot(px, py, lw=2, label="Pareto (computed)")
            self.ax.set_xlabel("Power (lower better)")
            self.ax.set_ylabel("Delay (lower better)")
            self.ax.legend()
            self.fig.tight_layout()
            self.canvas.draw_idle()
        self.after(REFRESH_MS, self.refresh)

if __name__ == "__main__":
    import sys
    cand = sys.argv[1] if len(sys.argv)>1 else ""
    par  = sys.argv[2] if len(sys.argv)>2 else ""
    app = ParetoGUI(cand, par)
    app.mainloop()
