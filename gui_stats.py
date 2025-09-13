# gui_stats.py
import os, csv, time, tkinter as tk
from tkinter import filedialog, messagebox

REFRESH_MS = 1000

# Order & pretty names to emulate your block
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

class StatsGUI(tk.Tk):
    def __init__(self, csv_path=""):
        super().__init__()
        self.title("RL Training Stats (Stable-Baselines3 progress.csv)")
        self.geometry("680x520")
        self.csv_path = csv_path

        top = tk.Frame(self); top.pack(fill="x", padx=8, pady=8)
        tk.Button(top, text="Open progress.csv...", command=self.pick_csv).pack(side="left")
        self.path_label = tk.Label(top, text=csv_path or "No file selected", anchor="w")
        self.path_label.pack(side="left", padx=10)

        self.txt = tk.Text(self, height=26, width=80, font=("Consolas", 11))
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)

        self.footer = tk.Label(self, text="—", anchor="w")
        self.footer.pack(fill="x", padx=8, pady=(0,8))

        self.after(500, self.refresh)

    def pick_csv(self):
        path = filedialog.askopenfilename(title="Select progress.csv",
                                          filetypes=[("CSV", "*.csv")])
        if path:
            self.csv_path = path
            self.path_label.config(text=path)

    def refresh(self):
        if self.csv_path and os.path.exists(self.csv_path):
            row = self._read_last_row(self.csv_path)
            if row:
                block = self._format_block(row)
                self.txt.delete("1.0", tk.END)
                self.txt.insert(tk.END, block)
                self.footer.config(text=f"Last updated: {time.strftime('%H:%M:%S')}  |  Rows read: {row.get('_row_idx','?')}")
        self.after(REFRESH_MS, self.refresh)

    def _read_last_row(self, path):
        last = None
        try:
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                for idx, rec in enumerate(r):
                    last = rec
                    last["_row_idx"] = idx + 1
        except Exception as e:
            # file may be in-use; skip this cycle
            return None
        return last

    def _format_block(self, row):
        # Build a pretty, boxed output similar to your example
        def fmt(val):
            try:
                # ints if possible
                iv = int(float(val))
                if abs(float(val) - iv) < 1e-9:
                    return str(iv)
            except Exception:
                pass
            try:
                return f"{float(val):.6g}"
            except Exception:
                return str(val)

        # Group headings
        rollout_keys = [k for k,_ in FIELD_ORDER if k.startswith("rollout/")]
        time_keys    = [k for k,_ in FIELD_ORDER if k.startswith("time/")]
        train_keys   = [k for k,_ in FIELD_ORDER if k.startswith("train/")]

        lines = []
        width = 41
        hr = "-" * width
        lines.append(hr)
        lines.append("| rollout/                |             |")
        for k, short in FIELD_ORDER:
            if k in rollout_keys:
                v = fmt(row.get(k, ""))
                lines.append(f"|    {short:<20} | {v:<11} |")
        lines.append("| time/                   |             |")
        for k, short in FIELD_ORDER:
            if k in time_keys:
                v = fmt(row.get(k, ""))
                lines.append(f"|    {short:<20} | {v:<11} |")
        lines.append("| train/                  |             |")
        for k, short in FIELD_ORDER:
            if k in train_keys:
                v = fmt(row.get(k, ""))
                lines.append(f"|    {short:<20} | {v:<11} |")
        lines.append(hr)
        return "\n".join(lines)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    app = StatsGUI(path)
    app.mainloop()
