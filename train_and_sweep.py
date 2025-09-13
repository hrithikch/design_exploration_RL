# train_and_sweep.py (save results to CSV)
import os, csv, time, numpy as np, matplotlib.pyplot as plt
from stable_baselines3 import PPO
from surrogate_env import SurrogatePPAEnv

RUN_DIR = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)
CAND_CSV = os.path.join(RUN_DIR, "candidates.csv")
PARETO_CSV = os.path.join(RUN_DIR, "pareto.csv")

def get_pareto(points):  # minimization in power, delay
    P = sorted(points, key=lambda d: (d["power"], d["delay"]))
    pareto = []
    best_d = float("inf")
    for p in P:
        if p["delay"] < best_d:
            pareto.append(p)
            best_d = p["delay"]
    return pareto

def main():
    env = SurrogatePPAEnv()
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, learning_rate=3e-4, gamma=0.99, verbose=0)
    model.learn(total_timesteps=10_000)

    # Sweep weights and record
    rows = []
    with open(CAND_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["w_power","w_delay","size","vdd","power","delay"])
        wtr.writeheader()
        for t in np.linspace(0, 1, 41):
            w = np.array([t, 1.0 - t], dtype=np.float32)
            action, _ = model.predict(w, deterministic=True)
            env.w = w
            _, _, _, _, info = env.step(action)
            rec = dict(
                w_power=float(w[0]), w_delay=float(w[1]),
                size=float(info["size"]), vdd=float(info["vdd"]),
                power=float(info["power"]), delay=float(info["delay"])
            )
            rows.append(rec)
            wtr.writerow(rec)

    pareto = get_pareto(rows)
    with open(PARETO_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["power","delay","size","vdd","w_power","w_delay"])
        wtr.writeheader()
        for r in pareto:
            wtr.writerow(r)

    print(f"Wrote: {CAND_CSV}\n       {PARETO_CSV}")

if __name__ == "__main__":
    main()
