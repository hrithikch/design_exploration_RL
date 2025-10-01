#train_worker.py
import os, csv, sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from env_factory import make_env
from shared.utils.pareto import pareto_from_dicts

def train_and_log(cfg, run_dir, done_event, sweep_done_event):
    os.makedirs(run_dir, exist_ok=True)
    log_files = cfg["log"]["files"]
    progress_csv = os.path.join(run_dir, log_files["progress"])
    cand_csv     = os.path.join(run_dir, log_files["candidates"])
    pareto_csv   = os.path.join(run_dir, log_files["pareto"])

    # Logger
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

    # Pre-create progress row
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

    pareto = pareto_from_dicts(rows)
    with open(pareto_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["power","delay","size","vdd","w_power","w_delay"])
        wtr.writeheader(); wtr.writerows(pareto)

    sweep_done_event.set()
