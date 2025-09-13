# train_with_logging.py
import os, time
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from surrogate_env import SurrogatePPAEnv  # your existing env

def main():
    log_root = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_root, exist_ok=True)
    print(f"[INFO] Log dir: {log_root}")
    # Configure SB3 logger to write CSV (progress.csv) in this dir
    new_logger = configure(log_root, ["csv"])  # ("stdout","csv") if you also want console prints

    env = SurrogatePPAEnv()
    model = PPO("MlpPolicy", env,
                n_steps=64, batch_size=64, learning_rate=3e-4,
                gamma=0.99, verbose=0)
    model.set_logger(new_logger)

    # Train; progress.csv will be written to log_root
    model.learn(total_timesteps=10_000)

    print("[INFO] Training finished.")
    print(f"[INFO] CSV path: {os.path.join(log_root, 'progress.csv')}")

if __name__ == "__main__":
    main()
