#main.py
import os, threading, argparse
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.utils.config_loader import load_config
from train_worker import train_and_log
from shared.live_stats import LiveStatsApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("../../", cfg["run"]["root"], f"rl_{ts}")

    done_event = threading.Event()
    sweep_done_event = threading.Event()

    t = threading.Thread(
        target=train_and_log,
        args=(cfg, run_dir, done_event, sweep_done_event),
        daemon=True
    )
    t.start()

    app = LiveStatsApp(cfg, run_dir, done_event, sweep_done_event)
    app.mainloop()
