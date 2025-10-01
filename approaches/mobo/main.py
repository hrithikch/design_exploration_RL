import os, yaml
from datetime import datetime
from core.loop import run
from ui.final_plots import plot_pareto, plot_attainment
from ui.live_gui import show_mobo_gui


def main():
    # Handle being called from different working directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_cfg_path = os.path.join(script_dir, "configs", "default.yaml")
    cfg_path = os.environ.get("MOBO_CFG", default_cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("../../", cfg["logging"]["dir"], f"mobo_{stamp}")

    print("Starting MOBO optimization...")
    X, Y, pareto_Y, pareto_hist = run(cfg, run_dir)

    # Generate plots
    plot_pareto(Y.numpy(), pareto_Y.numpy(), os.path.join(run_dir, "pareto.png"))
    if len(pareto_hist) > 1:
        import numpy as np
        ph = [p.numpy() for p in pareto_hist]
        plot_attainment(ph, os.path.join(run_dir, "attainment.png"))

    print(f"Done. Outputs in {run_dir}")

    # Show GUI with results
    print("Launching results GUI...")
    show_mobo_gui(run_dir)

if __name__ == "__main__":
    main()