# How it works

Goal. Train an RL agent to pick design parameters that trade off power vs. delay, then sweep preferences to produce a Pareto front of optimal PPA trade-offs.

Environment (toy surrogate). A single-step Gymnasium env takes a weight vector [w_power, w_delay] as the observation and outputs an action [size, vdd]∈[0,1]^2. Reward is the negative weighted sum of normalized power & delay (minimization). Power ≈ a·size²·vdd²; Delay ≈ b/(eps + size·vdd).

Training. Uses Stable-Baselines3 PPO with config-driven defaults (policy, timesteps, n_steps, batch_size, lr, γ, seed). Logs to CSV and pre-creates a progress row so the GUI can start immediately.

Preference sweep → candidates → Pareto. After learning, the script sweeps weights t∈[0,1] (power vs. delay preferences), queries the trained policy, evaluates power/delay once per weight, writes candidates.csv, then filters non-dominated points to pareto.csv using a simple skyline algorithm.

UI & outputs. A Tkinter Live Stats window tails progress.csv during training; when done, a plot window shows candidate scatter and the Pareto curve. All outputs are timestamped under runs/.

Project layout & config. Clean modules: envs/ (surrogate env), rl/ (config loader, Pareto, training), gui/ (live stats, final plot); run via main.py / rl_live_demo2.py with a YAML config controlling training/env/logging/sweep/plot.

# What it demonstrates

Feasibility: An RL policy can learn to map design-preference weights → design parameters that balance PPA objectives, even in a simple single-step setting.

Pareto extraction workflow: After training once, sweeping preferences generates a Pareto front that visualizes the design trade-space (and exports clean CSV artifacts for downstream analysis).

Reusable skeleton: A modular template—swap the env with a more realistic EDA surrogate or tool wrapper, keep the PPO/training, logging, sweep, and plotting pieces intact.