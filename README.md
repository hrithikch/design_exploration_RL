# RL PPA Exploration — Proof of Concept

This project demonstrates how **reinforcement learning (RL)** can automatically explore trade-offs between **power** and **delay** in circuit design. It uses a **surrogate environment** (simple analytic model) and trains an RL agent (PPO) to learn a **Pareto curve** of optimal trade-offs.  
It’s organized into clean modules (envs, RL, GUI) and run from a single entrypoint.

---

## 🧩 What you’ll see when it runs

- A **live stats window** (Tkinter) that shows training metrics (iterations, total timesteps, reward, loss, etc.).
- When training finishes, a **plot window** with the **candidates** and the computed **Pareto front**.
- A timestamped folder under `runs/` with CSV outputs so you can reuse the data.

---

## 🗂️ Project Structure (overview)

project-root/
├─ main.py
├─ config.yaml # All settings go here
├─ requirements.txt
├─ envs/
│ ├─ init.py
│ └─ surrogate_env.py # SurrogatePPAEnv: toy PPA formulas
├─ rl/
│ ├─ init.py
│ ├─ config_loader.py # YAML load + defaults merging
│ ├─ env_factory.py # Imports env class from config (safe kwargs)
│ ├─ pareto.py # Non-dominated filtering
│ └─ train_worker.py # Training + sweep + CSV outputs
└─ gui/
├─ init.py
├─ live_stats.py # Live training metrics window
└─ final_plot.py # Final candidates + Pareto plot

yaml
Copy code

---

## 🖥️ Setup (Windows)

> Assumes you have **Python 3.9+** and **Git** installed. If PowerShell blocks script activation, see step 2b.

### 1) Clone the repository
```powershell
git clone https://github.com/your-org/rl-ppa-demo.git
cd rl-ppa-demo
### 2) Create & activate a virtual environment
powershell
Copy code
python -m venv .venv
.venv\Scripts\Activate.ps1
2b) If activation is blocked (PowerShell policy):

powershell
Copy code
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
### 3) Install dependencies
powershell
Copy code
pip install --upgrade pip
pip install -r requirements.txt
If you installed Python via the Microsoft Store and the Tkinter GUI doesn’t appear, please install the python.org distribution instead. (Tkinter is included there by default.)

### 4) Run
powershell
Copy code
python main.py --config config.yaml
A live stats window opens immediately.

When training completes, a plot window opens (Pareto front).

Outputs go to runs/YYYYMMDD_HHMMSS/.

## Configuration (edit config.yaml)
yaml
Copy code
training:
  policy: "MlpPolicy"
  total_timesteps: 20000   # increase for smoother results
  n_steps: 64
  batch_size: 64
  learning_rate: 0.0003
  gamma: 0.99
  seed: null

env:
  module: "envs.surrogate_env"
  class: "SurrogatePPAEnv"
  params:
    a: 1.0
    b: 1.0
    eps: 0.001
    seed: null

run:
  root: "runs"

log:
  outputs: ["csv"]         # add "stdout" to also print in console
  precreate_progress_row: true
  files:
    progress: "progress.csv"
    candidates: "candidates.csv"
    pareto: "pareto.csv"

sweep:
  points: 61               # how many weights to sample along [0..1]

gui:
  refresh_ms: 500

plot:
  figure_dpi: 120
  figure_size: [6.5, 4.3]

## Outputs
Inside runs/<timestamp>/ you’ll find:

progress.csv — training metrics

candidates.csv — all evaluated points along the weight sweep

pareto.csv — non-dominated subset (Pareto front)

## Troubleshooting
“ModuleNotFoundError: envs.surrogate_env”
Make sure you run python main.py from the project root (so relative imports work) and that envs/__init__.py exists.

GUI doesn’t open
Verify your Python distribution includes Tkinter. The python.org installer does; some variants don’t.

Unexpected keyword argument ‘a’ (or similar)
Your env.params in config.yaml have keys the env’s __init__ doesn’t accept.
Remove those keys or add them to the class signature.

Timesteps end at a different number than requested
PPO processes steps in chunks of n_steps × n_envs. It rounds up to the next multiple.
