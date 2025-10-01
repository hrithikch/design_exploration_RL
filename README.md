# PPA Optimization: RL vs MOBO Comparison

This project demonstrates two different approaches to **Power-Performance-Area (PPA) optimization** in circuit design:
1. **Reinforcement Learning (RL)** with PPO - learns a policy to map preferences to design parameters
2. **Multi-Objective Bayesian Optimization (MOBO)** with qNEHVI - directly explores the Pareto frontier

Both approaches generate Pareto frontiers of optimal power/delay trade-offs for comparison.

---

## Project Structure

```
RL/  (project root)
├── approaches/
│   ├── rl/                          # Original RL approach
│   │   ├── config.yaml              # RL configuration
│   │   ├── main.py                  # RL entry point
│   │   ├── train_worker.py          # PPO training logic
│   │   ├── env_factory.py           # Environment creation
│   │   └── ...
│   └── mobo/                        # New MOBO approach
│       ├── configs/default.yaml     # MOBO configuration
│       ├── main.py                  # MOBO entry point
│       ├── core/                    # MOBO implementation
│       │   ├── loop.py              # BO iteration logic
│       │   ├── models.py            # Gaussian Process models
│       │   ├── acquisition.py       # qNEHVI, ParEGO acquisition
│       │   ├── search_space.py      # Variable bounds & normalization
│       │   └── evaluator/           # Objective evaluation
│       └── ui/                      # MOBO plotting utilities
├── shared/                          # Shared components
│   ├── envs/                        # PPA evaluation environments
│   │   └── surrogate_env.py         # Shared analytic PPA model
│   ├── utils/                       # Shared utilities
│   │   ├── pareto.py                # Unified Pareto computation
│   │   ├── config_loader.py         # YAML config loading
│   │   └── plotting.py              # Shared plotting functions
│   ├── live_stats.py                # Live monitoring GUI
│   └── final_plot.py                # Result visualization
├── runs/                            # Output directory
│   ├── rl_YYYYMMDD_HHMMSS/          # RL results
│   └── mobo_YYYYMMDD_HHMMSS/        # MOBO results
├── docs/                            # Documentation
│   ├── user-guide.md               # How to run experiments
│   ├── developer-guide.md          # Technical details
│   └── powershell-setup.md         # Windows PowerShell setup
├── legacy/                          # Legacy/unused code (archived)
├── compare.py                       # Side-by-side comparison script
├── run_comparison.py                # Automated run & compare script
├── run_with_venv.ps1               # PowerShell wrapper (optional)
├── requirements.txt                 # Combined dependencies
└── README.md                        # This file
```

---

## Quick Start

> 📖 **Detailed guides available in [`docs/`](docs/README.md)**
> - [User Guide](docs/user-guide.md) - Running experiments and configuration
> - [Developer Guide](docs/developer-guide.md) - Technical details and extending the framework
> - [PowerShell Setup](docs/powershell-setup.md) - Windows-native development

### 1. Setup Environment

**Windows (PowerShell - Recommended):**
```powershell
# Clone and setup
git clone <repo-url>
cd RL
python -m venv .RLtest
.RLtest\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Cross-platform (Bash):**
```bash
# Clone and setup
git clone <repo-url>
cd RL
python -m venv .RLtest
source .RLtest/Scripts/activate  # Windows: .RLtest\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Comparison

**Automated Comparison (Recommended):**
```powershell
# Activate virtual environment first
.RLtest\Scripts\Activate.ps1

# Run comparison
python run_comparison.py --quick    # Fast comparison
python run_comparison.py --full     # Full comparison

# Or use direct path (without activation)
.RLtest\Scripts\python.exe run_comparison.py --quick
```

**Manual Individual Runs:**
```powershell
# Activate virtual environment
.RLtest\Scripts\Activate.ps1

# Run RL approach (with live training GUI)
python approaches\rl\main.py --config approaches\rl\config.yaml

# Run MOBO approach (with results GUI)
python approaches\mobo\main.py

# Compare results
python compare.py --scan-latest
```

**Alternative: PowerShell Wrapper (Optional):**
```powershell
# Use wrapper script (equivalent to direct activation)
.\run_with_venv.ps1 run_comparison.py --quick
.\run_with_venv.ps1 approaches\mobo\main.py
.\run_with_venv.ps1 compare.py --scan-latest
```

> **Note:** The repository has been cleaned up. Legacy code from the original single-approach implementation has been moved to `legacy/` folder. The current structure uses a modular design with shared components and dedicated directories for each optimization approach.

---

## Approaches Comparison

| Aspect | RL (PPO) | MOBO (qNEHVI) |
|--------|----------|---------------|
| **Method** | Policy learning with preference weights | Direct Pareto exploration with GPs |
| **Strengths** | Generalizes to new preferences | Sample efficient, uncertainty-aware |
| **Sample Efficiency** | Needs many episodes | Very efficient with expensive evals |
| **Anytime Results** | Requires full training | Usable Pareto front at any iteration |
| **Scalability** | Better for high-D spaces | Excellent for 2-6 objectives |
| **Output** | Policy + discovered Pareto points | Explicit Pareto frontier |

---

## Configuration

### RL Configuration (`approaches/rl/config.yaml`)
```yaml
training:
  total_timesteps: 25000
  policy: MlpPolicy
  learning_rate: 0.0003

env:
  module: "shared.envs.surrogate_env"
  class: "SurrogatePPAEnv"
  params:
    a: 1.0    # Power scaling
    b: 1.0    # Delay scaling

sweep:
  points: 61  # Preference weight sweep resolution
```

### MOBO Configuration (`approaches/mobo/configs/default.yaml`)
```yaml
budget:
  init_evals: 12      # Initial Sobol samples
  batch_size: 4       # Parallel evaluations
  max_iters: 8        # BO iterations

acquisition:
  kind: qNEHVI        # or: ParEGO

search_space:
  size: {type: continuous, low: 0.1, high: 1.0}
  vdd:  {type: continuous, low: 0.7, high: 1.1}

stop:
  hv_plateau_window: 3
  hv_delta_tol: 0.01
```

---

## Outputs

Both approaches generate compatible outputs in `runs/` for direct comparison:

### Common Files
- `candidates.csv` - All evaluated design points with objectives
- `pareto.csv` - Final Pareto-optimal subset
- `progress.csv` - Iteration/training progress
- `pareto.png` - Pareto frontier visualization

### MOBO-Specific
- `attainment.png` - Convergence of Pareto fronts over iterations

### Comparison Outputs
- `comparison_plots/pareto_comparison.png` - Side-by-side Pareto frontiers
- `comparison_plots/convergence_comparison.png` - Training vs optimization progress

---

## Example Results

```bash
$ python compare.py --scan-latest

================================================================================
RL vs MOBO PPA Optimization Comparison
================================================================================
RL Results:   runs/rl_20250930_120000
MOBO Results: runs/mobo_20250930_130000
--------------------------------------------------------------------------------
Total Evaluations:
  RL:   305
  MOBO: 44

Pareto Frontier Size:
  RL:   23
  MOBO: 12

Hypervolume:
  RL:         0.284561
  MOBO:       0.301847
  Improvement: 6.08%
================================================================================
```

---

## Key Features

### Shared Components
- **Unified PPA Evaluator**: Same analytic model for fair comparison
- **Compatible Output Format**: Direct metric comparison between approaches
- **Shared Plotting**: Consistent visualization across methods
- **Unified Pareto Computation**: Works with both RL dicts and MOBO tensors

### RL-Specific
- **Preference Learning**: Learns policy mapping weights → design parameters
- **Stable-Baselines3 Integration**: Production-ready PPO implementation
- **Live Training GUI**: Real-time monitoring of training progress with Tkinter

### MOBO-Specific
- **qNEHVI Acquisition**: State-of-the-art multi-objective BO
- **ParEGO Alternative**: Scalarization-based baseline
- **Constraint Support**: Feasibility-weighted acquisition functions
- **Plateau Stopping**: Automatic convergence detection
- **Results GUI**: Automatic Pareto plot display when optimization completes

---

## Extending the Framework

### Adding New Objectives
1. Modify `shared/envs/surrogate_env.py` to return additional objectives
2. Update configs to include new objective names
3. Extend plotting utilities for >2D visualization

### Adding New Environments
1. Create new environment in `shared/envs/`
2. Update configs to reference new environment
3. Both approaches automatically work with new objective functions

### Adding New Approaches
1. Create new directory under `approaches/`
2. Implement using shared utilities for consistency
3. Add to comparison scripts

---

## Dependencies

- **Shared**: torch, numpy, pandas, matplotlib, pyyaml
- **RL**: gymnasium, stable-baselines3
- **MOBO**: botorch, gpytorch

See `requirements.txt` for exact versions.

---

## Troubleshooting

### Virtual Environment Issues
**"No module named 'botorch'"**: Use venv Python: `.RLtest/Scripts/python.exe -m pip install botorch`

**Activation doesn't work in Git Bash**: Use direct path instead: `.RLtest/Scripts/python.exe <script>`

**Wrong Python being used**: Check with `which python`, use wrapper script `./run_with_venv.sh`

See [VENV_USAGE.md](VENV_USAGE.md) for detailed virtual environment guide.

### General Issues
**Import Errors**: Ensure you're running from the project root directory

**Missing Dependencies**: Install all requirements: `.RLtest/Scripts/python.exe -m pip install -r requirements.txt`

**No Results Found**: Check that `runs/` directory contains timestamped result folders

**Plot Generation Fails**: Ensure matplotlib backend supports PNG output

**GUI doesn't show**: Verify Tkinter is available: `.RLtest/Scripts/python.exe -c "import tkinter"`