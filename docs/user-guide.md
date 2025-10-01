# User Guide: PPA Optimization with RL vs MOBO

## Overview

This project provides two approaches to optimize Power-Performance-Area (PPA) trade-offs in circuit design:

- **RL (Reinforcement Learning)**: Learns a policy to map preferences to design parameters
- **MOBO (Multi-Objective Bayesian Optimization)**: Directly explores the Pareto frontier using Gaussian Processes

## Quick Start

### Prerequisites

- Windows 10/11 (primary support)
- Python 3.9+
- Git

### Installation

1. **Clone the repository**:
   ```powershell
   git clone <repository-url>
   cd RL
   ```

2. **Set up virtual environment**:
   ```powershell
   python -m venv .RLtest
   .RLtest\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Test installation**:
   ```powershell
   .\run_with_venv.ps1 -c "import botorch; print('Ready!')"
   ```

## Running Experiments

### Method 1: Individual Approaches

**Run RL Approach (with live training GUI):**
```powershell
.\run_with_venv.ps1 approaches\rl\main.py --config approaches\rl\config.yaml
```

**Run MOBO Approach (with results GUI):**
```powershell
.\run_with_venv.ps1 approaches\mobo\main.py
```

### Method 2: Automated Comparison

**Quick comparison (reduced settings):**
```powershell
.\run_with_venv.ps1 run_comparison.py --quick
```

**Full comparison (default settings):**
```powershell
.\run_with_venv.ps1 run_comparison.py --full
```

### Method 3: Manual Comparison

```powershell
# Run both approaches individually, then compare
.\run_with_venv.ps1 compare.py --scan-latest
```

## Understanding the Output

### File Structure
After running, you'll find results in the `runs/` directory:

```
runs/
├── rl_YYYYMMDD_HHMMSS/          # RL results
│   ├── candidates.csv           # All design points evaluated
│   ├── pareto.csv              # Pareto-optimal solutions
│   ├── progress.csv            # Training progress
│   └── pareto.png              # Visualization
└── mobo_YYYYMMDD_HHMMSS/       # MOBO results
    ├── candidates.csv          # All design points evaluated
    ├── pareto.csv              # Pareto-optimal solutions
    ├── progress.csv            # Optimization progress
    ├── pareto.png              # Pareto frontier plot
    └── attainment.png          # Convergence visualization
```

### Key Metrics

- **Total Evaluations**: How many design points were tested
- **Pareto Frontier Size**: Number of optimal trade-off solutions found
- **Hypervolume**: Quality metric for multi-objective optimization (higher = better)
- **Ranges**: Min/max values for power and delay objectives

## Configuration Guide

### RL Configuration (`approaches/rl/config.yaml`)

#### Training Parameters
```yaml
training:
  total_timesteps: 25000      # More = better policy, slower training
  policy: "MlpPolicy"         # Neural network architecture
  learning_rate: 0.0003       # Learning speed (0.0001-0.001)
  gamma: 0.99                 # Future reward discount (0.9-0.999)
  n_steps: 64                 # Steps per training batch
  batch_size: 64              # Training batch size
  seed: null                  # Random seed (null = random)
```

#### Environment Parameters
```yaml
env:
  module: "shared.envs.surrogate_env"
  class: "SurrogatePPAEnv"
  params:
    a: 1.0                    # Power scaling factor
    b: 1.0                    # Delay scaling factor
    eps: 0.001                # Numerical stability term
    seed: null                # Environment random seed
```

#### Preference Sweep
```yaml
sweep:
  points: 61                  # Resolution of preference sweep (more = finer)
```

**Tuning Tips:**
- **Faster training**: Reduce `total_timesteps` to 10000-15000
- **Better quality**: Increase `total_timesteps` to 50000+
- **More exploration**: Lower `gamma` to 0.95
- **Finer Pareto front**: Increase `sweep.points` to 101+

### MOBO Configuration (`approaches/mobo/configs/default.yaml`)

#### Budget Parameters
```yaml
budget:
  init_evals: 12              # Initial random samples (8-20)
  batch_size: 4               # Parallel evaluations (1-8)
  max_iters: 8                # BO iterations (5-20)
```

#### Search Space
```yaml
search_space:
  size: {type: continuous, low: 0.1, high: 1.0}    # Transistor size
  vdd:  {type: continuous, low: 0.7, high: 1.1}    # Supply voltage
```

#### Acquisition Function
```yaml
acquisition:
  kind: qNEHVI                # Options: qNEHVI, ParEGO
```

#### Stopping Criteria
```yaml
stop:
  hv_plateau_window: 3        # Iterations to check for plateau
  hv_delta_tol: 0.01          # Hypervolume improvement threshold
```

**Tuning Tips:**
- **Faster results**: Reduce `init_evals` to 8, `max_iters` to 5
- **Better exploration**: Increase `init_evals` to 20
- **More thorough**: Increase `max_iters` to 15-20
- **Parallel hardware**: Increase `batch_size` to match CPU cores
- **Conservative stopping**: Increase `hv_plateau_window` to 5

### Global Settings

#### Logging
```yaml
log:
  outputs: ["csv"]            # Options: ["csv", "stdout"]
  precreate_progress_row: true
```

#### Plotting
```yaml
plot:
  figure_dpi: 120             # Image quality (100-300)
  figure_size: [6.5, 4.3]     # Plot dimensions [width, height]
```

## GUI Features

### RL Live Training GUI
- **Real-time metrics**: Reward, loss, timesteps
- **Progress tracking**: Training completion percentage
- **Live updates**: Refreshes every 500ms during training

### MOBO Results GUI
- **Pareto visualization**: Interactive matplotlib plot
- **Summary statistics**: Evaluations, hypervolume, ranges
- **Export functionality**: Save plots as PNG
- **File browser**: Direct links to generated files

## Comparison Analysis

The comparison script generates:

1. **Terminal Report**: Quantitative metrics comparison
2. **Pareto Comparison Plot**: Side-by-side frontier visualization
3. **Convergence Plot**: Training vs optimization progress

### Interpreting Results

**Sample Efficiency**: MOBO typically requires 50-80% fewer evaluations

**Quality Metrics**:
- Positive hypervolume improvement indicates MOBO advantage
- Tighter objective ranges suggest more focused search

**Speed**: MOBO usually completes in minutes, RL in 10-30 minutes

## Troubleshooting

### Common Issues

**"No module named 'botorch'"**:
```powershell
.\run_with_venv.ps1 -m pip install botorch
```

**PowerShell execution policy error**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**GUI doesn't appear**:
- Check if Tkinter is available: `.\run_with_venv.ps1 -c "import tkinter"`
- Ensure you're not running in headless/SSH environment

**Poor RL performance**:
- Increase `total_timesteps` in RL config
- Check that training is completing (watch GUI)

**MOBO convergence issues**:
- Increase `init_evals` for better initial coverage
- Try `ParEGO` acquisition if `qNEHVI` has issues

### Getting Help

1. Check the [Developer Guide](developer-guide.md) for technical details
2. Review [Virtual Environment Guide](../VENV_USAGE.md) for setup issues
3. Examine log files in `runs/` directories for error details

## Advanced Usage

### Custom Objectives

To modify the PPA formulation, edit `shared/envs/surrogate_env.py`:

```python
# Change power/delay calculations
power = self.a * (size**2) * (vdd**2)      # Your power model
delay = self.b / (self.eps + size * vdd)   # Your delay model
```

### Adding Constraints

In MOBO config:
```yaml
constraints:
  - name: timing_ok
    type: hard
    expr: "slack >= 0"
  - name: power_limit
    type: hard
    expr: "power <= 0.5"
```

### Multiple Runs

For statistical analysis:
```powershell
# Run multiple times with different seeds
for ($i=1; $i -le 5; $i++) {
    .\run_with_venv.ps1 approaches\mobo\main.py
}
```

This creates a robust comparison framework for evaluating different optimization approaches to circuit design problems.