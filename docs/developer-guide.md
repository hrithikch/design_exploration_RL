# Developer Guide: PPA Optimization Framework

## Introduction to Machine Learning in Circuit Design

### What is Multi-Objective Optimization?

In circuit design, we often face trade-offs between competing objectives:
- **Power**: Energy consumption (lower is better)
- **Performance**: Speed/delay (lower delay = higher performance)
- **Area**: Silicon real estate (lower is better)

The goal is to find the **Pareto frontier** - a set of designs where improving one objective requires sacrificing another.

### Why Machine Learning?

Traditional approaches require:
1. Manual parameter sweeping (slow, incomplete)
2. Domain expertise for each new problem
3. Separate optimization for each objective

Machine learning approaches:
1. **Learn** optimal design patterns automatically
2. **Explore** the design space intelligently
3. **Generalize** to new scenarios with minimal setup

## Framework Architecture

### High-Level Design

```
RL/  (project root)
├── approaches/           # Different optimization methods
│   ├── rl/              # Reinforcement Learning approach
│   └── mobo/            # Multi-Objective Bayesian Optimization
├── shared/              # Common components
│   ├── envs/            # Problem environments
│   ├── utils/           # Shared utilities
│   ├── live_stats.py    # GUI components
│   └── final_plot.py
└── docs/               # Documentation
```

### Core Concepts

#### 1. Environment (Problem Definition)
The **environment** defines what we're optimizing. It:
- Takes design parameters as input (e.g., size, voltage)
- Returns objective values (e.g., power, delay)
- Can include constraints (e.g., timing requirements)

#### 2. Optimizer (Search Strategy)
The **optimizer** decides which designs to try next:
- **RL**: Learns a policy through trial and error
- **MOBO**: Uses Gaussian Processes to model the design space

#### 3. Evaluation Loop
Both approaches follow the same pattern:
1. Propose new design parameters
2. Evaluate objectives (power, delay, etc.)
3. Update internal models
4. Repeat until convergence

## Understanding Each Approach

### Reinforcement Learning (RL)

**How it works:**
```
1. Agent observes "preference weights" [w_power, w_delay]
2. Agent proposes design parameters [size, vdd]
3. Environment evaluates → power, delay values
4. Agent gets reward = -(w_power×power + w_delay×delay)
5. Agent learns to maximize reward (minimize weighted objectives)
```

**Key files:**
- `approaches/rl/train_worker.py`: PPO training loop
- `shared/envs/surrogate_env.py`: Problem environment
- `approaches/rl/env_factory.py`: Environment creation

**When to use RL:**
- Need a policy that generalizes to new preferences
- Have many different objective weightings to explore
- Design space is high-dimensional (>10 variables)

### Multi-Objective Bayesian Optimization (MOBO)

**How it works:**
```
1. Start with random design samples
2. Fit Gaussian Process models to observed data
3. Use acquisition function to find promising areas
4. Evaluate most promising designs
5. Update models and repeat
```

**Key files:**
- `approaches/mobo/core/loop.py`: Main optimization loop
- `approaches/mobo/core/models.py`: Gaussian Process models
- `approaches/mobo/core/acquisition.py`: qNEHVI acquisition function

**When to use MOBO:**
- Evaluations are expensive (EDA tools, simulations)
- Want direct Pareto frontier
- 2-6 objectives, moderate dimensional space

## Modular Design & Extensibility

### Adding New Environments

1. **Create environment file** in `shared/envs/`:

```python
# shared/envs/my_new_env.py
import numpy as np
import gymnasium as gym

class MyCircuitEnv(gym.Env):
    def __init__(self, **kwargs):
        # Define observation/action spaces
        self.observation_space = gym.spaces.Box(...)
        self.action_space = gym.spaces.Box(...)

    def _evaluate(self, design_params):
        # Your custom PPA evaluation
        power = calculate_power(design_params)
        delay = calculate_delay(design_params)
        area = calculate_area(design_params)
        return {"power": power, "delay": delay, "area": area}

    def step(self, action):
        metrics = self._evaluate(action)
        reward = self.compute_reward(metrics)
        return observation, reward, done, truncated, info
```

2. **Update configuration**:

```yaml
# In RL config
env:
  module: "shared.envs.my_new_env"
  class: "MyCircuitEnv"
  params:
    custom_param: 1.0
```

3. **MOBO compatibility**: MOBO can use any environment that provides objective values.

### Adding New Objectives

1. **Update environment** to return additional objectives:

```python
def _evaluate(self, design_params):
    return {
        "power": ...,
        "delay": ...,
        "area": ...,        # New objective
        "leakage": ...      # Another new objective
    }
```

2. **Update configurations**:

```yaml
# MOBO config
objectives: [power, delay, area, leakage]  # Add new objectives
```

3. **Update plotting**: Modify `shared/utils/plotting.py` for >2D visualization.

### Adding New Acquisition Functions

1. **Create acquisition class** in `approaches/mobo/core/acquisition.py`:

```python
class MyCustomAcquisition:
    def __init__(self, model, **kwargs):
        self.model = model
        # Initialize your acquisition function

    def __call__(self, X):
        # Return acquisition values for candidate points X
        return acquisition_values
```

2. **Update loop** in `approaches/mobo/core/loop.py`:

```python
# Add to acquisition selection
if acq_kind.lower() == "mycustom":
    acq = MyCustomAcquisition(model, **config)
```

3. **Update config options**:

```yaml
acquisition:
  kind: MyCustom  # Your new acquisition function
```

### Adding New Optimization Approaches

1. **Create approach directory**:

```
approaches/
└── my_new_approach/
    ├── main.py
    ├── configs/
    ├── core/
    └── ui/
```

2. **Follow the interface pattern**:

```python
# approaches/my_new_approach/main.py
def main():
    # Load config
    cfg = load_config("configs/default.yaml")

    # Create timestamped output directory
    run_dir = f"../../runs/mynew_{timestamp}"

    # Run optimization
    results = optimize(cfg, run_dir)

    # Generate outputs (candidates.csv, pareto.csv, plots)
    export_results(results, run_dir)

    # Optional: Show GUI
    show_gui(run_dir)
```

3. **Update comparison scripts** to recognize new output format.

## Understanding the Code

### Key Data Structures

#### RL Results Format
```python
# List of dictionaries, one per design point
rl_results = [
    {
        "size": 0.5, "vdd": 0.9,           # Design parameters
        "power": 0.234, "delay": 1.567,    # Objectives
        "w_power": 0.3, "w_delay": 0.7     # Preference weights
    },
    # ... more points
]
```

#### MOBO Results Format
```python
# Tensors for efficient computation
X = torch.tensor([[0.5, 0.9], ...])        # Design parameters (N, D)
Y = torch.tensor([[0.234, 1.567], ...])    # Objectives (N, M)
```

### Shared Utilities

#### Pareto Computation (`shared/utils/pareto.py`)
```python
# Works with both formats
rl_pareto = pareto_from_dicts(rl_results)
mobo_pareto_idx, _ = pareto_from_tensors(Y)

# Convert between formats
Y_tensor = convert_rl_to_mobo_format(rl_results)
rl_format = convert_mobo_to_rl_format(X, Y)
```

#### Configuration Loading (`shared/utils/config_loader.py`)
```python
cfg = load_config("path/to/config.yaml")
merged = merge_configs(base_config, override_config)
```

### GUI Components

#### RL Live Stats (`shared/live_stats.py`)
- Monitors CSV files during training
- Updates plots in real-time
- Handles training completion events

#### MOBO Results GUI (`approaches/mobo/ui/live_gui.py`)
- Waits for optimization completion
- Displays final Pareto frontier
- Provides export functionality

## Development Workflow

### 1. Setting Up Development Environment

```powershell
# Clone and setup
git clone <repo>
cd RL

# Create development environment
python -m venv .RLtest
.RLtest\Scripts\Activate.ps1
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 jupyter
```

### 2. Making Changes

#### Code Style
- Use black for formatting: `black .`
- Check with flake8: `flake8 .`
- Document functions with docstrings

#### Testing
```python
# Test new environments
env = MyNewEnv()
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(action)

# Test configurations
cfg = load_config("my_config.yaml")
assert cfg["expected_key"] == expected_value
```

### 3. Adding Features

#### New Objective Functions
1. Modify environment's `_evaluate()` method
2. Update configs to include new objectives
3. Test with both RL and MOBO
4. Update plotting for new dimensionality

#### New Constraints
1. Add constraint evaluation to environment
2. Update MOBO acquisition to handle constraints
3. For RL, incorporate constraints into reward

#### Performance Improvements
1. Profile code: `python -m cProfile script.py`
2. Use vectorized operations where possible
3. Consider GPU acceleration for large problems

## Integration with Real EDA Tools

### Replacing the Surrogate

The current implementation uses analytic formulas. To use real EDA tools:

1. **Create EDA wrapper**:

```python
# shared/envs/eda_env.py
class EDAEnvironment:
    def _evaluate(self, design_params):
        # Write design to file
        write_design_file(design_params, "temp_design.sp")

        # Run EDA tool
        result = subprocess.run(["tool", "temp_design.sp"],
                              capture_output=True)

        # Parse results
        power, delay, area = parse_results(result.stdout)
        return {"power": power, "delay": delay, "area": area}
```

2. **Handle tool-specific issues**:
- Timeouts and retries
- License management
- File system cleanup
- Error handling for failed runs

3. **Optimization considerations**:
- Cache results to avoid re-evaluation
- Use parallel tool licenses for MOBO batching
- Handle tool crashes gracefully

### Scaling to Production

#### Database Integration
```python
# Store results in database for analysis
import sqlite3

def save_results(run_id, X, Y, metadata):
    conn = sqlite3.connect("optimization_results.db")
    # Store design points, objectives, and run metadata
```

#### Cluster Computing
```python
# Distribute evaluations across compute cluster
from dask.distributed import Client

client = Client("scheduler-address")
futures = client.map(evaluate_design, design_candidates)
results = client.gather(futures)
```

## Advanced Topics

### Custom Kernels for MOBO

For specialized design spaces:

```python
# Custom kernel for mixed continuous/categorical variables
from gpytorch.kernels import RBFKernel, ScaleKernel

class MixedKernel(Kernel):
    def __init__(self):
        super().__init__()
        self.continuous_kernel = RBFKernel()
        self.categorical_kernel = HammingKernel()

    def forward(self, x1, x2):
        # Combine kernels for different variable types
        cont_part = self.continuous_kernel(x1[:, :n_cont], x2[:, :n_cont])
        cat_part = self.categorical_kernel(x1[:, n_cont:], x2[:, n_cont:])
        return cont_part * cat_part
```

### Transfer Learning

Reuse knowledge across similar problems:

```python
# Transfer GP hyperparameters from previous runs
def transfer_hyperparameters(new_model, old_model):
    new_model.load_state_dict(old_model.state_dict(), strict=False)
```

### Multi-Fidelity Optimization

Use cheap approximations to guide expensive evaluations:

```python
class MultiFidelityEnvironment:
    def evaluate(self, design, fidelity="high"):
        if fidelity == "low":
            return fast_approximation(design)
        else:
            return accurate_evaluation(design)
```

This modular framework makes it easy to experiment with new optimization methods, integrate with real EDA tools, and scale to production circuit design workflows.