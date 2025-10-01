# MOBO-Based PPA Pareto Front Exploration

## Core Concept

Replace the RL-based preference learning approach with **Multi-Objective Bayesian Optimization (MOBO)** to directly explore the Power-Performance-Area (PPA) trade-space and generate Pareto-optimal designs.

## Key Advantages Over RL

- **Sample Efficiency**: MOBO excels with expensive evaluations, using Gaussian Process surrogates to intelligently propose next evaluation points
- **Direct Pareto Exploration**: No need for preference weight sweeping - MOBO naturally explores the entire Pareto frontier
- **Uncertainty Quantification**: GP models provide confidence estimates for unexplored regions
- **Anytime Results**: Usable Pareto front available at any iteration, not just after full training

## Problem Formulation

**Objective**: Minimize `(Power, Delay)` simultaneously
**Design Variables**: `[size, vdd]` ∈ [0,1]²
**Constraints**: Optional timing constraints (e.g., slack ≥ 0)

## MOBO Pipeline

### 1. Initialization
- **Latin Hypercube Sampling**: Generate 12-20 initial design points
- **Evaluate PPA**: Run surrogate model to get (power, delay) for each point
- **Feasibility Check**: Apply any hard constraints (timing, DRC)

### 2. Surrogate Modeling
- **Independent GPs**: One Gaussian Process per objective (power, delay)
- **Kernels**: ARD RBF kernels with automatic relevance determination
- **Noise Handling**: Inferred noise levels for each objective

### 3. Acquisition Function
- **qNEHVI** (Noisy Expected Hypervolume Improvement): Accounts for GP uncertainty and supports batch evaluation
- **Alternative**: qEHVI for noiseless case, or ParEGO for simpler scalarization approach
- **Constraint Handling**: Multiply acquisition by feasibility probability

### 4. Batch Optimization
- **Multi-start LBFGS**: Optimize acquisition function to find next q=4 candidate points
- **Parallel Evaluation**: Evaluate all candidates simultaneously using surrogate
- **Fantasy Points**: Handle pending evaluations during batch optimization

### 5. Model Updates
- **Incremental Fitting**: Add new data and refit GP hyperparameters
- **Standardization**: Normalize objectives for stable GP training
- **Reference Point**: Update based on worst observed values + margin

### 6. Convergence & Stopping
- **Hypervolume Plateau**: Stop when hypervolume improvement < threshold for N iterations
- **Budget Exhaustion**: Fixed evaluation budget (e.g., 100 evaluations)
- **Wall Clock**: Time-based stopping for expensive evaluations

## Implementation Structure

```
mobo_ppa/
├── main.py                    # Entry point with config loading
├── config/
│   └── default.yaml          # MOBO hyperparameters and search space
├── core/
│   ├── search_space.py       # Variable bounds, normalization, constraints
│   ├── init_design.py        # Sobol/LHS initial sampling
│   ├── models.py             # GP fitting, standardization, caching
│   ├── acquisition.py        # qNEHVI, constraint handling
│   ├── optimize_acq.py       # Acquisition optimization (LBFGS)
│   ├── evaluator/
│   │   ├── toy_surrogate.py  # Fast analytic PPA model
│   │   └── eda_wrapper.py    # Real EDA tool interface
│   ├── loop.py               # Main BO iteration logic
│   └── pareto.py             # Non-dominated filtering, hypervolume
├── ui/
│   ├── live_stats.py         # Real-time hypervolume tracking
│   └── final_plots.py        # Pareto scatter, parallel coordinates
└── runs/
    └── YYYYMMDD_HHMMSS/      # Timestamped results
        ├── progress.csv      # Iteration, hypervolume, eval count
        ├── candidates.csv    # All evaluated designs
        ├── pareto.csv        # Final non-dominated set
        └── pareto.png        # Visualization
```

## Configuration

```yaml
seed: 0
budget:
  init_evals: 16              # Sobol initialization points
  batch_size: 4               # Parallel evaluations per iteration
  max_iters: 20               # BO iterations

objectives: [power, delay]    # Can extend to [power, delay, area]

constraints:
  - name: timing_ok
    type: hard
    expr: "slack >= 0"

search_space:
  size: {type: continuous, low: 0.1, high: 1.0}
  vdd:  {type: continuous, low: 0.7, high: 1.1}

model:
  kind: gp_independent        # Independent GPs per objective
  noise: inferred             # Learn noise automatically

acquisition:
  kind: qNEHVI               # Noisy Expected Hypervolume Improvement
  constraints: ["timing_ok"]  # Apply feasibility weighting

optimizer:
  inner: lbfgs               # Acquisition optimization method
  restarts: 10               # Multi-start for global optimization
  q: 4                       # Batch size

evaluator:
  backend: toy_surrogate     # Switch to eda_wrapper for real tools

stop:
  hv_plateau_window: 5       # Iterations to check for plateau
  hv_delta_tol: 0.01         # Hypervolume improvement threshold
```

## Outputs (Compatible with RL Version)

- **progress.csv**: Iteration, hypervolume, cumulative evaluations
- **candidates.csv**: All evaluated designs with objectives and feasibility
- **pareto.csv**: Final Pareto-optimal subset
- **pareto.png**: Scatter plot with Pareto frontier highlighted

## Extension Points

1. **Mixed Variables**: Add categorical variables (e.g., cell types) with appropriate kernels
2. **Multi-Fidelity**: Incorporate low/high fidelity evaluations for faster exploration
3. **Constraint Learning**: Learn constraint boundaries from failed evaluations
4. **Transfer Learning**: Warm-start with data from previous design explorations
5. **Real EDA Integration**: Replace surrogate with OpenROAD/Innovus/Cadence tools

## Expected Performance vs RL

- **Sample Efficiency**: 5-10x fewer evaluations to reach similar Pareto quality
- **Stability**: Lower variance across random seeds due to principled uncertainty quantification
- **Interpretability**: Clear acquisition function reasoning vs black-box policy learning
- **Scalability**: Better for expensive evaluations, RL better for high-dimensional spaces