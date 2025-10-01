# Outline for a Multi-Objective Bayesian Optimization (MOBO) project that targets the same end result: a Pareto front of PPA trade-offs.

## Project goals

Efficiently explore design parameters to minimize (Power, Delay, Area) (choose any 2–3).

Produce stable Pareto set & Pareto front artifacts identical to the RL flow (CSV + plots).

Be drop-in replaceable for the “objective evaluator” (toy surrogate ↔ real EDA run).

## High-level pipeline

Define search space & constraints

Continuous (e.g., size∈[0,1], vdd∈[0.7,1.1]), integer (e.g., unroll∈{1..8}), categorical (cells, corners).

Hard constraints (DRC/STA fail) & soft constraints (e.g., slack ≥ 0).

Initialize design points

Latin Hypercube / Sobol samples (e.g., 8–32 points depending on dimension).

Evaluate PPA; store (x, y_power, y_delay, y_area, feasibility).

Fit surrogate(s)

One GP per objective (or a multi-output GP); heteroscedastic noise optional.

Mixed-variable kernels (Hamming for categorical + ARD RBF for continuous).

Choose MOBO acquisition (support both to compare):

qEHVI / qNEHVI (expected hypervolume improvement; supports batching & constraints).

ParEGO (random scalarizations, single-objective EI loop; simple & robust).

Optional: Tchebycheff scalarization, Knowledge Gradient variants.

Batch candidate generation

Optimize acquisition (multi-start LBFGS for continuous; evolutionary fallback for mixed).

Get q candidates per iteration (parallel EDA runs).

Evaluate candidates (objective evaluator)

Toy surrogate for quick demo; EDA wrapper for real runs (returns power/delay/area + validity flags).

Log results; append to dataset.

Update models & loop

Refit GPs (warm start); adapt trust region if desired (to stabilize in higher-D).

Stop on budget, wall clock, or hypervolume plateau.

Post-processing

Compute non-dominated set; export candidates.csv and pareto.csv.

Plot scatter (objectives), Pareto front, hypervolume vs. iteration, and parallel-coordinates.

# Possible repo structure

mobo_ppa/
  main.py
  configs/
    default.yaml
  core/
    search_space.py         # bounds, types, constraints
    init_design.py          # LHS/Sobol
    models.py               # GP(s), transforms, noise handling
    acquisition.py          # qEHVI, qNEHVI, ParEGO, constraints
    optimize_acq.py         # inner-loop optimizers (LBFGS/ES)
    evaluator/
      toy_surrogate.py      # fast analytic PPA
      eda_wrapper.py        # real-flow adapter (cmd, parse, timeout)
    loop.py                 # BO iterate/refit/schedule/stop
    pareto.py               # non-domination, hypervolume
  ui/
    live_stats.py           # tail logs; show hypervolume curve
    final_plots.py          # Pareto scatter, attainment, parallel coords
  runs/
    2025-09-30_.../         # progress.csv, candidates.csv, pareto.csv, plots

## Possible config yaml

seed: 0
budget:
  init_evals: 16
  batch_size: 4
  max_iters: 20
objectives: [power, delay]   # optionally: [power, delay, area]
constraints:
  - type: hard
    name: timing_ok
    expr: "slack >= 0"
search_space:
  size: {type: continuous, low: 0.1, high: 1.0}
  vdd:  {type: continuous, low: 0.7, high: 1.1}
  cell: {type: categorical, values: [LVT, SVT, HVT]}
model:
  kind: gp_independent        # or: gp_multioutput
  noise: inferred             # or: fixed
acquisition:
  kind: qNEHVI                # options: qEHVI | ParEGO
  constraints: ["timing_ok"]
optimizer:
  inner: lbfgs                # fallback: evolutionary for mixed spaces
  restarts: 10
evaluator:
  backend: toy_surrogate      # or: eda_wrapper
  parallel: true
logging:
  dir: runs/
  live_ui: true
plots:
  parallel_coords: true
stop:
  hv_plateau_window: 5
  hv_delta_tol: 0.01

# Key components (what each part does)

search_space.py: Encodes variable types, normalizes to [0,1], masks invalid regions, defines constraint callbacks.

init_design.py: Generates Sobol/LHS seeds; ensures categorical balance and feasibility filtering.

models.py:

Builds standardized inputs/outputs; Y-transform (e.g., log for power).

Fits GP(s) with ARD; caches hyperparameters; supports incremental refits.

acquisition.py:

q(E)HVI: computes (noisy) expected hypervolume improvement; supports constraints via feasibility probabilities.

ParEGO: draws random weights, scalarizes objectives (Tchebycheff), then uses standard EI.

Handles batching (fantasies) and mixed variables (continuous relax + rounding).

optimize_acq.py: Multi-start gradient search in normalized space; retries with evolutionary search if gradients unreliable.

evaluator/eda_wrapper.py:

Launch script/tool (e.g., OpenROAD/Innovus), capture logs, parse PPA, set timing_ok.

Timeouts, retries, caching by (x, corner).

loop.py: Orchestrates: fit → propose → evaluate → update; tracks dominated hypervolume, iteration timing.

pareto.py: Fast non-domination filter (skyline) and dominated hypervolume (reference point chosen from worst-observed + margins).

ui/*:

Live chart: hypervolume vs. iteration, eval queue status.

Final charts: 2D/3D Pareto, empirical attainment function, parallel coordinates.

## Outputs (same as RL flow + a bit more)

progress.csv: iteration, batch, hypervolume, best known points.

candidates.csv: all evaluated designs with objectives, feasibility flags.

pareto.csv: final non-dominated set (the designs to carry forward).

Plots: Pareto scatter, hypervolume curve, attainment plot, parallel-coords.

## Implementation notes & options

Library choices (pick one path):

PyTorch BoTorch/Ax: native qEHVI/qNEHVI, constraints, batching, mixed-var workarounds.

Trieste (TF) or Emukit: simpler APIs; less feature-rich for qNEHVI.

Constraints: Use Expected Feasible Hypervolume Improvement (multiply by feasibility probability) or inequality transforms.

Mixed variables: one-hot for categorical; or latent embedding GP for large categorical sets.

Parallelism: Use q-batch MOBO for N parallel EDA slots; fantasies handle pending evals.

Stopping: hypervolume plateau, fixed budget, or %improvement in best objective(s).

Reproducibility: fixed seeds, config-driven runs, deterministic kernels where practical.

## How MOBO vs RL compare (what to measure)

Sample efficiency: dominated hypervolume vs evaluations (MOBO typically strong with expensive evals).

Wall-clock: time per iteration (RL training vs BO model fit + acq optimize).

Stability: variance across seeds (MOBO usually lower).

Scalability: high-D, large categorical spaces (RL may scale better; MOBO needs kernels/embeddings).

Anytime behavior: MOBO yields usable Pareto early; RL needs full training pass.

Generalization: RL learns a policy mapping “preference → design”; MOBO yields explicit designs for the explored frontier.
