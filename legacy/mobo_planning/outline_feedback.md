What’s great

Clear objectives/variables/constraints, and a pipeline that matches standard MOBO practice.

Good choices on independent GPs, qNEHVI, and hypervolume-based stopping—these align with best practice for 2–3 objectives and expensive evals.

Output artifacts match your RL demo, so comparisons will be apples-to-apples.

Tighten these details

Reference point (HV)

Specify how you set it: e.g., ref = Y_max + 0.1*(Y_max − Y_min) per objective (since you minimize). Keep it fixed per run for comparability.

Standardization

Normalize X to [0,1]^d and standardize Y per objective (zero mean / unit variance) before GP fitting; store transforms for plotting/exports.

Constraints in acquisition

If you keep slack ≥ 0, incorporate it as a feasibility-weighted qNEHVI (multiply by P(feasible)). If using the toy slack, you can model a third GP for the constraint or compute feasibility analytically.

Batching & pending points

When proposing q=4, enable fantasies (the default in BoTorch’s qNEHVI path) so pending evaluations don’t distort the posterior.

Initialization size

With 2D continuous variables, 12–16 Sobol/LHS points is fine. If you add categorical/more dims later, bump to ~10*d as a rule of thumb.

Stopping rule

Your plateau rule is good; implement rolling ΔHV over hv_plateau_window and stop when median ΔHV < hv_delta_tol.

Logging

Log per-iteration: n_evals, ΔHV, best Y so far, wall time. You’ll want this for RL vs MOBO comparisons.

Reproducibility

Fix seeds for Sobol, GP inits, and acquisition restarts. Record them in progress.csv header or a run.yaml snapshot.

Nice-to-haves (quick wins)

ParEGO toggle in config for a scalarization baseline.

Parallel-coordinates plot for >2 objectives; keep off by default, as you noted.

Attainment plot (β% front) to visualize convergence across iterations.

Trust region heuristic if you later scale to higher-D (stabilizes proposals).

Small wording/structure nits

In “Batch Optimization,” say “optimize acquisition in normalized space (multi-start LBFGS; evolutionary fallback for mixed variables).”

In “Parallel Evaluation,” clarify: evaluations are on the true objective (surrogate or EDA)—the surrogate GP is only for proposing points.

In “Model Updates,” note: warm-start hyperparameters between fits to cut wall-clock.

Compatibility with your scaffold

Everything you listed matches the earlier scaffold (qNEHVI + Sobol init + CSV/plots). We could:

add constraint-aware qNEHVI wiring,

expose ParEGO as acquisition.kind: ParEGO,

and include pareto.csv export + attainment plot.