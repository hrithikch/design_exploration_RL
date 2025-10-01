#!/usr/bin/env bash
# MOBO PPA scaffold — creates a minimal BoTorch-based project with the same outputs as the RL demo
# Usage: bash mobo_ppa_scaffold.sh && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python main.py

set -euo pipefail
mkdir -p mobo_ppa/{configs,core/evaluator,ui,runs}

######## requirements ########
cat > requirements.txt << 'REQ'
botorch>=0.10.0
gpytorch>=1.13
torch>=2.2
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
pyyaml>=6.0.1
tqdm>=4.66
REQ

######## config ########
cat > mobo_ppa/configs/default.yaml << 'YAML'
seed: 0
budget:
  init_evals: 12
  batch_size: 4
  max_iters: 8
objectives: [power, delay]
constraints:
  - name: timing_ok
    type: hard
    expr: "slack >= 0"
search_space:
  size: {type: continuous, low: 0.1, high: 1.0}
  vdd:  {type: continuous, low: 0.7, high: 1.1}
model:
  kind: gp_independent
acquisition:
  kind: qNEHVI
optimizer:
  restarts: 10
  raw_samples: 128
  q: 4
  num_candidates: 4
evaluator:
  backend: toy_surrogate
logging:
  dir: runs
  live_ui: false
plots:
  parallel_coords: false
stop:
  hv_plateau_window: 3
  hv_delta_tol: 0.01
YAML

######## search_space.py ########
cat > mobo_ppa/core/search_space.py << 'PY'
from dataclasses import dataclass
import torch

@dataclass
class Bounds:
    lower: torch.Tensor
    upper: torch.Tensor

class SearchSpace:
    def __init__(self, cfg):
        items = []
        for k, spec in cfg["search_space"].items():
            if spec["type"] != "continuous":
                raise NotImplementedError("This scaffold handles continuous vars; extend for mixed spaces.")
            items.append((k, float(spec["low"]), float(spec["high"])) )
        self.names = [k for k,_,_ in items]
        lows  = torch.tensor([lo for _,lo,_ in items])
        highs = torch.tensor([hi for _,_,hi in items])
        self.bounds = Bounds(lows, highs)

    def normalize(self, X):
        lo, hi = self.bounds.lower, self.bounds.upper
        return (X - lo) / (hi - lo)

    def unnormalize(self, Xn):
        lo, hi = self.bounds.lower, self.bounds.upper
        return lo + Xn * (hi - lo)

    @property
    def dim(self):
        return len(self.names)
PY

######## init_design.py ########
cat > mobo_ppa/core/init_design.py << 'PY'
import torch
from torch.quasirandom import SobolEngine

def sobol_init(n, bounds):
    d = len(bounds.lower)
    eng = SobolEngine(dimension=d, scramble=True)
    Xn = eng.draw(n)
    X  = bounds.lower + Xn * (bounds.upper - bounds.lower)
    return X
PY

######## evaluator/toy_surrogate.py ########
cat > mobo_ppa/core/evaluator/toy_surrogate.py << 'PY'
import torch

def eval_batch(X):
    """Analytic toy PPA: power ~ a*size^2*vdd^2; delay ~ b/(eps + size*vdd).
    Slack >= 0 if (size*vdd)>=0.35.
    X: (q,d) in *original* units: columns [size, vdd]
    Returns: Y (q,2) objectives to *minimize*, feas (q,) bool, aux dict with slack
    """
    size = X[:,0]
    vdd  = X[:,1]
    a, b, eps = 1.0, 1.0, 1e-3
    power = a * size.pow(2) * vdd.pow(2)
    delay = b / (eps + size * vdd)
    slack = size * vdd - 0.35
    feas = slack >= 0.0
    Y = torch.stack([power, delay], dim=-1)
    return Y, feas, {"slack": slack}
PY

######## models.py ########
cat > mobo_ppa/core/models.py << 'PY'
import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch import settings
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

settings.debug._set_state(False)

class MOBOModel:
    def __init__(self):
        self.model = None
        self.outcome_transform = None

    def fit(self, Xn, Yn):
        # One GP per objective for simplicity
        gps = []
        for i in range(Yn.shape[-1]):
            gp = SingleTaskGP(Xn, Yn[..., i:i+1])
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gps.append(gp)
        self.model = ModelListGP(*gps)
        return self

    def posterior(self, Xn):
        return self.model.posterior(Xn)

    @property
    def num_outputs(self):
        return len(self.model.models)
PY

######## acquisition.py ########
cat > mobo_ppa/core/acquisition.py << 'PY'
import torch
from typing import Optional
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
    qExpectedImprovement,
)
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.objective import GenericMCObjective


class QNEHVI:
    """Noisy Expected Hypervolume Improvement with optional feasibility weighting.

    If a `feas_prob_fn` is provided, we multiply the acquisition by an estimate
    of feasibility probability for candidates (simple scalar weight in [0,1]).
    This keeps the scaffold light while demonstrating constraint-aware MOBO.
    """

    def __init__(self, model, ref_point, Xn_obs, Yn_obs, feas_prob_fn: Optional[callable] = None):
        self.ref_point = ref_point
        self.feas_prob_fn = feas_prob_fn
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        self.base_acqf = qNoisyExpectedHypervolumeImprovement(
            model=model.model,
            ref_point=ref_point.tolist(),
            X_baseline=Xn_obs,
            sampler=sampler,
            prune_baseline=True,
            marginalize_dim=None,
            cache_root=True,
        )

    def __call__(self, Xn):
        val = self.base_acqf(Xn)
        if self.feas_prob_fn is not None:
            w = self.feas_prob_fn(Xn)  # shape (q,) or (q,1)
            val = val * w.mean()  # simple scalar down-weighting
        return val


class ParEGO:
    """ParEGO-style scalarization with MC qEI and a Tchebycheff transform.

    Draws a random weight vector per call (or use a fixed `weights` tensor) and
    optimizes qEI over a scalarized objective built from the multi-output model.
    """

    def __init__(self, model, Yn_obs, weights: Optional[torch.Tensor] = None):
        self.model = model
        self.Ymin = Yn_obs.min(dim=0).values  # for Tchebycheff ref
        if weights is None:
            w = torch.rand(Yn_obs.shape[-1])
            w = w / w.sum()
            self.weights = w
        else:
            self.weights = weights

        def tcheby(obj_samples):
            # obj_samples: (..., m) minimized
            diff = (obj_samples - self.Ymin)
            tch = torch.max(self.weights * diff, dim=-1).values
            # Add small L1 term to break ties
            return tch + 0.05 * torch.sum(self.weights * diff, dim=-1)

        self.objective = GenericMCObjective(lambda Z: -tcheby(-Z))  # keep in "min" convention
        self.best_f = None  # computed lazily from posterior on observed X

    def build(self, Xn_obs):
        # Compute best_f under the scalarized objective at observed points
        with torch.no_grad():
            post = self.model.posterior(Xn_obs)
            samples = post.rsample(torch.Size([128]))[..., 0, :]  # (S,N,m)
            s = self.objective(samples)  # (S,N)
            self.best_f = s.max(dim=-1).values.mean()  # MC estimate of best value
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        self.acqf = qExpectedImprovement(
            model=self.model.model,  # ModelListGP is supported with GenericMCObjective
            best_f=self.best_f.item(),
            sampler=sampler,
            objective=self.objective,
        )
        return self.acqf
PY

######## optimize_acq.py ########
cat > mobo_ppa/core/optimize_acq.py << 'PY'
import torch
from botorch.optim import optimize_acqf

def propose_batch(acqf, bounds01, q=4, restarts=10, raw_samples=128):
    cand, _ = optimize_acqf(
        acq_function=acqf,
        bounds=torch.stack([torch.zeros_like(bounds01.lower), torch.ones_like(bounds01.upper)]),
        q=q,
        num_restarts=restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return cand
PY

######## pareto.py ########
cat > mobo_ppa/core/pareto.py << 'PY'
import torch
import pandas as pd
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume


def non_dominated(Y):
    mask = is_non_dominated(Y)
    idx = torch.arange(Y.shape[0])[mask]
    return idx, mask


def dominated_hv(Y, ref_point):
    hv = Hypervolume(ref_point=ref_point)
    return hv.compute(Y)


def export_csv(run_dir, X, Y, feas_mask, space_names, pareto_idx):
    # candidates.csv
    df_c = pd.DataFrame({**{f"x_{n}": X[:, i].tolist() for i, n in enumerate(space_names)},
                         **{f"obj_{i}": Y[:, i].tolist() for i in range(Y.shape[1])},
                         "feasible": feas_mask.tolist()})
    df_c.to_csv(f"{run_dir}/candidates.csv", index=False)

    # pareto.csv (only feasible + non-dominated subset indices provided)
    P = pareto_idx
    df_p = pd.DataFrame({**{f"x_{n}": X[P, i].tolist() for i, n in enumerate(space_names)},
                         **{f"obj_{i}": Y[P, i].tolist() for i in range(Y.shape[1])}})
    df_p.to_csv(f"{run_dir}/pareto.csv", index=False)
PY

######## ui/final_plots.py ########
cat > mobo_ppa/ui/final_plots.py << 'PY'
import numpy as np
import matplotlib.pyplot as plt


def plot_pareto(Y_all, pareto_Y, out_png):
    plt.figure()
    plt.scatter(Y_all[:, 0], Y_all[:, 1], alpha=0.4, label="candidates")
    # Sort pareto by first objective for a clean line
    P = pareto_Y[np.argsort(pareto_Y[:, 0])]
    plt.plot(P[:, 0], P[:, 1], marker='o', linestyle='-', label="pareto")
    plt.xlabel("Power (min)")
    plt.ylabel("Delay (min)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)


def plot_attainment(pareto_history, out_png):
    """Empirical attainment: show several intermediate Pareto fronts.
    pareto_history: list of ndarray fronts (k_i, m), m=2
    """
    plt.figure()
    for i, F in enumerate(pareto_history):
        F = F[np.argsort(F[:, 0])]
        alpha = 0.2 + 0.6 * (i + 1) / len(pareto_history)
        label = "iter %d" % i if i in {0, len(pareto_history) - 1} else None
        plt.plot(F[:, 0], F[:, 1], marker='.', linestyle='-', alpha=alpha, label=label)
    plt.xlabel("Power (min)"); plt.ylabel("Delay (min)")
    if len(pareto_history) >= 2:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
PY

######## loop.py ########
cat > mobo_ppa/core/loop.py << 'PY'
import os, time, yaml
import torch
import pandas as pd
from dataclasses import dataclass
from .search_space import SearchSpace
from .init_design import sobol_init
from .models import MOBOModel
from .acquisition import QNEHVI, ParEGO
from .optimize_acq import propose_batch
from .evaluator.toy_surrogate import eval_batch
from .pareto import non_dominated, dominated_hv, export_csv

@dataclass
class RunState:
    X: torch.Tensor
    Y: torch.Tensor
    feas: torch.Tensor


def _feas_only(Y, feas):
    return Y[feas]


def run(cfg, run_dir):
    torch.manual_seed(int(cfg.get("seed", 0)))
    os.makedirs(run_dir, exist_ok=True)

    space = SearchSpace(cfg)
    bcfg = cfg["budget"]; ocfg = cfg["optimizer"]
    q = int(ocfg.get("q", 4))

    # Init
    t0 = time.time()
    X0 = sobol_init(bcfg["init_evals"], space.bounds)
    Y0, feas0, aux0 = eval_batch(X0)

    # Normalize for modeling
    Xn = space.normalize(X0)
    Ymu, Ystd = Y0.mean(0), Y0.std(0).clamp_min(1e-6)
    Yn = (Y0 - Ymu) / Ystd

    model = MOBOModel().fit(Xn, Yn)

    # Reference point: fixed per run from initial observations (minimization)
    worst = Y0.max(0).values + 0.1 * (Y0.max(0).values - Y0.min(0).values + 1e-6)
    ref_point = worst

    X_all, Y_all, feas_all = [X0], [Y0], [feas0]

    hv_hist = []
    pareto_history = []

    def log_progress(iter_idx, hv, n_evals, dt, dhv):
        row = {"iter": iter_idx, "hv": hv, "delta_hv": dhv, "n_evals": n_evals, "seconds": dt}
        pd.DataFrame([row]).to_csv(
            f"{run_dir}/progress.csv",
            mode='a', header=not os.path.exists(f"{run_dir}/progress.csv"), index=False
        )

    # initial HV / Pareto on feasible subset
    Y_all_t = torch.cat(Y_all, dim=0)
    feas_all_t = torch.cat(feas_all, dim=0)
    Y_feas = _feas_only(Y_all_t, feas_all_t)
    hv = dominated_hv(Y_feas, ref_point)
    hv_hist.append(hv.item())
    idx0, _ = non_dominated(Y_feas)
    pareto_history.append(Y_feas[idx0].detach().clone())
    log_progress(-1, hv.item(), X0.shape[0], 0.0, 0.0)

    # BO loop
    for it in range(cfg["budget"]["max_iters"]):
        it_t0 = time.time()

        # Choose acquisition
        acq_kind = cfg.get("acquisition", {}).get("kind", "qNEHVI")
        if acq_kind.lower() == "parego":
            acq = ParEGO(model, Yn_obs=Yn)
            acqf = acq.build(Xn_obs=Xn)
        else:
            acq = QNEHVI(model, ref_point, Xn_obs=Xn, Yn_obs=Yn, feas_prob_fn=None)
            acqf = acq.base_acqf

        Cn = propose_batch(acqf, space.bounds, q=q, restarts=ocfg["restarts"], raw_samples=ocfg["raw_samples"])
        C = space.unnormalize(Cn)
        Yc, feasc, aux = eval_batch(C)

        # Update data
        X_all.append(C); Y_all.append(Yc); feas_all.append(feasc)
        X_all_t = torch.cat(X_all, dim=0)
        Y_all_t = torch.cat(Y_all, dim=0)
        feas_all_t = torch.cat(feas_all, dim=0)

        # Refit (standardize with current stats)
        Xn = space.normalize(X_all_t)
        Ymu, Ystd = Y_all_t.mean(0), Y_all_t.std(0).clamp_min(1e-6)
        Yn = (Y_all_t - Ymu) / Ystd
        model.fit(Xn, Yn)

        # Logging & HV on feasible
        Y_feas = _feas_only(Y_all_t, feas_all_t)
        hv_new = dominated_hv(Y_feas, ref_point)
        dhv = hv_new.item() - hv_hist[-1]
        hv_hist.append(hv_new.item())
        idx, _ = non_dominated(Y_feas)
        pareto_history.append(Y_feas[idx].detach().clone())

        dt = time.time() - it_t0
        log_progress(it, hv_new.item(), X_all_t.shape[0], dt, dhv)

        # Plateau stopping
        win = int(cfg["stop"].get("hv_plateau_window", 3))
        tol = float(cfg["stop"].get("hv_delta_tol", 0.01))
        if len(hv_hist) > win:
            recent = torch.tensor(hv_hist[-win:])
            if (recent[-1] - recent[0]).abs().item() < tol:
                break

    # Pareto + exports (feasible only)
    Y_feas = _feas_only(Y_all_t, feas_all_t)
    idx, mask = non_dominated(Y_feas)
    pareto_Y = Y_feas[idx]

    # Map feasible indices back to original indexing for export
    feas_idx = torch.nonzero(feas_all_t, as_tuple=False).squeeze(-1)
    pareto_global_idx = feas_idx[idx]
    export_csv(run_dir, X_all_t, Y_all_t, feas_all_t, space.names, pareto_global_idx)

    return X_all_t, Y_all_t, pareto_Y, pareto_history
PY

######## main.py ########
cat > main.py << 'PY'
import os, yaml
from datetime import datetime
from mobo_ppa.core.loop import run
from mobo_ppa.ui.final_plots import plot_pareto, plot_attainment


def main():
    cfg_path = os.environ.get("MOBO_CFG", "mobo_ppa/configs/default.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["logging"]["dir"], stamp)
    X, Y, pareto_Y, pareto_hist = run(cfg, run_dir)
    plot_pareto(Y.numpy(), pareto_Y.numpy(), os.path.join(run_dir, "pareto.png"))
    if len(pareto_hist) > 1:
        import numpy as np
        ph = [p.numpy() for p in pareto_hist]
        plot_attainment(ph, os.path.join(run_dir, "attainment.png"))
    print(f"Done. Outputs in {run_dir}")

if __name__ == "__main__":
    main()
PY

chmod +x mobo_ppa_scaffold.sh

echo "Scaffold written. Next steps:\n1) python -m venv .venv && source .venv/bin/activate\n2) pip install -r requirements.txt\n3) python main.py\nOutputs: runs/<stamp>/{progress.csv,candidates.csv,pareto.png} (extend pareto.csv export in pareto.py if desired)."
------------------------------------------
# What’s new (at a glance)

Constraint-aware qNEHVI hook: acquisition now supports simple feasibility weighting (stubbed via a callback) so you can demonstrate constraint handling without extra GP plumbing.

ParEGO toggle: added a ParEGO acquisition path using a Tchebycheff scalarization + MC qEI. Switch by setting acquisition.kind: ParEGO.

Proper pareto.csv export: in addition to candidates.csv, we now write the final non-dominated feasible subset to pareto.csv.

Attainment plot: added attainment.png showing intermediate Pareto fronts over iterations.

Rolling HV plateau stop: early stopping triggers when hypervolume gain is below a threshold over a sliding window.

Better logging: progress.csv now includes delta_hv and seconds per iteration.

File-level diffs (highlights)

core/acquisition.py

New QNEHVI class with optional feas_prob_fn (constraint weighting).

New ParEGO class using GenericMCObjective + qEI (Tchebycheff scalarization).

core/loop.py

Fixed reference point per run from initial observations.

Tracks feasible-only hypervolume, Pareto history, ΔHV, and wall time.

Implements plateau stopping via stop.hv_plateau_window and stop.hv_delta_tol.

Returns pareto_history for the attainment plot and exports feasible Pareto indices.

core/pareto.py

Writes both candidates.csv and pareto.csv.

ui/final_plots.py

Keeps plot_pareto() and adds plot_attainment() to visualize front evolution.

main.py

Calls both plotting functions; saves pareto.png and attainment.png.

How to use the new bits

Switch acquisition:

acquisition:
  kind: qNEHVI   # or: ParEGO


Stopping and logging (already wired):

stop:
  hv_plateau_window: 5
  hv_delta_tol: 0.01


Run flow (unchanged):

bash mobo_ppa_scaffold.sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py


Outputs now include:

progress.csv (iter, hv, delta_hv, n_evals, seconds)

candidates.csv, pareto.csv

pareto.png, attainment.png