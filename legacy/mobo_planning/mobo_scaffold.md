# Possible Scaffold

### Outputs will land in runs/<timestamp>/:

progress.csv (iteration, n_evals, hypervolume)

candidates.csv (all designs & objectives)

pareto.png (scatter + Pareto curve)

What’s included

## BoTorch-based qNEHVI loop with Sobol init and GP surrogates

Toy PPA surrogate (power, delay, slack/feasibility)

CSV + plot outputs to match the RL demo’s artifacts

Clean modules you can swap out later (e.g., replace the toy surrogate with an EDA wrapper)

# Scaffold Code
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
from botorch.acquisition.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

class QNEHVI:
    def __init__(self, model, ref_point, Xn_obs, Yn_obs):
        # Determine box decomposition from *observations*
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Yn_obs)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        self.acqf = qNoisyExpectedHypervolumeImprovement(
            model=model.model,
            ref_point=ref_point.tolist(),
            X_baseline=Xn_obs,
            sampler=sampler,
            prune_baseline=True,
            marginalize_dim=None,
            cache_root=True,
        )

    def __call__(self, Xn):
        return self.acqf(Xn)
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


def export_csv(run_dir, X, Y, feas_mask, space_names):
    # candidates.csv
    df_c = pd.DataFrame({**{f"x_{n}": X[:,i].tolist() for i,n in enumerate(space_names)},
                         **{f"obj_{i}": Y[:,i].tolist() for i in range(Y.shape[1])},
                         "feasible": feas_mask.tolist()})
    df_c.to_csv(f"{run_dir}/candidates.csv", index=False)
PY

######## ui/final_plots.py ########
cat > mobo_ppa/ui/final_plots.py << 'PY'
import matplotlib.pyplot as plt

def plot_pareto(Y_all, pareto_Y, out_png):
    plt.figure()
    plt.scatter(Y_all[:,0], Y_all[:,1], alpha=0.4, label="candidates")
    plt.plot(pareto_Y[:,0], pareto_Y[:,1], marker='o', linestyle='-', label="pareto")
    plt.xlabel("Power (min)"); plt.ylabel("Delay (min)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
PY

######## loop.py ########
cat > mobo_ppa/core/loop.py << 'PY'
import os, time, yaml, math
import torch
import pandas as pd
from dataclasses import dataclass
from .search_space import SearchSpace
from .init_design import sobol_init
from .models import MOBOModel
from .acquisition import QNEHVI
from .optimize_acq import propose_batch
from .evaluator.toy_surrogate import eval_batch
from .pareto import non_dominated, dominated_hv, export_csv

@dataclass
class RunState:
    X: torch.Tensor
    Y: torch.Tensor
    feas: torch.Tensor


def run(cfg, run_dir):
    torch.manual_seed(int(cfg.get("seed",0)))
    os.makedirs(run_dir, exist_ok=True)

    space = SearchSpace(cfg)
    bcfg = cfg["budget"]; ocfg = cfg["optimizer"]
    q = int(ocfg.get("q", 4))

    # Init
    X0 = sobol_init(bcfg["init_evals"], space.bounds)
    Y0, feas0, aux0 = eval_batch(X0)

    # Normalize for modeling
    Xn = space.normalize(X0)
    Yn = (Y0 - Y0.mean(0)) / Y0.std(0)  # simple standardize

    model = MOBOModel().fit(Xn, Yn)

    # Reference point: a bit worse than worst observed (since we minimize)
    worst = Y0.max(0).values + 0.1 * (Y0.max(0).values - Y0.min(0).values + 1e-6)
    ref_point = worst

    X_all, Y_all, feas_all = [X0], [Y0], [feas0]

    # BO loop
    for it in range(cfg["budget"]["max_iters"]):
        acq = QNEHVI(model, ref_point, Xn, Yn)
        Cn = propose_batch(acq.acqf, space.bounds, q=q, restarts=ocfg["restarts"], raw_samples=ocfg["raw_samples"])
        C  = space.unnormalize(Cn)
        Yc, feasc, aux = eval_batch(C)

        # Update data
        X_all.append(C); Y_all.append(Yc); feas_all.append(feasc)
        X_all_t = torch.cat(X_all, dim=0)
        Y_all_t = torch.cat(Y_all, dim=0)
        feas_all_t = torch.cat(feas_all, dim=0)

        # Refit
        Xn = space.normalize(X_all_t)
        Yn = (Y_all_t - Y_all_t.mean(0)) / Y_all_t.std(0)
        model.fit(Xn, Yn)

        # Logging
        hv = dominated_hv(Y_all_t, ref_point)
        pd.DataFrame({"iter":[it], "hv":[hv.item()], "n_evals":[X_all_t.shape[0]]}).to_csv(f"{run_dir}/progress.csv", mode='a', header=not os.path.exists(f"{run_dir}/progress.csv"), index=False)

    # Pareto + exports
    idx, mask = non_dominated(Y_all_t)
    pareto_Y = Y_all_t[idx]
    export_csv(run_dir, X_all_t, Y_all_t, feas_all_t, space.names)

    return X_all_t, Y_all_t, pareto_Y
PY

######## main.py ########
cat > main.py << 'PY'
import os, sys, time, yaml
from datetime import datetime
import torch
from mobo_ppa.core.loop import run
from mobo_ppa.ui.final_plots import plot_pareto


def main():
    cfg_path = os.environ.get("MOBO_CFG", "mobo_ppa/configs/default.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["logging"]["dir"], stamp)
    X, Y, pareto_Y = run(cfg, run_dir)
    plot_pareto(Y.numpy(), pareto_Y.numpy(), os.path.join(run_dir, "pareto.png"))
    print(f"Done. Outputs in {run_dir}")

if __name__ == "__main__":
    main()
PY

chmod +x mobo_ppa_scaffold.sh

echo "Scaffold written. Next steps:\n1) python -m venv .venv && source .venv/bin/activate\n2) pip install -r requirements.txt\n3) python main.py\nOutputs: runs/<stamp>/{progress.csv,candidates.csv,pareto.png} (extend pareto.csv export in pareto.py if desired)."

