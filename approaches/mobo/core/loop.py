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
    hv_val = hv.item() if hasattr(hv, 'item') else float(hv)
    hv_hist.append(hv_val)
    idx0, _ = non_dominated(Y_feas)
    pareto_history.append(Y_feas[idx0].detach().clone())
    log_progress(-1, hv_val, X0.shape[0], 0.0, 0.0)

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
        hv_new_val = hv_new.item() if hasattr(hv_new, 'item') else float(hv_new)
        dhv = hv_new_val - hv_hist[-1]
        hv_hist.append(hv_new_val)
        idx, _ = non_dominated(Y_feas)
        pareto_history.append(Y_feas[idx].detach().clone())

        dt = time.time() - it_t0
        log_progress(it, hv_new_val, X_all_t.shape[0], dt, dhv)

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