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