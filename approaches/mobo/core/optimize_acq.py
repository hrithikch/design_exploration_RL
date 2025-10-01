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