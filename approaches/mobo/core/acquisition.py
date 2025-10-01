import torch
from typing import Optional
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler
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