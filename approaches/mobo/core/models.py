import torch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

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