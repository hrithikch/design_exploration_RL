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