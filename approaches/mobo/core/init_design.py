import torch
from torch.quasirandom import SobolEngine

def sobol_init(n, bounds):
    d = len(bounds.lower)
    eng = SobolEngine(dimension=d, scramble=True)
    Xn = eng.draw(n)
    X  = bounds.lower + Xn * (bounds.upper - bounds.lower)
    return X