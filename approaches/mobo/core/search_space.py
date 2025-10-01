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