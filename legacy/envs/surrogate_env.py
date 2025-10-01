# surrogate_env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SurrogatePPAEnv(gym.Env):
    """
    Single-step, weight-conditioned PPA environment.
    Obs = [w_power, w_delay]  (weights sampled per episode; sum=1)
    Act = [size, vdd] in [0,1]
    Reward = - (w_pwr * power_norm + w_del * delay_norm)  # we minimize PPA
    Episode ends after one step -> propose design -> evaluate -> done.
    """
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        # Observation: weights (2D) in simplex
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        # Action: normalized params
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Constants for surrogate; choose so objectives have different curvature/scales
        self.a = 1.0
        self.b = 1.0
        self.eps = 1e-3

        # Precompute min/max for normalization on the [0,1]^2 box (coarse grid)
        grid = np.linspace(0, 1, 101)
        ss, vv = np.meshgrid(grid, grid)
        power = self.a * (ss**2) * (vv**2)
        delay = self.b / (self.eps + ss * vv)
        # Avoid infinities at 0 by small eps in denom (already handled)
        self.pwr_min, self.pwr_max = float(power.min()), float(power.max())
        self.dly_min, self.dly_max = float(delay.min()), float(delay.max())

        self.w = np.array([0.5, 0.5], dtype=np.float32)  # set on reset

    def _normalize(self, val, vmin, vmax):
        return float((val - vmin) / (vmax - vmin + 1e-12))

    def _evaluate(self, size, vdd):
        power = self.a * (size**2) * (vdd**2)
        delay = self.b / (self.eps + size * vdd)
        p_norm = self._normalize(power, self.pwr_min, self.pwr_max)
        d_norm = self._normalize(delay, self.dly_min, self.dly_max)
        return dict(power=power, delay=delay, p_norm=p_norm, d_norm=d_norm)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # sample a preference weight on the 2D simplex
        w = self.rng.random(2)
        w = w / (w.sum() + 1e-12)
        self.w = w.astype(np.float32)
        obs = self.w.copy()
        info = {}
        return obs, info

    def step(self, action):
        # clip to [0,1]
        size = float(np.clip(action[0], 0.0, 1.0))
        vdd  = float(np.clip(action[1], 0.0, 1.0))
        metrics = self._evaluate(size, vdd)
        # scalarized reward (negated because we minimize)
        reward = -(self.w[0] * metrics["p_norm"] + self.w[1] * metrics["d_norm"])
        terminated = True   # single-step episode
        truncated  = False
        obs = self.w.copy() # obs does not change in single-step
        info = {
            "size": size, "vdd": vdd,
            **metrics
        }
        return obs, float(reward), terminated, truncated, info
