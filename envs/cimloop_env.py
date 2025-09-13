# cimloop_env.py (skeleton)
import json, subprocess, shlex, numpy as np, gymnasium as gym
from gymnasium import spaces

class CimloopEnv(gym.Env):
    """
    Single-step environment that calls cimloop (CLI or Python)
    to get raw PPA metrics. You scalarize them with weights self.w.
    """
    def __init__(self, cmd_template: str, param_bounds=((0,1),(0,1))):
        super().__init__()
        self.cmd_template = cmd_template  # e.g., "python run_cimloop.py --size {size} --vdd {vdd} --json_out out.json"
        self.param_bounds = np.array(param_bounds, dtype=float)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)  # weights
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        # running min/max for normalization (bootstrap with a few randoms offline)
        self.mins = {"power": 1e9, "delay": 1e9}
        self.maxs = {"power": 0.0, "delay": 0.0}
        self.w = np.array([0.5,0.5], dtype=np.float32)

    def _denorm(self, a01):
        # map [0,1] -> actual bounded range
        lows = self.param_bounds[:,0]; highs = self.param_bounds[:,1]
        return np.clip(lows + (highs - lows) * a01, lows, highs)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        w = np.random.rand(2); w /= (w.sum() + 1e-12)
        self.w = w.astype(np.float32)
        return self.w.copy(), {}

    def _call_cimloop(self, size, vdd):
        # Example: call your script that runs cimloop and writes JSON with {"power":..,"delay":..,"area":..}
        cmd = self.cmd_template.format(size=size, vdd=vdd)
        subprocess.run(shlex.split(cmd), check=True)
        with open("out.json","r") as f:
            result = json.load(f)
        # If cimloop logs "Cycles" and "Energy" only, map to perf/power proxies here.
        return result  # dict with keys you need

    def _norm01(self, k, v):
        self.mins[k] = min(self.mins[k], v)
        self.maxs[k] = max(self.maxs[k], v)
        return (v - self.mins[k]) / (self.maxs[k] - self.mins[k] + 1e-12)

    def step(self, action):
        size, vdd = self._denorm(action)
        res = self._call_cimloop(float(size), float(vdd))
        # Expect res to contain "power" and "delay" (or derive delay from max freq / slack)
        p = float(res["power"]); d = float(res["delay"])
        pz = self._norm01("power", p); dz = self._norm01("delay", d)
        reward = -(self.w[0]*pz + self.w[1]*dz)
        info = {"size": float(size), "vdd": float(vdd), "power": p, "delay": d}
        return self.w.copy(), reward, True, False, info
