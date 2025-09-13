# ring_osc_env.py (quick drop-in)
import numpy as np, gymnasium as gym
from gymnasium import spaces

class RingOscEnv(gym.Env):
    def __init__(self, vth=0.2, k=1.0):
        super().__init__()
        self.vth, self.k = vth, k
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)  # weights
        self.action_space      = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)  # [size,vdd,cload]
        self.mins={"power":1e9,"delay":1e9}; self.maxs={"power":0.0,"delay":0.0}
        self.w = np.array([0.5,0.5], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        w = np.random.rand(2); w /= (w.sum()+1e-12)
        self.w = w.astype(np.float32)
        return self.w.copy(), {}

    def _norm01(self, k, v):
        self.mins[k]=min(self.mins[k],v); self.maxs[k]=max(self.maxs[k],v)
        return (v-self.mins[k])/(self.maxs[k]-self.mins[k]+1e-12)

    def step(self, action):
        size, vdd, cl = np.clip(action, 0, 1)
        f = self.k * max(vdd - self.vth, 1e-3) / max(cl*size + 1e-3, 1e-3)
        power = (cl + 0.2*size) * (vdd**2) * f
        delay = 1.0 / max(f, 1e-6)
        pz = self._norm01("power", power); dz = self._norm01("delay", delay)
        reward = -(self.w[0]*pz + self.w[1]*dz)
        info = {"size":float(size),"vdd":float(vdd),"cload":float(cl),
                "power":float(power),"delay":float(delay),"freq":float(f)}
        return self.w.copy(), float(reward), True, False, info
