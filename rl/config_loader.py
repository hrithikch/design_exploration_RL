#config_loader.py
import os
from copy import deepcopy

DEFAULT_CONFIG = {
    "training": {
        "policy": "MlpPolicy",
        "total_timesteps": 10000,
        "n_steps": 64,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "seed": None,
    },
    "env": {
        "module": "envs.surrogate_env",
        "class": "SurrogatePPAEnv",
        "params": {"a": 1.0, "b": 1.0, "eps": 1e-3, "seed": None},
    },
    "run": {"root": "runs"},
    "log": {
        "outputs": ["csv"],
        "precreate_progress_row": True,
        "files": {
            "progress": "progress.csv",
            "candidates": "candidates.csv",
            "pareto": "pareto.csv",
        },
    },
    "sweep": {"points": 61, "grid": None},
    "gui": {"refresh_ms": 500},
    "plot": {"figure_dpi": 120, "figure_size": [6.5, 4.3]},
}

def load_config(path: str):
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("Please `pip install pyyaml` to use YAML configs.") from e

    if not os.path.exists(path):
        with open(path, "w") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)
        print(f"[INFO] Wrote default config to {path}")
        return deepcopy(DEFAULT_CONFIG)

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return _merge_defaults(cfg, deepcopy(DEFAULT_CONFIG))

def _merge_defaults(cfg, defaults):
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            _merge_defaults(cfg[k], v)
    return cfg
