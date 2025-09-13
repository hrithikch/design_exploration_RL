#env_factory.py
import importlib, inspect

def make_env(cfg):
    env_cfg = cfg["env"]
    module_path = env_cfg["module"]
    class_name  = env_cfg["class"]
    params      = env_cfg.get("params", {}) or {}

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in params.items() if k in accepted}

    dropped = [k for k in params if k not in accepted]
    if dropped:
        print(f"[WARN] Dropping unsupported env params for {class_name}: {dropped}")

    return cls(**filtered)
