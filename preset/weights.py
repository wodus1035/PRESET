"""Checkpoint discovery and loading."""

import os
import torch
from .model import ModelConfig, PRESET

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "preset_v1.0.pt")


def resolve_weights(path=None):
    """Order: explicit path > $PRESET_WEIGHTS > packaged preset_v1.0.pt."""
    p = path or os.environ.get("PRESET_WEIGHTS") or DEFAULT_WEIGHTS
    if not os.path.exists(p):
        raise FileNotFoundError(f"PRESET weights not found: {p} (set PRESET_WEIGHTS or pass weights=...)")
    return p


def load_model(path=None, device=None):
    """Returns (model in eval mode, device, checkpoint metadata)."""
    p = resolve_weights(path)
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(p, map_location=dev, weights_only=False)
    cfg = ModelConfig(**ck["mcfg"]) if "mcfg" in ck else ModelConfig()
    model = PRESET(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    # head kind from the checkpoint config ("bar" if absent)
    meta = {
        "weights": p,
        "step": ck.get("step"),
        "averaged_from": ck.get("averaged_from"),
        "config": ck.get("mcfg"),
        "head": cfg.head,
    }
    return model, dev, meta
