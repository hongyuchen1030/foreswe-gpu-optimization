"""Verbatim model construction from the historical ForeSWE training scripts.

The training scripts are monolithic (data loading + training at import time), so they cannot be
imported. Instead the `CrossVarTransformer` / `SWETransformer` classes and the per-sample
`get_inputs`/`get_outputs` helpers are extracted with `ast` and exec'd in an isolated namespace.
"""
import ast
import math
import re

import numpy as np
import torch
import torch.nn as nn

from . import paths

HELPERS = {"planB24": ("get_inputs", "get_outputs"), "weekly4": ("getInputs", "getOutputs"),
           "daily10": ("getInputs", "getOutputs")}

# Per-variant configuration, copied from the production scripts (see engineering_audit docs).
VARIANTS = {
    "planB24": dict(horizon=24, step=7, windows=dict(daily=(30, 1), weekly=(30, 7), yearly=(5, 365)),
                    lr=5e-6, weight_decay=1e-4),
    "weekly4": dict(horizon=4, step=7, windows=dict(daily=(30, 1), weekly=(30, 7), yearly=(5, 365)),
                    lr=5e-6, weight_decay=1e-4),
    "daily10": dict(horizon=10, step=1, windows=dict(daily=(1, 1), weekly=(120, 7), yearly=(5, 365)),
                    lr=5e-4, weight_decay=1e-4),
}

# CrossVarTransformer.encoder_layers are constructed but never called in forward().
DEAD_PARAM_RE = re.compile(r"^historical_(daily|monthly|yearly)_att\.encoder_layers\.")


def extract_source(source_path, names):
    """Return an ast.Module holding only the named top-level classes/functions of source_path."""
    with open(source_path) as f:
        tree = ast.parse(f.read(), filename=str(source_path))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    if missing:
        raise RuntimeError("{} not found in {}".format(sorted(missing), source_path))
    return ast.Module(body=nodes, type_ignores=[])


def load_namespace(variant="planB24", device="cuda"):
    """Exec the production classes + helpers; returns the namespace dict.

    `device` is the module-level global the production classes use for a few `.to(device)` calls.
    `model_inputs` / `model_outputs` must be set on the namespace before the helpers are called.
    """
    import sys
    if str(paths.FORE_SWE_DIR) not in sys.path:
        sys.path.insert(0, str(paths.FORE_SWE_DIR))
    from custom_encoder_model import EncoderLayer as CustomEncoderLayer

    ns = dict(torch=torch, nn=nn, np=np, math=math, device=torch.device(device),
              CustomEncoderLayer=CustomEncoderLayer, train_loc_indices=list(range(512)))
    names = ("CrossVarTransformer", "SWETransformer") + HELPERS[variant]
    src = paths.SOURCES[variant]
    exec(compile(extract_source(src, names), str(src), "exec"), ns)
    return ns


def build_model(ns, variant="planB24", seed=0, device="cuda", **overrides):
    """Construct SWETransformer exactly as production does, with a seeded CPU initialisation.

    Construction on CPU makes the initial weights independent of GPU RNG state and identical
    across precision/optimizer configurations.
    """
    kw = dict(model_dim=1024, prompt_dim=1536, n_output_heads=VARIANTS[variant]["horizon"], nhead=16)
    kw.update(overrides)
    devices_state = ns["device"]
    ns["device"] = torch.device("cpu")  # variable-embedding init uses the module-global device
    try:
        torch.manual_seed(seed)
        model = ns["SWETransformer"](**kw)
    finally:
        ns["device"] = devices_state
    return model.to(device)


def load_checkpoint_weights(model, path):
    """Load `swe_model_dict` from a historical checkpoint (read-only)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(ckpt["swe_model_dict"])
    meta = {k: ckpt[k] for k in ckpt if k not in ("swe_model_dict", "optimizer_dict")}
    del ckpt
    return meta


def is_dead(name):
    return bool(DEAD_PARAM_RE.match(name))


def live_named_parameters(model):
    """Parameters that participate in forward(); excludes the never-called CrossVar encoder layers."""
    return [(n, p) for n, p in model.named_parameters() if not is_dead(n)]


def param_group(name):
    """Coarse layer group used for layer-wise error reports."""
    m = re.match(r"^historical_(daily|monthly|yearly)_att\.", name)
    if m:
        return "crossvar_" + {"daily": "daily", "monthly": "weekly", "yearly": "yearly"}[m.group(1)]
    m = re.match(r"^encoder_layers\.(\d+)\.(self_attn|feed_forward|norm\d)", name)
    if m:
        return "encoder.L{}.{}".format(m.group(1), {"self_attn": "attn", "feed_forward": "ffn"}.get(m.group(2), "norm"))
    if name.startswith("output_layers."):
        return "output_mlp"
    if name.startswith("final_output."):
        return "final_head"
    return "other"
