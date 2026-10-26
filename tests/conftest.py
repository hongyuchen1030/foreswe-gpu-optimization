import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "fore_swe"))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (run on a Perlmutter compute node)")
    config.addinivalue_line("markers", "realdata: needs the real-data cache ($FSW_CACHE_DIR)")


def pytest_collection_modifyitems(config, items):
    import torch
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def small_model_factory():
    """Real production classes at reduced width/depth (CPU-friendly)."""
    from foreswe_bench.model import build_model, load_namespace

    def make(seed=0, device="cpu"):
        ns = load_namespace("planB24", device)
        return ns, build_model(ns, "planB24", seed=seed, device=device, model_dim=64, nhead=4, enc_layers_count=1)
    return make


def small_batch(n_stations=24, device="cpu", seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    return dict(daily=torch.rand(30, n_stations, 19, generator=g).to(device),
                weekly=torch.rand(30, n_stations, 19, generator=g).to(device),
                yearly=torch.rand(5, n_stations, 19, generator=g).to(device),
                prompts=torch.rand(1, n_stations, 1536, generator=g).to(device),
                dist=(torch.rand(n_stations, n_stations, generator=g) * 1500).to(device),
                ang=(torch.rand(n_stations, n_stations, generator=g) * 20).to(device),
                target=(torch.rand(n_stations, 24, generator=g) * 1000).to(device))
