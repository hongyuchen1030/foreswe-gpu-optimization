"""Model inputs: production-shaped synthetic data and a cached copy of the real ForeSWE tensors.

Both sources keep the whole dataset resident on the GPU and build each training example with the
production helpers (`get_inputs` / `get_outputs`), so the per-step data path is the real one.
"""
import os
import time
import warnings

import numpy as np
import torch

from . import paths
from .model import VARIANTS

N_DAYS, N_STATIONS, N_FEATURES, PROMPT_DIM = 10305, 512, 19, 1536
TEST_YEARS = (2015, 2016, 2017, 2018, 2019)
# Reference values recorded by the historical Plan B run (experiments/planB_24week/planB_shape_check.txt)
PLANB_FEB1_ISSUE = {2015: 8511, 2016: 8876, 2017: 9241, 2018: 9606, 2019: 9971}
PLANB_TRAIN_COUNT = 3297
# Production test-station list (identical in daily/weekly/Plan B scripts).
TEST_STATION_INDICES = sorted([
    43, 66, 338, 463, 115, 105, 83, 306, 481, 21, 185, 136, 110, 406, 408, 132,
    120, 197, 7, 301, 375, 236, 269, 287, 49, 241, 212, 145, 264, 435, 73, 277, 364,
    438, 372, 368, 243, 493, 497, 14, 32, 433, 313, 213, 42, 475, 160, 190, 281, 285,
    316, 461, 154, 424, 310, 340, 179, 167, 193, 466, 56, 158, 365, 492, 432, 415, 68,
    409, 76, 148, 267, 227, 91, 189, 472, 162, 147, 395, 113, 2, 271, 357, 347, 335, 507,
    465, 57, 19, 442, 381, 398, 79, 337, 501, 41, 339, 331, 63, 359, 232, 323, 258])


class Dataset:
    """GPU-resident tensors + production example builder."""

    def __init__(self, ns, variant, model_inputs, model_outputs, prompts, distances, angularities,
                 train_indices, test_issue, device, meta=None):
        self.ns, self.variant, self.cfg = ns, variant, VARIANTS[variant]
        self.device = torch.device(device)
        ns["model_inputs"] = model_inputs.to(self.device, torch.float32)
        ns["model_outputs"] = model_outputs.to(self.device, torch.float32)
        self.prompts = prompts.to(self.device, torch.float32)
        self.distances = np.asarray(distances)        # float64 numpy, converted per step as in production
        self.angularities = np.asarray(angularities)
        self.train_indices = np.asarray(train_indices)
        self.test_issue = dict(test_issue)
        self.all_indices = list(range(N_STATIONS))
        self.meta = meta or {}

    def example(self, index):
        """Exactly the per-sample work of Plan B `get_example` (planB_24week_att_model.py:297-313)."""
        get_inputs = self.ns[{"planB24": "get_inputs"}.get(self.variant, "getInputs")]
        get_outputs = self.ns[{"planB24": "get_outputs"}.get(self.variant, "getOutputs")]
        (dw, ds), (ww, ws), (yw, ys) = (self.cfg["windows"][k] for k in ("daily", "weekly", "yearly"))
        idx = self.all_indices
        daily = get_inputs(index, idx, window=dw, step_size=ds)
        weekly = get_inputs(index, idx, window=ww, step_size=ws)
        yearly = get_inputs(index, idx, window=yw, step_size=ys)
        target = get_outputs(index, idx, window=self.cfg["horizon"], step_size=self.cfg["step"])
        dist = torch.from_numpy(self.distances[idx, :][:, idx]).to(device=self.device, dtype=torch.float32)
        ang = torch.from_numpy(self.angularities[idx, :][:, idx]).to(device=self.device, dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # production's torch.tensor(tensor) copy-construct warning
            return dict(daily=torch.tensor(daily).to(device=self.device, dtype=torch.float32),
                        weekly=torch.tensor(weekly).to(device=self.device, dtype=torch.float32),
                        yearly=torch.tensor(yearly).to(device=self.device, dtype=torch.float32),
                        target=torch.tensor(target).to(device=self.device, dtype=torch.float32),
                        prompts=self.prompts, dist=dist, ang=ang)


def synthetic(ns, variant="planB24", device="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = np.arange(2000, 2000 + PLANB_TRAIN_COUNT)
    return Dataset(ns, variant,
                   model_inputs=torch.rand(N_DAYS, N_STATIONS, N_FEATURES, generator=g),
                   model_outputs=torch.rand(N_DAYS, N_STATIONS, 1, generator=g) * 1000,
                   prompts=torch.rand(1, N_STATIONS, PROMPT_DIM, generator=g),
                   distances=rng.uniform(0, 1500, (N_STATIONS, N_STATIONS)),
                   angularities=rng.uniform(0, 20, (N_STATIONS, N_STATIONS)),
                   train_indices=train, test_issue=PLANB_FEB1_ISSUE, device=device,
                   meta=dict(source="synthetic", seed=seed))


def real(ns, variant="planB24", device="cuda", cache_path=None):
    cache_path = cache_path or paths.DATA_CACHE
    if not os.path.exists(cache_path):
        raise FileNotFoundError("{} missing: run `python -m benchmarks.build_data_cache` on a compute node".format(cache_path))
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    horizon, step = VARIANTS[variant]["horizon"], VARIANTS[variant]["step"]
    train = np.asarray(c["training_data_indices_raw"])[: -(horizon * step)]
    return Dataset(ns, variant, c["model_inputs"], c["model_outputs"], c["all_prompts"], c["distances"],
                   c["angularities"], train, c["test_issue_feb1"], device,
                   meta=dict(source="real", cache=str(cache_path), built=c["meta"]))


def build_cache(out_path=None):
    """Run the production DataLoader once (CPU; pandas + sentence-transformer prompts) and cache tensors.

    Replicates planB_24week_att_model.py:103-142 call-for-call. Must run on a compute node.
    """
    out_path = out_path or paths.DATA_CACHE
    os.environ.setdefault("BASE_URL", str(paths.REPO))  # extended_dl_forecasting reads {BASE_URL}/data
    import sys
    sys.path.insert(0, str(paths.FORE_SWE_DIR))
    import extended_dl_forecasting as DL

    t0 = time.time()
    dl = DL.DataLoader(test_indices=TEST_STATION_INDICES, temp_window=7)
    distances, angularities = dl.getDistanceAndAngularity()
    dl.cleanDataJunk()
    dl.normalizeData()
    dl.addDayInfo()
    dl.prepareInputsAndOutputs()
    model_inputs, model_outputs = dl.getModelInputsAndOutputs()
    dl.completeInputsAndOutputs()
    train_raw = np.array(dl.getTrainingDataIndices())
    testing = dl.getTestingDataIndices()
    train_loc = dl.getTrainingIndices()
    _, _, all_prompts = dl.getModelPrompts()
    max_swe, min_swe = dl.getMaxMinSWE()
    all_dates = np.array(dl.all_dates)
    feb1 = {}
    for year, idxs in zip(TEST_YEARS, testing):
        m = [int(i) for i in idxs if str(all_dates[i])[:10] == "{}-02-01".format(year)]
        assert len(m) == 1, (year, m)
        feb1[year] = m[0]
    cache = dict(model_inputs=torch.from_numpy(np.asarray(model_inputs, dtype=np.float32)),
                 model_outputs=torch.from_numpy(np.asarray(model_outputs, dtype=np.float32)),
                 all_prompts=torch.from_numpy(np.asarray(all_prompts, dtype=np.float32)),
                 distances=np.asarray(distances), angularities=np.asarray(angularities),
                 training_data_indices_raw=train_raw, testing_indices=[np.asarray(t) for t in testing],
                 test_issue_feb1=feb1, train_loc_indices=np.asarray(train_loc),
                 all_dates=[str(d) for d in all_dates], max_swe=float(max_swe), min_swe=float(min_swe),
                 meta=dict(built_unix=time.time(), build_seconds=time.time() - t0,
                           data_dir=str(paths.DATA_DIR), prompt_mode="sentence-transformer all-MiniLM-L6-v2"))
    # cross-check against the historical Plan B run
    checks = dict(model_inputs_shape=tuple(cache["model_inputs"].shape) == (N_DAYS, N_STATIONS, N_FEATURES),
                  planB_train_count=len(train_raw[: -(24 * 7)]) == PLANB_TRAIN_COUNT,
                  feb1_indices=feb1 == PLANB_FEB1_ISSUE)
    cache["meta"]["checks_vs_historical_planB"] = checks
    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    torch.save(cache, out_path)
    return cache["meta"]
