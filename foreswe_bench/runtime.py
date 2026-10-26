"""Timed training / inference steps, environment capture, structured logging."""
import json
import logging
import os
import platform
import socket
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from .precision import autocast

LOG = logging.getLogger("foreswe_bench")


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return LOG


def env_info():
    p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    info = dict(host=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if p else None,
                gpu_mem_gib=p.total_memory / 2**30 if p else None, sm_count=p.multi_processor_count if p else None,
                cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING", "unset"),
                git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain")))
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,clocks.sm,power.limit,uuid",
                                     "--format=csv,noheader"], text=True).strip().splitlines()[0]
        info["driver"], info["max_sm_clock"], info["sm_clock_at_start"], info["power_limit"], info["gpu_uuid"] = \
            [s.strip() for s in q.split(",")]
    except Exception:
        pass
    return info


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(__file__), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {}
    q1, q3 = np.percentile(a, [25, 75])
    return dict(median=float(np.median(a)), mean=float(a.mean()), std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)), iqr=float(q3 - q1),
                min=float(a.min()), max=float(a.max()), n=int(a.size))


class Nvtx:
    enabled = False

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if Nvtx.enabled:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, *exc):
        if Nvtx.enabled:
            torch.cuda.nvtx.range_pop()


def forward_loss(model, ex, precision, loss_fn=nn.MSELoss()):
    with autocast(precision):
        out, rep = model(ex["daily"], ex["weekly"], ex["yearly"], ex["prompts"], ex["dist"], ex["ang"])
        loss = loss_fn(out.reshape(ex["daily"].shape[1], -1), ex["target"])
    return out, rep, loss


TRAIN_REGIONS = ("prep", "forward", "loss_item", "zero_grad", "backward", "optimizer")


def timed_train_step(model, opt, data, index, precision):
    """One production-order training step, every region bracketed by CUDA events.

    Order matches planB `train_epoch`: forward+loss -> loss.item() -> zero_grad -> backward -> step.
    """
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(TRAIN_REGIONS) + 1)]
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    ev[0].record()
    with Nvtx("train/prep"):
        ex = data.example(int(index))
    ev[1].record()
    with Nvtx("train/forward"):
        _, _, loss = forward_loss(model, ex, precision)
    ev[2].record()
    with Nvtx("train/loss_item"):
        loss_value = loss.item()
    ev[3].record()
    with Nvtx("train/zero_grad"):
        opt.zero_grad()
    ev[4].record()
    with Nvtx("train/backward"):
        loss.backward()
    ev[5].record()
    with Nvtx("train/optimizer"):
        opt.step()
    ev[6].record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - w0) * 1e3
    rec = {r: ev[i].elapsed_time(ev[i + 1]) for i, r in enumerate(TRAIN_REGIONS)}
    rec["step_gpu"] = ev[0].elapsed_time(ev[-1])
    rec["step_wall"] = wall
    rec["loss"] = loss_value
    return rec


def timed_infer_step(model, data, index, precision):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    with torch.no_grad():
        ev[0].record()
        with Nvtx("infer/prep"):
            ex = data.example(int(index))
        ev[1].record()
        with Nvtx("infer/forward"):
            with autocast(precision):
                out, rep = model(ex["daily"], ex["weekly"], ex["yearly"], ex["prompts"], ex["dist"], ex["ang"])
        ev[2].record()
        with Nvtx("infer/d2h"):
            out.float().cpu(), rep.float().cpu()
        ev[3].record()
    torch.cuda.synchronize()
    return dict(prep=ev[0].elapsed_time(ev[1]), forward=ev[1].elapsed_time(ev[2]),
                d2h=ev[2].elapsed_time(ev[3]), step_wall=(time.perf_counter() - w0) * 1e3)


def write_json(obj, path):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
