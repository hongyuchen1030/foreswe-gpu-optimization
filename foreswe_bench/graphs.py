"""E4 training-step variants: eager (G0), torch.compile (G1/G2), manual CUDA Graphs (G3).

Every variant exposes the same interface so the benchmark and correctness harnesses are shared:

    step = make_step(cfg, model, opt_factory)        # cfg: see CONFIGS
    step.setup(example)        # compile / warm up / capture; restores pristine weights+optimizer state
    loss = step.train_step(example)   # end-to-end: copy inputs -> run -> read loss (host sync)
    step.run_staged()          # GPU-only region on already staged inputs (no host prep, no sync)
    out = step.infer(example)  # forward only, eval/no_grad

E4 semantics are preserved: TF32 on, fused AdamW (lr 5e-6, wd 1e-4), MSE loss, production phase order.
"""
import contextlib
import time

import torch
import torch.nn as nn

from .runtime import Nvtx

INPUT_KEYS = ("daily", "weekly", "yearly", "prompts", "dist", "ang", "target")

# name -> dict(kind, options). G4 is filled in after G1-G3 are measured (see COMPILE_AND_CUDAGRAPH_REPORT.md).
CONFIGS = {
    "G0": dict(kind="eager", capturable=False),
    "G0c": dict(kind="eager", capturable=True),            # ungraphed control for G3's optimizer setting
    "G1": dict(kind="compile", mode="default", scope="model"),
    "G1L": dict(kind="compile", mode="default", scope="layers"),
    "G2": dict(kind="compile", mode="reduce-overhead", scope="model"),
    "G2L": dict(kind="compile", mode="reduce-overhead", scope="layers"),
    "G3f": dict(kind="graph", region="fwd_bwd"),           # forward+backward captured, optimizer eager
    "G3": dict(kind="graph", region="full"),               # forward+backward+fused AdamW captured
    "G4": dict(kind="compile_graph", mode="default", scope="model"),  # inductor kernels inside a manual full-step graph
}


def loss_fn():
    return nn.MSELoss()


def forward_loss(model, b):
    out, rep = model(b["daily"], b["weekly"], b["yearly"], b["prompts"], b["dist"], b["ang"])
    loss = loss_fn()(out.reshape(b["daily"].shape[1], -1), b["target"])
    return out, loss


def make_optimizer(model, capturable, lr=5e-6, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                             foreach=False, fused=True, capturable=capturable)


def snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def restore(model, opt, snap):
    """Put parameters back (same storage, so captured addresses stay valid) and reset AdamW to its
    fresh state: zero moments and step=0 is exactly what lazy state initialisation would create."""
    with torch.no_grad():
        for p, s in zip(model.parameters(), snap):
            p.copy_(s)
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    v.zero_()


class StaticBatch:
    """Persistent device buffers; captured graphs read from these addresses."""

    def __init__(self, example):
        self.buf = {k: example[k].detach().clone() for k in INPUT_KEYS}

    def load(self, example):
        for k in INPUT_KEYS:
            self.buf[k].copy_(example[k], non_blocking=True)
        return self.buf


class EagerStep:
    def __init__(self, model, capturable=False):
        self.model = model
        self.opt = make_optimizer(model, capturable)
        self.setup_s = 0.0
        self.static = None

    def setup(self, example):
        self.static = StaticBatch(example)

    def _step(self, b):
        with Nvtx("train/forward"):
            _, loss = forward_loss(self.model, b)
        with Nvtx("train/zero_grad"):
            self.opt.zero_grad()
        with Nvtx("train/backward"):
            loss.backward()
        with Nvtx("train/optimizer"):
            self.opt.step()
        return loss

    def run_staged(self):
        return self._step(self.static.buf)

    def train_step(self, example):
        # production order reads the loss before backward; reading it after the step is the same value
        loss = self._step(example)
        return loss.item()

    def phases(self, b):
        """Event-timed forward / backward / optimizer (eager-able variants only)."""
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        _, loss = forward_loss(self.model, b)
        ev[1].record()
        self.opt.zero_grad()
        loss.backward()
        ev[2].record()
        self.opt.step()
        ev[3].record()
        torch.cuda.synchronize()
        return dict(forward=ev[0].elapsed_time(ev[1]), backward=ev[1].elapsed_time(ev[2]),
                    optimizer=ev[2].elapsed_time(ev[3]))

    def infer(self, example):
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(example["daily"], example["weekly"], example["yearly"], example["prompts"],
                                example["dist"], example["ang"])
        self.model.train(was_training)
        return out


class CompiledStep(EagerStep):
    """torch.compile of the whole model or of each repeated encoder layer; optimizer stays fused eager."""

    def __init__(self, model, mode="default", scope="model"):
        super().__init__(model, capturable=False)
        self.mode, self.scope = mode, scope
        if scope == "model":
            self.fmodel = torch.compile(model, mode=mode)
        elif scope == "layers":
            for layer in model.encoder_layers:
                layer.compile(mode=mode)
            self.fmodel = model
        else:
            raise ValueError(scope)
        self.cudagraph_trees = mode in ("reduce-overhead", "max-autotune")

    def _mark(self):
        if self.cudagraph_trees:
            torch.compiler.cudagraph_mark_step_begin()

    def setup(self, example, warmup=3):
        """Compile (forward + AOT backward) on a throw-away copy of the state, then restore it."""
        self.static = StaticBatch(example)
        snap = snapshot(self.model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(warmup):
            self._mark()
            _, loss = forward_loss(self.fmodel, self.static.buf)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        torch.cuda.synchronize()
        self.setup_s = time.perf_counter() - t0
        restore(self.model, self.opt, snap)
        del snap

    def _step(self, b):
        self._mark()
        with Nvtx("train/forward"):
            _, loss = forward_loss(self.fmodel, b)
        with Nvtx("train/zero_grad"):
            self.opt.zero_grad()
        with Nvtx("train/backward"):
            loss.backward()
        with Nvtx("train/optimizer"):
            self.opt.step()
        return loss

    def train_step(self, example):
        loss = self._step(example)
        return loss.item()

    def phases(self, b):
        self._mark()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        _, loss = forward_loss(self.fmodel, b)
        ev[1].record()
        self.opt.zero_grad()
        loss.backward()
        ev[2].record()
        self.opt.step()
        ev[3].record()
        torch.cuda.synchronize()
        return dict(forward=ev[0].elapsed_time(ev[1]), backward=ev[1].elapsed_time(ev[2]),
                    optimizer=ev[2].elapsed_time(ev[3]))

    def infer(self, example):
        was_training = self.model.training
        self.model.eval()
        self._mark()
        with torch.no_grad():
            out, _ = self.fmodel(example["daily"], example["weekly"], example["yearly"], example["prompts"],
                                 example["dist"], example["ang"])
        self.model.train(was_training)
        return out.clone()


class GraphStep:
    """Manual torch.cuda.CUDAGraph capture of the training step.

    region="fwd_bwd": forward + loss + backward captured; fused AdamW (capturable) runs eagerly.
    region="full":    forward + loss + backward + fused AdamW (capturable=True) captured.

    Grads are set to None before capture, so backward allocates them from the graph pool and every
    replay overwrites them (no accumulation, no stale grads). Dropout uses PyTorch's graph-safe Philox
    offsets, so each replay draws fresh masks. Warm-up runs on a side stream and its effect on the
    weights and AdamW state is undone afterwards (same storage => captured addresses stay valid).
    """

    def __init__(self, model, region="full", fmodel=None):
        self.model, self.region = model, region
        self.fmodel = fmodel or model
        self.opt = make_optimizer(model, capturable=True)
        self.graph = None
        self.infer_graph = None
        self.setup_s = 0.0

    def _body(self, b):
        _, loss = forward_loss(self.fmodel, b)
        loss.backward()
        if self.region == "full":
            self.opt.step()
        return loss

    def setup(self, example, warmup=3):
        self.static = StaticBatch(example)
        snap = snapshot(self.model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.opt.zero_grad(set_to_none=True)
                self._body(self.static.buf)
                if self.region != "full":
                    self.opt.step()
        torch.cuda.current_stream().wait_stream(s)
        self.opt.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_loss = self._body(self.static.buf)
        torch.cuda.synchronize()
        self.setup_s = time.perf_counter() - t0
        restore(self.model, self.opt, snap)
        del snap

    def run_staged(self):
        self.graph.replay()
        if self.region != "full":
            self.opt.step()
        return self.static_loss

    def train_step(self, example):
        self.static.load(example)
        self.run_staged()
        return self.static_loss.item()

    def phases(self, b):
        """Graph replay is one submission: report replay and (fwd_bwd region) the eager optimizer."""
        self.static.load(b)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ev[0].record()
        self.graph.replay()
        ev[1].record()
        if self.region != "full":
            self.opt.step()
        ev[2].record()
        torch.cuda.synchronize()
        if self.region == "full":
            return dict(replay_fwd_bwd_opt=ev[0].elapsed_time(ev[1]))
        return dict(replay_fwd_bwd=ev[0].elapsed_time(ev[1]), optimizer=ev[1].elapsed_time(ev[2]))

    def capture_inference(self):
        was_training = self.model.training
        self.model.eval()
        self.infer_static = StaticBatch(self.static.buf)
        b = self.infer_static.buf
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self.fmodel(b["daily"], b["weekly"], b["yearly"], b["prompts"], b["dist"], b["ang"])
        torch.cuda.current_stream().wait_stream(s)
        self.infer_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.infer_graph), torch.no_grad():
            self.infer_out, _ = self.fmodel(b["daily"], b["weekly"], b["yearly"], b["prompts"], b["dist"], b["ang"])
        self.model.train(was_training)

    def infer(self, example):
        if self.infer_graph is None:
            self.capture_inference()
        self.infer_static.load(example)
        self.infer_graph.replay()
        return self.infer_out


def make_step(name, model):
    c = CONFIGS[name]
    if c["kind"] == "eager":
        return EagerStep(model, capturable=c["capturable"])
    if c["kind"] == "compile":
        return CompiledStep(model, mode=c["mode"], scope=c["scope"])
    if c["kind"] == "graph":
        return GraphStep(model, region=c["region"])
    if c["kind"] == "compile_graph":
        # inductor-generated kernels (no inductor cudagraphs) captured inside the manual full-step graph
        fmodel = torch.compile(model, mode=c["mode"])
        return GraphStep(model, region="full", fmodel=fmodel)
    raise ValueError(name)


@contextlib.contextmanager
def dynamo_counters():
    """Snapshot of dynamo/inductor counters accumulated inside the block."""
    from torch._dynamo.utils import counters
    counters.clear()
    out = {}
    try:
        yield out
    finally:
        out.update({k: dict(v) for k, v in counters.items()})
