"""Precision modes and optimizer implementations for the E0-E5 experiment matrix."""
import contextlib

import torch

PRECISIONS = ("fp32", "tf32", "bf16")
OPTIMIZERS = ("foreach", "fused")
EXPERIMENTS = {
    "E0": ("fp32", "foreach"), "E1": ("tf32", "foreach"), "E2": ("bf16", "foreach"),
    "E3": ("fp32", "fused"), "E4": ("tf32", "fused"), "E5": ("bf16", "fused"),
}


def set_matmul_flags(precision):
    """Global math-mode flags. BF16 keeps TF32 off so any matmul left in FP32 stays true FP32."""
    tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


@contextlib.contextmanager
def matmul_flags(precision):
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision())
    set_matmul_flags(precision)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev[:2]
        torch.set_float32_matmul_precision(prev[2])


def autocast(precision, device_type="cuda"):
    """Forward/loss context. BF16 = mixed precision: FP32 master weights, grads and optimizer state."""
    if precision == "bf16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_optimizer(params, impl, lr=5e-6, weight_decay=1e-4):
    """Production AdamW hyper-parameters; only the GPU implementation differs."""
    if impl == "foreach":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, foreach=True, fused=False)
    if impl == "fused":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, foreach=False, fused=True)
    if impl == "single_tensor":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, foreach=False, fused=False)
    raise ValueError(impl)
