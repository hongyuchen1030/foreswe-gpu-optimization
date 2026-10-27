"""Kernel-name classification: categories and Tensor-Core vs SIMT GEMMs."""
import re

# Tensor-Core GEMM kernels on sm80: cuBLAS/cuBLASLt/CUTLASS naming (s1688 = TF32 HMMA, s16816 = BF16/FP16
# HMMA, xmma/tensorop/hmma/wmma families). SIMT FP32 kernels are named ampere_sgemm_*/gemv*.
TENSOR_CORE_RE = re.compile(r"s1688|s16816|h1688|h16816|i16832|tensorop|xmma|hmma|wmma|_tf32|tf32_|bf16.*gemm|gemm.*bf16"
                            r"|cutlass.*(?:80|90)_.*gemm", re.I)
GEMM_RE = re.compile(r"gemm|gemv|xmma|cutlass|tensorop|dot_kernel|splitKreduce", re.I)

CATEGORIES = [
    ("GEMM (Tensor Core)", lambda n: bool(GEMM_RE.search(n)) and bool(TENSOR_CORE_RE.search(n))),
    ("GEMM (SIMT FP32)", lambda n: bool(GEMM_RE.search(n))),
    ("AdamW (multi_tensor_apply: foreach or fused)", lambda n: "multi_tensor_apply" in n),
    ("AdamW fused", lambda n: "fused_adam" in n.lower() or "FusedAdam" in n or "_fused_adamw" in n.lower()),
    ("index gather", lambda n: "index_elementwise" in n or "indexing_backward" in n or "index_put" in n),
    ("softmax", lambda n: "softmax" in n.lower()),
    ("layer_norm", lambda n: "layer_norm" in n.lower() or "LayerNorm" in n),
    ("reductions", lambda n: "reduce_kernel" in n),
    ("dtype casts (to/copy)", lambda n: "direct_copy" in n or "copy_kernel" in n or "_to_copy" in n),
    ("concat/stack", lambda n: "CatArrayBatched" in n),
    ("memcpy/memset", lambda n: n.startswith(("Memcpy", "Memset", "[CUDA memcpy", "[CUDA memset"))),
    ("other elementwise", lambda n: True),
]


def categorize(name):
    for cat, fn in CATEGORIES:
        if fn(name):
            return cat
    return "other"


def is_tensor_core_gemm(name):
    return bool(GEMM_RE.search(name)) and bool(TENSOR_CORE_RE.search(name))
