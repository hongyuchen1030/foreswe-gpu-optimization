"""Per-kernel registers / spills / shared memory / occupancy of each v2 preset, TF32 and FP32 builds.

    python -m benchmarks.attention_resources
Writes benchmark_results/final/launch_resources.json (fwd + bwd launched under each precision mode).
"""
import torch

from benchmarks.attention_tiling import occupancy
from foreswe_bench import attn_check
from foreswe_bench import geo_attention_v2 as v2
from foreswe_bench import paths
from foreswe_bench.precision import matmul_flags
from foreswe_bench.runtime import env_info, setup_logging, write_json

OUT = paths.REPO / "engineering_audit" / "optimization" / "fused_attention" / "benchmark_results" / "final"


def main():
    log = setup_logging()
    inp = attn_check.case_inputs("near_one_hot", B=1, H=16, N=512, D=192)
    res = dict(env=env_info(), tiles=v2.TUNED_TILES, presets={})
    for preset in v2.PRESETS:
        v2.use_preset(preset)
        res["presets"][preset] = {}
        for p in ("tf32", "fp32"):
            v2.LAUNCH_INFO.clear()
            x = [t.clone().requires_grad_(i in (0, 1, 2, 5, 6, 7)) for i, t in enumerate(inp)]
            with matmul_flags(p):
                v2.fused_geo_attention_v2(*x).sum().backward()
            torch.cuda.synchronize()
            res["presets"][preset][p] = {k: dict(v, **occupancy(v)) for k, v in v2.LAUNCH_INFO.items()}
            for k, v in res["presets"][preset][p].items():
                log.info("%s %s %s: regs %s spills %s smem %s occ %.4f (%s)", preset, p, k, v["n_regs"], v["n_spills"],
                         v["shared_bytes"], v["theoretical_occupancy"], v["limiter"])
    v2.use_preset("accurate_experimental")
    write_json(res, OUT / "launch_resources.json")


if __name__ == "__main__":
    main()
