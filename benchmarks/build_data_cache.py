"""Build the real-data tensor cache (run once per pscratch lifetime, on a compute node).

    python -m benchmarks.build_data_cache [--out PATH]
"""
import argparse
import json

from foreswe_bench import data, paths
from foreswe_bench.runtime import setup_logging


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(paths.DATA_CACHE))
    a = p.parse_args()
    log = setup_logging()
    meta = data.build_cache(a.out)
    log.info("cache written to %s", a.out)
    print(json.dumps(meta, indent=1, default=str))
    if not all(meta["checks_vs_historical_planB"].values()):
        raise SystemExit("cache does not match the historical Plan B run: {}".format(meta["checks_vs_historical_planB"]))


if __name__ == "__main__":
    main()
