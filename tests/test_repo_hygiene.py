"""The repo is self-contained: no runnable code points at the legacy source tree; no large files in git."""
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_SOURCE = re.compile(r"Snow-Predication-at-Sierra/(thirdparties/SWE-Forecasting|experiments/)")
CODE_EXT = (".py", ".sh", ".sbatch")
# Files that legitimately mention legacy locations: provenance/migration tooling and read-only
# historical output stores on pscratch.
ALLOWED = {"scripts/migrate_from_legacy.sh", "scripts/link_legacy_results.sh", "scripts/recover_sierra_sources.py",
           "configs/external_paths.env", "foreswe_bench/paths.py"}


def tracked(*args):
    out = subprocess.check_output(["git", "-C", REPO, "ls-files", *args], text=True)
    return [l for l in out.splitlines() if l]


def test_no_legacy_source_paths_in_runnable_code():
    bad = []
    for f in tracked():
        if not f.endswith(CODE_EXT) or f in ALLOWED or "/recovered_from_transcripts/" in f:
            continue
        with open(os.path.join(REPO, f), errors="ignore") as fh:
            for i, line in enumerate(fh, 1):
                if LEGACY_SOURCE.search(line) and "/pscratch/" not in line:
                    bad.append("{}:{}: {}".format(f, i, line.strip()[:120]))
    assert not bad, "legacy home source paths remain:\n" + "\n".join(bad)


def test_no_large_files_tracked():
    limit = 2 * 1024 * 1024
    big = [f for f in tracked() if not f.startswith("data/") and os.path.exists(os.path.join(REPO, f))
           and os.path.getsize(os.path.join(REPO, f)) > limit]
    assert not big, big


def test_no_checkpoints_or_traces_tracked():
    bad = [f for f in tracked() if f.endswith((".nsys-rep", ".sqlite", ".ncu-rep")) or
           (f.endswith((".pt", ".pth")) and not f.endswith("example_checkpoint.pt"))]
    assert not bad, bad
