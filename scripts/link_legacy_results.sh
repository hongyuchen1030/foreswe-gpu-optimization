#!/usr/bin/env bash
# Opt-in, read-only analysis links: makes the historical result pickles visible at the paths the
# migrated plotting/analysis scripts expect (results/actual, results/predict). The links are
# git-ignored. Checkpoints are deliberately NOT linked into the repo: training scripts write
# checkpoints, and a link would let a new run overwrite a historical one. Use $FSW_CKPT_* instead.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/configs/external_paths.env"

link_dir() {  # link every file of <src dir> into <repo-relative dst dir> unless a real file exists
  local src="$1" dst="$REPO/$2"
  mkdir -p "$dst"
  for f in "$src"/*; do
    [[ -e "$f" ]] || continue
    local t="$dst/$(basename "$f")"
    if [[ -e "$t" && ! -L "$t" ]]; then continue; fi
    ln -sfn "$f" "$t"
  done
}
link_dir "$FSW_LEGACY_HOME/thirdparties/SWE-Forecasting/results/actual" results/actual
link_dir "$FSW_LEGACY_SCRATCH/thirdparties/SWE-Forecasting/results/predict" results/predict
echo "linked legacy result pickles into results/actual and results/predict (git-ignored)"
