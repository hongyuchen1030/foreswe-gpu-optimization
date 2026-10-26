#!/usr/bin/env bash
# Thin venv on top of NERSC's pytorch/2.6.0 module (system site-packages reused: torch, numpy,
# pandas, sklearn, transformers, triton, pytest). Adds only what the module lacks, at the versions
# pinned in requirement_files/. Lives on pscratch: if the purge removes it, just re-run this script.
#   bash scripts/setup_venv.sh          # on a compute node (needs outbound network for pip)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/fsw_env.sh"
VENV="${FSW_VENV:-$FSW_SCRATCH_ROOT/venv-pytorch2.6.0}"
if [[ ! -x "$VENV/bin/python" ]]; then
  python -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade-strategy only-if-needed \
  "sentence-transformers==3.2.1" "hydroeval==0.1.0" "gpytorch==1.13"
"$VENV/bin/python" - <<'EOF'
import torch, sentence_transformers, gpytorch, hydroeval
print("venv ok: torch", torch.__version__, "| sentence-transformers", sentence_transformers.__version__,
      "| gpytorch", gpytorch.__version__)
EOF
echo "export FSW_VENV=$VENV   # fsw_env.sh activates it when present"
