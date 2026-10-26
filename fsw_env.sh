# Source me on a Perlmutter compute node:  source fsw_env.sh
# Sets up the software stack and every path the repo needs; nothing points at the legacy source tree.
FSW_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FSW_REPO

module load "${FSW_TORCH_MODULE:-pytorch/2.6.0}" > /dev/null 2>&1 || echo "warning: could not load ${FSW_TORCH_MODULE:-pytorch/2.6.0}" >&2

# shellcheck source=configs/external_paths.env
source "$FSW_REPO/configs/external_paths.env"

export FSW_DATA_DIR="${FSW_DATA_DIR:-$FSW_REPO/data}"
# New outputs (runs, benchmark traces, data caches) live on pscratch, never in the repo or legacy dirs.
export FSW_SCRATCH_ROOT="${FSW_SCRATCH_ROOT:-/pscratch/sd/h/hyvchen/foreswe-gpu-optimization}"
export FSW_RUN_ROOT="${FSW_RUN_ROOT:-$FSW_SCRATCH_ROOT/runs}"
export FSW_CACHE_DIR="${FSW_CACHE_DIR:-$FSW_SCRATCH_ROOT/cache}"

export HF_HOME="${HF_HOME:-$FSW_HF_CACHE/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$FSW_HF_CACHE/huggingface/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$FSW_HF_CACHE/sentence_transformers}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export PYTHONPATH="$FSW_REPO:$FSW_REPO/fore_swe${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR_OVERRIDE:-/pscratch/sd/h/hyvchen/tmp}"
export PYTHONUNBUFFERED=1
mkdir -p "$FSW_RUN_ROOT" "$FSW_CACHE_DIR" "$TMPDIR"
