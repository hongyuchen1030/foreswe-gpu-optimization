"""Path resolution. Everything is derived from the repo location or FSW_* environment variables."""
import os
from pathlib import Path

REPO = Path(os.environ.get("FSW_REPO", Path(__file__).resolve().parents[1]))
FORE_SWE_DIR = REPO / "fore_swe"
DATA_DIR = Path(os.environ.get("FSW_DATA_DIR", REPO / "data"))
SCRATCH_ROOT = Path(os.environ.get("FSW_SCRATCH_ROOT", "/pscratch/sd/h/hyvchen/foreswe-gpu-optimization"))
CACHE_DIR = Path(os.environ.get("FSW_CACHE_DIR", SCRATCH_ROOT / "cache"))
ARTIFACT_DIR = Path(os.environ.get("FSW_ARTIFACT_DIR", SCRATCH_ROOT / "artifacts"))
RESULTS_DIR = REPO / "engineering_audit" / "optimization" / "benchmark_results"
PROFILING_SUMMARY_DIR = REPO / "engineering_audit" / "optimization" / "profiling_summaries"

SOURCES = {
    "planB24": REPO / "experiments" / "planB_24week" / "planB_24week_att_model.py",
    "weekly4": FORE_SWE_DIR / "forecasting_att_weekly_model.py",
    "daily10": FORE_SWE_DIR / "forecasting_att_daily_model.py",
}

CHECKPOINTS = {
    "planB24": os.environ.get(
        "FSW_CKPT_PLANB",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_planB_24week/checkpoints/planB_24week_att_model.pt"),
    "weekly4": os.environ.get(
        "FSW_CKPT_WEEKLY",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512/checkpoints/forecasting_modified_weekly_att_climax_model_gp_act.pt"),
    "daily10": os.environ.get(
        "FSW_CKPT_DAILY",
        "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting/checkpoints/forecasting_modified_daily_att_climax_model_gp_act.pt"),
}

DATA_CACHE = CACHE_DIR / "planB24_real_data_cache.pt"
