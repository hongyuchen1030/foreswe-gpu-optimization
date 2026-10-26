# ForeSWE Weekly Replication Plan

This experiment isolates the released weekly ForeSWE pipeline while reusing the successful full-domain daily ForeSWE replication artifacts.

Paper-aligned weekly horizons:
- 7 days
- 14 days
- 21 days
- 28 days

The 28-day horizon is the anchor reference to the paper's weekly ForeSWE setup.

Key implementation choice:
- the released weekly attention script expects `2014_daily_predict.pkl` through `2018_daily_predict.pkl`
- the reproduced daily ForeSWE pipeline produced `2014_daily_gp_predict.pkl` through `2018_daily_gp_predict.pkl`
- this experiment provides experiment-local symlinks so the weekly code can read the existing full-domain daily GP predictions without touching the original third-party tree

Experiment root:
- `/global/homes/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512`

Runtime root:
- `/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512`

Copied weekly scripts:
- `scripts/forecasting_att_weekly_model.py`
- `scripts/forecasting_weekly_gp_model.py`
- `scripts/extended_dl_forecasting.py`
- `scripts/custom_encoder_model.py`

Dry-run checker:
- `scripts/check_weekly_pipeline.py`

Interactive debug runner:
- `scripts/run_weekly_att_interactive.sh`

10-minute watcher:
- `scripts/monitor_weekly_att_debug.sh`

Batch attention script:
- `slurm/run_weekly_att_gpu.sbatch`
