# Migration & provenance (2026-10-26)

This repository is the canonical working location for all ForeSWE work. It was created from the legacy
layout inside `Snow-Predication-at-Sierra` without modifying or deleting any legacy file.

## Git history

| Ref | Meaning |
|---|---|
| `upstream-393de07` (tag), `upstream-dev` | Upstream `https://github.com/Krishuthapa/SWE-Forecasting` @ `393de0750160b257eab83495848a75c089068b76` (2026-05-04), full upstream history preserved via `git clone` of the legacy checkout |
| `upstream-main` | Upstream `main` (`32bd4b8`), fetched from the legacy checkout's remote-tracking ref |
| `main` commit "Apply local reproduction edits" | The legacy checkout's uncommitted edits, verbatim (`att_daily_run.sh` base_url, `results/actual` pickle dumps) |
| `main` commit "Migrate all ForeSWE experiments…" | Verbatim copy (no edits) of every ForeSWE experiment folder, the engineering audit and design notes |
| `main` commit "Make paths repo-relative…" | Path-only edits: `git diff <that>~1 <that>` shows every changed line |
| later commits | New benchmark infrastructure and results |

No remote is configured (the clone's `origin` pointed at the legacy local folder and was removed).
Nothing has been pushed anywhere.

## Source mapping

| Repo path | Legacy source | Notes |
|---|---|---|
| `fore_swe/`, `raw_gp/`, `transformer_ar/`, `other_comparing_models/`, `data/`, `requirement_files/` | `Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting` (git) | upstream, with history; `data/` (632 MB CSVs) is tracked upstream |
| `engineering_audit/` (except `optimization/`) | `thirdparties/SWE-Forecasting/engineering_audit` (untracked there) | audit reports + first profiling suite |
| `experiments/original_daily/` | `experiments/foreswe_reproduction` (+ pscratch logs, legacy `outputs/` logs, small result pickles) | Fig. 3/4 reproduction scripts, SLURM, logs |
| `experiments/original_weekly/` | `experiments/foreswe_weekly_reproduction_full512` (+ `pscratch/` logs) | |
| `experiments/planA_stitching/` | `experiments/fig6_weekly_stitching` | Fig. 6 stitched-window plots |
| `experiments/planB_24week/` | `/pscratch/.../experiments/foreswe_planB_24week` | **only copy was on pscratch** (purge-exposed); code, logs, figures, small pickles |
| `experiments/sierra_subset/` | `/pscratch/.../experiments/foreswe_sierra_station` (outputs/logs) | source code had been deleted from disk; see below |
| `experiments/domain_sensitivity/D2_sierra_plus_500km/` | `experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km` (+ `pscratch/`) | 326 MB `data_subset/` kept external |
| `experiments/bias_diagnosis/` | `experiments/foreswe_bias_diagnosis` | |
| `docs/legacy/` | `docs/foreswe_memo.md`, `docs/ForeSWE_T0_Architecture_Mapping.md` | |

Full file lists: `docs/migration/copied_files.tsv` (343 files) and
`docs/migration/external_artifacts_manifest.tsv` (136 items left in place: size, mtime, reason).
Re-run: `bash scripts/migrate_from_legacy.sh` (idempotent).

### Sierra 43-station source recovery

`experiments/foreswe_sierra_station/scripts/` no longer exists anywhere on disk. The Codex `apply_patch`
payloads that created and edited it survive in `~/.codex/sessions`. `scripts/recover_sierra_sources.py`
extracted 24 patches and rebuilt 15 files in `experiments/sierra_subset/recovered_from_transcripts/files/`:
all "Add File" scripts verbatim, plus the three core scripts (`forecasting_att_daily_model.py`,
`extended_dl_forecasting.py`, `forecasting_daily_gp_model.py`) by replaying their edit hunks onto the
upstream files they were copied from (every hunk applied). One edit to a Markdown note failed.
Treat these as **best-effort reconstructions**: edits made outside `apply_patch` (shell `cp`/`sed`)
cannot be recovered. `RECOVERY_LOG.txt` lists every step.

## Path changes (portability)

Legacy source root `…/Snow-Predication-at-Sierra/thirdparties/SWE-Forecasting` → repo root;
legacy home experiment folders → `experiments/<new name>`. New outputs never go to the repo or to a
historical folder: Plan B, D2, weekly and daily launchers default to `$FSW_RUN_ROOT/<name>`
(`/pscratch/sd/h/hyvchen/foreswe-gpu-optimization/runs`). Historical pscratch result stores remain as
read-only inputs. `tests/test_repo_hygiene.py` enforces that no runnable code refers to the legacy tree.

## External artifacts still required

| Artifact | Location | Size | Needed for |
|---|---|---:|---|
| Plan B checkpoint | `/pscratch/.../experiments/foreswe_planB_24week/checkpoints/planB_24week_att_model.pt` | 5.9 GB | trained-weight numerics, inference |
| Daily checkpoint | `/pscratch/.../thirdparties/SWE-Forecasting/checkpoints/forecasting_modified_daily_att_climax_model_gp_act.pt` | 5.1 GB | daily inference |
| Daily pre-scratch checkpoint (older) | home `experiments/foreswe_reproduction/archive_pre_scratch_attention_20260506_144156/checkpoints/` | 5.1 GB | provenance only |
| Weekly checkpoint | `/pscratch/.../experiments/foreswe_weekly_reproduction_full512/checkpoints/` | 5.9 GB | weekly inference |
| Sierra-43 checkpoint | `/pscratch/.../experiments/foreswe_sierra_station/checkpoints/` | 5.1 GB | Sierra inference |
| D2 checkpoint | `/pscratch/.../experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/checkpoints/` | 5.1 GB | D2 inference |
| GP-head input/output tensors (`*_for_gp_act.pt`) | pscratch `outputs/` of each experiment | 30–163 MB each | rerunning GP heads without the attention stage |
| Result pickles > 1 MiB | legacy `results/{actual,predict}` | 3.7 MB each | historical plots (`scripts/link_legacy_results.sh`) |
| D2 `data_subset/` | home legacy D2 folder | 326 MB | D2 reruns (builder script not found) |
| HF sentence-transformer cache | `/pscratch/sd/h/hyvchen/foreswe_model_cache` | small | building station prompts offline |
| Real-data tensor cache | `$FSW_CACHE_DIR/planB24_real_data_cache.pt` | 0.42 GB | benchmarks; rebuild with `python -m benchmarks.build_data_cache` (CPU, on a compute node) |
| Python venv | `$FSW_SCRATCH_ROOT/venv-pytorch2.6.0` | small | sentence-transformers / gpytorch / hydroeval; rebuild with `scripts/setup_venv.sh` |

All paths are configurable in `configs/external_paths.env`.

## At-risk artifacts and preservation plan

pscratch purges files not accessed for 8 weeks. Before this work the weekly, Sierra-43 and D2 checkpoints
had last been read on 2026-09-14 (purge-eligible ≈ 2026-11-09). Computing the SHA-256 checksums on
2026-10-26 read every checkpoint, so **all six are now purge-eligible from ≈ 2026-12-21**. Reading a file
only postpones the purge; it does not preserve the file.

The historical env `$SCRATCH/conda_envs/foreswe_repro_att` has already been partially purged.

Recommended (not executed; needs your go-ahead):

```bash
# HPSS tape archive (no quota pressure); verify afterwards with `htar -tvf`
cd /pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra
htar -cvf foreswe_checkpoints_2026-10-26.tar \
  experiments/foreswe_planB_24week/checkpoints experiments/foreswe_weekly_reproduction_full512/checkpoints \
  experiments/foreswe_sierra_station/checkpoints experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km/checkpoints \
  thirdparties/SWE-Forecasting/checkpoints
```

Alternative: weights-only copies (dropping the AdamW state) are 1.6–2.3 GB each (~9.6 GB total), which
would fit in home (40 GB quota, ~15 GB free) but leave little headroom. CFS project `m2637` is 94% full
and shared, so it was not used. `docs/migration/checkpoint_checksums.tsv` records size, mtime and SHA-256
of all six checkpoints (computed 2026-10-26 on a compute node) for verifying any archive copy.

## Legacy files

Nothing in `Snow-Predication-at-Sierra` was modified or deleted. The legacy folders remain as references.
Remaining dependencies on them are data-only (listed above). The legacy third-party checkout
`thirdparties/SWE-Forecasting` is git-ignored by the parent repo, so removing it later would not affect
that repo's history; it can be deleted once you are satisfied with this migration.
