#!/usr/bin/env bash
# One-shot, re-runnable copy of every ForeSWE experiment from the legacy layout into this repo.
# Copies small text/figure/result files verbatim (no edits); everything else (checkpoints,
# .pt tensors, large pickles, data subsets) is left in place and listed in
# docs/migration/external_artifacts_manifest.tsv. Legacy files are never modified or deleted.
#
#   bash scripts/migrate_from_legacy.sh            # from the repo root
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEG_HOME=/global/homes/h/hyvchen/Snow-Predication-at-Sierra
LEG_SCR=/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra
MAX_BYTES=$((1024 * 1024))          # 1 MiB per file
EXT_RE='\.(py|sh|sbatch|save|md|txt|csv|json|log|out|err|png|pdf|pkl|yaml|yml)$'
MANIFEST="$REPO/docs/migration/external_artifacts_manifest.tsv"
COPIED="$REPO/docs/migration/copied_files.tsv"
mkdir -p "$(dirname "$MANIFEST")"
printf "legacy_path\tbytes\tmtime\treason\n" > "$MANIFEST"
printf "legacy_path\trepo_path\tbytes\n" > "$COPIED"

# copy_tree <legacy dir> <repo-relative dest> [prune-regex]
copy_tree() {
  local src="$1" dst="$REPO/$2" prune="${3:-^$}"
  [[ -d "$src" ]] || { echo "skip (missing): $src"; return; }
  while IFS= read -r -d '' f; do
    local rel="${f#"$src"/}"
    if [[ "$rel" =~ $prune || "$rel" == *__pycache__* || "$rel" == *.pyc ]]; then
      continue
    fi
    if [[ -L "$f" ]]; then
      printf "%s\t-\t-\tsymlink -> %s\n" "$f" "$(readlink "$f")" >> "$MANIFEST"; continue
    fi
    local size; size=$(stat -c %s "$f")
    if [[ "$f" =~ $EXT_RE && $size -le $MAX_BYTES ]]; then
      mkdir -p "$(dirname "$dst/$rel")"
      cp -p "$f" "$dst/$rel"
      printf "%s\t%s\t%s\n" "$f" "${dst#"$REPO"/}/$rel" "$size" >> "$COPIED"
    else
      printf "%s\t%s\t%s\t%s\n" "$f" "$size" "$(stat -c %y "$f" | cut -d. -f1)" \
        "$([[ $size -gt $MAX_BYTES ]] && echo 'large (>1 MiB)' || echo 'binary/other type')" >> "$MANIFEST"
    fi
  done < <(find "$src" -mindepth 1 \( -type f -o -type l \) -print0 | sort -z)
}

# ---- engineering audit (legacy third-party checkout, untracked there)
copy_tree "$LEG_HOME/thirdparties/SWE-Forecasting/engineering_audit" engineering_audit
# ---- original daily reproduction (512 stations)
copy_tree "$LEG_HOME/experiments/foreswe_reproduction" experiments/original_daily '^archive_pre_scratch_attention_[0-9_]+/(checkpoints|outputs)/.*\.pt$'
copy_tree "$LEG_SCR/experiments/foreswe_reproduction/logs" experiments/original_daily/logs_pscratch
copy_tree "$LEG_SCR/thirdparties/SWE-Forecasting/outputs" experiments/original_daily/legacy_outputs
copy_tree "$LEG_SCR/thirdparties/SWE-Forecasting/checkpoints" experiments/original_daily/legacy_checkpoints
copy_tree "$LEG_SCR/thirdparties/SWE-Forecasting/results" experiments/original_daily/legacy_results
cp -p "$LEG_HOME/thirdparties/SWE-Forecasting/foreswe_interactive.log" "$REPO/experiments/original_daily/legacy_outputs/"
# ---- original weekly reproduction (512 stations)
copy_tree "$LEG_HOME/experiments/foreswe_weekly_reproduction_full512" experiments/original_weekly
copy_tree "$LEG_SCR/experiments/foreswe_weekly_reproduction_full512" experiments/original_weekly/pscratch
# ---- Plan A (stitched 4-week windows, Fig. 6)
copy_tree "$LEG_HOME/experiments/fig6_weekly_stitching" experiments/planA_stitching
# ---- Plan B (24-week direct head); its only copy lives on pscratch
copy_tree "$LEG_SCR/experiments/foreswe_planB_24week" experiments/planB_24week
# ---- Sierra 43-station subset (source deleted from home; outputs on pscratch)
copy_tree "$LEG_SCR/experiments/foreswe_sierra_station" experiments/sierra_subset/pscratch
copy_tree /pscratch/sd/h/hyvchen/foreswe_sierra_station experiments/sierra_subset/pscratch_alt
# ---- Sierra + 500 km domain sensitivity (D2)
copy_tree "$LEG_HOME/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km" experiments/domain_sensitivity/D2_sierra_plus_500km
copy_tree "$LEG_SCR/experiments/foreswe_domain_sensitivity/D2_sierra_plus_500km" experiments/domain_sensitivity/D2_sierra_plus_500km/pscratch
# ---- bias diagnosis
copy_tree "$LEG_HOME/experiments/foreswe_bias_diagnosis" experiments/bias_diagnosis
# ---- project-level ForeSWE design notes
mkdir -p "$REPO/docs/legacy"
for d in foreswe_memo.md ForeSWE_T0_Architecture_Mapping.md; do
  cp -p "$LEG_HOME/docs/$d" "$REPO/docs/legacy/$d"
  printf "%s\t%s\t%s\n" "$LEG_HOME/docs/$d" "docs/legacy/$d" "$(stat -c %s "$LEG_HOME/docs/$d")" >> "$COPIED"
done

echo "copied: $(($(wc -l < "$COPIED") - 1)) files; external: $(($(wc -l < "$MANIFEST") - 1)) entries"
