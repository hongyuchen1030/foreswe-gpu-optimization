#!/usr/bin/env bash
# SHA-256 + size + mtime of every historical checkpoint in configs/external_paths.env, for verifying
# archive copies. Run on a compute node (reads ~33 GB).
#   bash scripts/slurm/run.sh $JOB bash scripts/checksum_checkpoints.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/configs/external_paths.env"
out="$REPO/docs/migration/checkpoint_checksums.tsv"
printf "name\tpath\tbytes\tmtime\tsha256\n" > "$out"
for v in FSW_CKPT_PLANB FSW_CKPT_DAILY FSW_CKPT_DAILY_PRE_SCRATCH FSW_CKPT_WEEKLY FSW_CKPT_SIERRA43 FSW_CKPT_D2; do
  f="${!v}"
  if [[ -f "$f" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\n" "$v" "$f" "$(stat -c %s "$f")" "$(stat -c %y "$f" | cut -d. -f1)" \
      "$(sha256sum "$f" | cut -d' ' -f1)" >> "$out"
  else
    printf "%s\t%s\t-\t-\tMISSING\n" "$v" "$f" >> "$out"
  fi
done
cat "$out"
